#!/usr/bin/env python3
"""
webui.py - a small local web UI for spike.py (bias-poc v3, remote API only).

What it does:
  * RUN view: pick any of the configured models, build a spike.py command from a form, run it
    as a subprocess and stream its output. One job at a time; a Stop button terminates it.
    Shows an estimated cost before a paid run. Lists existing runs and label CSVs.
  * LABEL view ("ติดป้าย"): judge each direct_question control answer right/wrong in the
    browser (question, expected answer and the full response side by side; blind by default).
    Labels go to the append-only labels table via spike.write_label(); `score` uses them.
  * MODELS view: browse the public OpenRouter catalog (no key needed), pick ANY model, choose
    which upstream provider / quantization to pin, and add it to config/models.yaml. Models
    added here can be removed here again; the two core entries are never touched.

It does NOT change how anything is measured: every action is an ordinary spike.py call and a
model added here is an ordinary config entry, so the CLI and the UI produce identical evidence.

Safety:
  * Binds to 127.0.0.1 only. Requests with any other Host header are refused (blocks
    DNS-rebinding), and every POST (and every outbound-fetching GET) needs a per-start random
    token that only the page served by this process knows (blocks other websites from
    triggering paid runs or config edits).
  * Arguments are validated against allow-lists and passed as an argv list (no shell). Config
    entries are written as JSON scalars and validated with spike.load_models() before the
    file is replaced, so a bad entry can never leave config/models.yaml unloadable.
  * The API key is read by spike.py from the environment you started this server in. This
    server never reads, prints or stores it. Set it before starting:
        export OPENROUTER_API_KEY=...
  * The only outbound requests are GETs to https://openrouter.ai/api/v1/models[...] (public
    catalog; no key is sent).

Usage:
  python3 webui.py                          # then open http://127.0.0.1:8765
  python3 webui.py --port 9000
  python3 webui.py --config path/to/models.yaml   # e.g. a scratch copy for testing

Standard library only (plus pyyaml and spike.py, which are already needed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import date
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import yaml

import spike  # load_models / load_prompts are reused so the UI validates exactly like the CLI

# ----------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent          # project directory; all paths are relative to it
SPIKE = ROOT / "spike.py"
DEFAULT_CONFIG = ROOT / "config" / "models.yaml"
CONFIG = DEFAULT_CONFIG                         # overridden by --config in main()
CONFIG_LOCK = threading.Lock()
DBS = {"real": ROOT / "bias_poc.db", "dry": ROOT / "bias_poc_dry.db"}
TOKEN = secrets.token_urlsafe(24)               # new on every start; embedded in the page only

OPENROUTER_API = "https://openrouter.ai/api/v1"
CATALOG_TTL_S = 600                             # cache the public model list for 10 minutes
API_KEY_ENV = "OPENROUTER_API_KEY"

RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
PROMPT_ID_RE = re.compile(r"^[A-Za-z0-9._#-]{1,120}$")
LABEL_CSV_RE = re.compile(r"^(v3|spike)-[A-Za-z0-9._-]{1,80}\.csv$")  # same patterns as .gitignore
MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}/[A-Za-z0-9._:-]{1,96}$")  # author/slug[:variant]
MODEL_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,47}$")
SLUG_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
MARK_RE = re.compile(r"^  # >>> webui-added (\S+) .*$", re.M)  # marks entries this UI may remove
PRICE_NOTE_RE = re.compile(r"\$([0-9.]+)/M in, \$([0-9.]+)/M out")  # how notes record the price
MAX_OUTPUT_CHARS = 2_000_000                    # keep the in-memory log bounded

# Rough token model for the pre-run cost estimate (see estimate()): Thai text is token-heavy,
# so input is estimated at 1 token per 2 characters; a typical answer (A/B + 2 sentences, or
# a short explanation) at 120 tokens. The "worst case" assumes every reply hits max_tokens.
CHARS_PER_TOKEN = 2.0
TYPICAL_OUTPUT_TOKENS = 120


# ----------------------------------------------------------------------------------------
# One background job (a spike.py subprocess)
# ----------------------------------------------------------------------------------------
class Job:
    """Holds the single running/finished spike.py process and its captured output."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.proc: Optional[subprocess.Popen] = None
        self.argv: list[str] = []
        self.output = ""
        self.exit_code: Optional[int] = None
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, argv: list[str]) -> None:
        with self.lock:
            if self.running():
                raise RuntimeError("มีงานกำลังรันอยู่ รอให้เสร็จหรือกดหยุดก่อน")
            self.argv = argv
            self.output = ""
            self.exit_code = None
            self.started_at = time.time()
            self.finished_at = None
            env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
            self.proc = subprocess.Popen(
                [sys.executable, str(SPIKE), *argv],
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        """Copy the subprocess output into memory line by line until it exits."""
        proc = self.proc
        assert proc is not None and proc.stdout is not None
        for line in proc.stdout:
            with self.lock:
                self.output = (self.output + line)[-MAX_OUTPUT_CHARS:]
        code = proc.wait()
        with self.lock:
            self.exit_code = code
            self.finished_at = time.time()

    def stop(self) -> bool:
        with self.lock:
            if not self.running():
                return False
            assert self.proc is not None
            self.proc.terminate()  # spike.py saves each finished call, so a re-run resumes
            self.output += "\n[webui] ส่งสัญญาณหยุดแล้ว (call ที่เสร็จแล้วถูกบันทึกไว้ รันคำสั่งเดิมซ้ำเพื่อทำต่อได้)\n"
            return True

    def snapshot(self, since: int = 0) -> dict:
        with self.lock:
            return {
                "running": self.running(),
                "argv": self.argv,
                "output": self.output[since:],
                "length": len(self.output),
                "exit_code": self.exit_code,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }


JOB = Job()


class BadRequest(Exception):
    """A request the user can fix; the message is shown in the page."""


# ----------------------------------------------------------------------------------------
# Config file (read + safe edit)
# ----------------------------------------------------------------------------------------
def read_config_text() -> str:
    try:
        return CONFIG.read_text(encoding="utf-8")
    except OSError:
        return ""


def _parse_price_note(note: Optional[str]) -> tuple[Optional[float], Optional[float]]:
    """(input, output) USD per 1M tokens as recorded in a model's note, or (None, None)."""
    m = PRICE_NOTE_RE.search(note or "")
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def load_models() -> list[dict]:
    """Model entries reduced to what the page needs. API keys are never read."""
    try:
        data = yaml.safe_load(read_config_text()) or {}
    except yaml.YAMLError:
        return []
    removable = set(MARK_RE.findall(read_config_text()))
    out = []
    for m in data.get("models", []):
        env_name = m.get("api_key_env")
        p_in, p_out = _parse_price_note(m.get("note"))
        out.append({
            "key": m.get("key"),
            "model": m.get("model"),
            "note": m.get("note"),
            "api_key_env": env_name,
            "key_set": bool(env_name and os.environ.get(env_name)),
            "removable": m.get("key") in removable,
            "price_in": p_in,
            "price_out": p_out,
            "max_tokens": m.get("max_tokens", 600),
        })
    return out


def _validate_and_write(new_text: str) -> None:
    """
    Write `new_text` as the config only if spike.load_models() accepts it (same checks as a
    real run: api_key_env required, no localhost, no duplicate keys, ...). The swap is atomic,
    so a failed or interrupted edit never leaves a broken config behind.
    """
    tmp = CONFIG.with_name(CONFIG.name + ".tmp")
    tmp.write_text(new_text, encoding="utf-8")
    try:
        try:
            spike.load_models(tmp)
        except SystemExit as exc:  # load_models reports problems via sys.exit(message)
            raise BadRequest(f"config ไม่ผ่านการตรวจ ไม่ได้บันทึก: {exc}")
        os.replace(tmp, CONFIG)
    finally:
        if tmp.exists():
            tmp.unlink()


def _q(value: Any) -> str:
    """A YAML scalar/flow value. JSON is valid YAML, so json.dumps quotes strings safely."""
    return json.dumps(value, ensure_ascii=False)


def remove_model(key: str) -> None:
    if not MODEL_KEY_RE.match(key):
        raise BadRequest("key ไม่ถูกรูปแบบ")
    with CONFIG_LOCK:
        text = read_config_text()
        if key not in set(MARK_RE.findall(text)):
            raise BadRequest("ลบได้เฉพาะโมเดลที่เพิ่มผ่านหน้าเว็บนี้ (โมเดลหลักลบไม่ได้)")
        block = re.compile(
            rf"^  # >>> webui-added {re.escape(key)} .*?^  # <<< webui-added {re.escape(key)}[ \t]*\n?",
            re.S | re.M)
        new_text, n = block.subn("", text)
        if n != 1:
            raise BadRequest("หาบล็อกของโมเดลนี้ในไฟล์ไม่เจอ")
        new_text = re.sub(r"\n{3,}", "\n\n", new_text).rstrip("\n") + "\n"
        _validate_and_write(new_text)


# ----------------------------------------------------------------------------------------
# OpenRouter public catalog (read-only, no key)
# ----------------------------------------------------------------------------------------
_catalog_cache: dict = {"ts": 0.0, "models": []}
_catalog_lock = threading.Lock()


def fetch_json(url: str, timeout: float = 20) -> Any:
    """GET a public OpenRouter URL. No credentials are attached."""
    if not url.startswith(OPENROUTER_API + "/"):
        raise RuntimeError("refusing to fetch a non-OpenRouter URL")
    req = urllib.request.Request(url, headers={"User-Agent": "bias-poc-webui/1", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read(8_000_000).decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"OpenRouter ตอบ HTTP {exc.code}")
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise RuntimeError(f"เชื่อมต่อ OpenRouter ไม่ได้: {getattr(exc, 'reason', exc)}")


def _per_million(v: Any) -> Optional[float]:
    """Price strings from OpenRouter are USD per token; the page shows USD per 1M tokens."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f * 1e6, 6) if f >= 0 else None   # negative = router pseudo-models, not real prices


def _slim_model(m: dict) -> Optional[dict]:
    mid = m.get("id") or ""
    if not MODEL_ID_RE.match(mid):
        return None   # aliases like "~vendor/x-latest" move under you: bad for provenance
    if mid.endswith(":batch"):
        return None
    arch = m.get("architecture") or {}
    out_mod = set(arch.get("output_modalities") or ["text"])
    if out_mod - {"text"}:
        return None   # image/audio generators cannot answer these prompts
    pr = m.get("pricing") or {}
    p_in, p_out = _per_million(pr.get("prompt")), _per_million(pr.get("completion"))
    if p_in is None or p_out is None:
        return None
    sp = m.get("supported_parameters") or []
    rz = m.get("reasoning") or {}   # OpenRouter: {"mandatory": bool, "supported_efforts": [...]}
    return {"id": mid, "name": m.get("name") or mid, "in": p_in, "out": p_out,
            "ctx": m.get("context_length"), "seed": "seed" in sp, "reasoning": "reasoning" in sp,
            "reasoning_mandatory": bool(rz.get("mandatory")),
            "efforts": [e for e in (rz.get("supported_efforts") or []) if isinstance(e, str)],
            "free": mid.endswith(":free") or (p_in == 0 and p_out == 0), "created": m.get("created") or 0}


EFFORT_ORDER = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]


def _lowest_effort(efforts: list[str]) -> Optional[str]:
    """Cheapest reasoning effort a model accepts, or None if it does not list any."""
    known = [e for e in EFFORT_ORDER if e in efforts]
    return known[0] if known else None


def catalog_entry(model_id: str) -> Optional[dict]:
    """The catalog row for one model id, or None if the catalog is unavailable / lacks it."""
    try:
        return next((m for m in get_catalog() if m["id"] == model_id), None)
    except RuntimeError:
        return None


def get_catalog() -> list[dict]:
    with _catalog_lock:
        if time.time() - _catalog_cache["ts"] > CATALOG_TTL_S or not _catalog_cache["models"]:
            data = fetch_json(OPENROUTER_API + "/models")
            models = [s for s in (_slim_model(m) for m in data.get("data", [])) if s]
            _catalog_cache.update(ts=time.time(), models=models)
        return _catalog_cache["models"]


# Well-known model makers, in the order the page lists them. Each entry is (label, author
# slugs in the OpenRouter id "author/model"). Curated by hand: which vendors count as "famous"
# is a judgement call and easy to change here. The MODELS shown for each vendor are not
# hard-coded: the newest ones in the live catalog are used, so the list never goes stale.
FAMOUS_VENDORS: list[tuple[str, list[str]]] = [
    ("OpenAI (GPT)", ["openai"]),
    ("Anthropic (Claude)", ["anthropic"]),
    ("Google (Gemini/Gemma)", ["google"]),
    ("Meta (Llama)", ["meta-llama", "meta"]),
    ("Qwen (Alibaba)", ["qwen"]),
    ("DeepSeek", ["deepseek"]),
    ("Mistral", ["mistralai"]),
    ("xAI (Grok)", ["x-ai"]),
    ("Moonshot (Kimi)", ["moonshotai"]),
    ("Z.ai (GLM)", ["z-ai"]),
    ("Cohere", ["cohere"]),
    ("Amazon (Nova)", ["amazon"]),
    ("NVIDIA", ["nvidia"]),
    ("MiniMax", ["minimax"]),
]
_SLUG_GROUP = {slug: (i, label) for i, (label, slugs) in enumerate(FAMOUS_VENDORS) for slug in slugs}
# Specialised (non-chat) models are left out of the curated overview; they stay searchable.
NON_CHAT_RE = re.compile(r"guard|safety|moderation|embed|rerank|whisper|tts|ocr|translat|-mt\b", re.I)
PER_VENDOR_OVERVIEW = 5


def _vendor_of(m: dict) -> str:
    return m["id"].split("/")[0]


def _family_of(m: dict) -> str:
    """Model line, e.g. "gemini-" / "gemma-" / "claude-opus-": the name up to its first digit."""
    return re.split(r"\d", m["id"].split("/", 1)[1], maxsplit=1)[0]


def search_catalog(q: str, sort: str, free_only: bool, seed_only: bool, limit: int,
                   scope: str = "all") -> dict:
    """
    scope = "featured": famous vendors only. With no search text it is a curated overview: the
              newest PER_VENDOR_OVERVIEW chat models of each vendor (one per model line first) (paid variants, unless the
              free filter is on), grouped by vendor.
            "v:<slug>": every model of one famous vendor (slug = first slug of its group).
            "all": the whole catalog.
    """
    words = q.lower().split()
    rows = [m for m in get_catalog()
            if all(w in (m["id"] + " " + m["name"]).lower() for w in words)
            and (not free_only or m["free"]) and (not seed_only or m["seed"])]
    keys = {
        "price": lambda m: (m["out"], m["in"], m["id"]),
        "name": lambda m: m["id"],
        "new": lambda m: (-m["created"], m["id"]),
        "ctx": lambda m: (-(m["ctx"] or 0), m["id"]),
    }
    if scope == "featured":
        rows = [m for m in rows if _vendor_of(m) in _SLUG_GROUP]
        rows.sort(key=lambda m: (_SLUG_GROUP[_vendor_of(m)][0], -m["created"], m["id"]))
        if not words:
            # Per vendor: the newest model of each distinct line first (so Gemini AND Gemma,
            # Llama AND Muse, Opus AND Sonnet all show up), then fill up with the newest others.
            per_group: dict[int, list[dict]] = {}
            for m in rows:   # rows are already newest-first within each vendor group
                if (m["free"] and not free_only) or NON_CHAT_RE.search(m["id"]):
                    continue
                per_group.setdefault(_SLUG_GROUP[_vendor_of(m)][0], []).append(m)
            kept = []
            for g in sorted(per_group):
                cands, picked, lines = per_group[g], [], set()
                for m in cands:
                    if len(picked) < PER_VENDOR_OVERVIEW and _family_of(m) not in lines:
                        lines.add(_family_of(m))
                        picked.append(m)
                for m in cands:
                    if len(picked) < PER_VENDOR_OVERVIEW and m not in picked:
                        picked.append(m)
                kept += sorted(picked, key=lambda m: (-m["created"], m["id"]))
            rows = kept
    else:
        if scope.startswith("v:"):
            slugs = next((sl for _, sl in FAMOUS_VENDORS if sl[0] == scope[2:]), None)
            if slugs is None:
                raise BadRequest("ไม่รู้จักผู้พัฒนานี้")
            rows = [m for m in rows if _vendor_of(m) in slugs]
        rows.sort(key=keys.get(sort, keys["new"]))
    shown = rows if (scope == "featured" and not words) else rows[:limit]
    # Copies: the rows live in the shared catalog cache and must not be modified per request.
    shown = [dict(m, vendor=(_SLUG_GROUP.get(_vendor_of(m)) or (None, _vendor_of(m)))[1]) for m in shown]
    catalog = get_catalog()
    chips = [{"key": sl[0], "label": label,
              "count": sum(1 for m in catalog if _vendor_of(m) in sl)} for label, sl in FAMOUS_VENDORS]
    return {"total": len(rows), "items": shown, "chips": chips,
            "fetched_at": _catalog_cache["ts"], "catalog_size": len(catalog)}


def fetch_endpoints(model_id: str) -> dict:
    """Upstream providers that serve a model, with quantization, price and supported params."""
    if not MODEL_ID_RE.match(model_id):
        raise BadRequest("model id ต้องอยู่ในรูป ผู้พัฒนา/ชื่อรุ่น เช่น qwen/qwen3.5-9b")
    try:
        data = fetch_json(f"{OPENROUTER_API}/models/{model_id}/endpoints")["data"]
    except RuntimeError as exc:
        raise BadRequest(str(exc))
    endpoints = []
    for e in data.get("endpoints", []):
        sp = e.get("supported_parameters") or []
        pr = e.get("pricing") or {}
        tag = e.get("tag") or ""
        slug = tag.split("/")[0] if tag else (e.get("provider_name") or "").lower()
        quant = (e.get("quantization") or "").strip()
        if not SLUG_RE.match(slug):
            continue
        endpoints.append({
            "provider": e.get("provider_name") or slug, "slug": slug, "tag": tag or slug,
            "quant": "" if quant.lower() in ("", "unknown") else quant,
            "in": _per_million(pr.get("prompt")), "out": _per_million(pr.get("completion")),
            "ctx": e.get("context_length"),
            "seed": "seed" in sp, "temperature": "temperature" in sp, "reasoning": "reasoning" in sp,
            # spike.py always sends max_tokens and configs use require_parameters=true, so an
            # endpoint without max_tokens support would reject every call.
            "max_tokens": "max_tokens" in sp,
        })
    endpoints.sort(key=lambda e: ((e["in"] or 0) + (e["out"] or 0), e["tag"]))
    cat = catalog_entry(data.get("id") or model_id) or {}
    return {"id": data.get("id") or model_id, "name": data.get("name") or model_id, "endpoints": endpoints,
            "reasoning_mandatory": bool(cat.get("reasoning_mandatory")),
            "efforts": cat.get("efforts") or []}


# ----------------------------------------------------------------------------------------
# Add a model to the config
# ----------------------------------------------------------------------------------------
def _fmt_price(x: Optional[float]) -> str:
    return "?" if x is None else f"{x:g}"


def add_model(req: dict) -> dict:
    """
    Validate the request against a FRESH endpoint listing (the client is not trusted about
    what a provider supports), build a config entry and append it to config/models.yaml.
    """
    model_id = str(req.get("model", "")).strip()
    key = str(req.get("key", "")).strip()
    if not MODEL_KEY_RE.match(key):
        raise BadRequest("ชื่อ key ใช้ได้เฉพาะ a-z 0-9 . _ - (ขึ้นต้นด้วยตัวอักษรหรือตัวเลข ไม่เกิน 48 ตัว)")
    info = fetch_endpoints(model_id)
    pin = bool(req.get("pin", True))
    ep = None
    if pin:
        ep = next((e for e in info["endpoints"] if e["tag"] == str(req.get("tag", ""))), None)
        if ep is None:
            raise BadRequest("ไม่พบ provider ที่เลือกในรายการล่าสุดของ OpenRouter ลองโหลดใหม่")
    pool = [ep] if ep else info["endpoints"]
    if not pool:
        raise BadRequest("โมเดลนี้ไม่มี provider ให้ใช้งานตอนนี้")
    # Fix: spike.py always sends max_tokens and require_parameters=true routes only to endpoints
    # that support it. Refuse a config that could never be served instead of failing every call.
    if not any(e["max_tokens"] for e in pool):
        raise BadRequest("provider นี้ไม่รับพารามิเตอร์ max_tokens ซึ่ง spike.py ส่งทุกครั้ง "
                         "จึงจะ error ทุก call — เลือก provider อื่น หรือโมเดลอื่น")

    def any_supports(field: str) -> bool:
        return any(e[field] for e in pool)

    # temperature: same value for every model is a design rule; drop it only if unsupported.
    temperature: Optional[float] = None
    raw_t = req.get("temperature")
    if raw_t not in (None, ""):
        try:
            temperature = float(raw_t)
        except (TypeError, ValueError):
            raise BadRequest("temperature ต้องเป็นตัวเลข")
        if not 0 <= temperature <= 2:
            raise BadRequest("temperature ต้องอยู่ระหว่าง 0-2")
    warnings = []
    if temperature is not None and not any_supports("temperature"):
        temperature = None
        warnings.append("provider นี้ไม่รับ temperature จึงไม่ส่งค่า (เทียบกับโมเดลอื่นที่ใช้ 0.7 ตรงๆ ไม่ได้)")
    try:
        max_tokens = int(req.get("max_tokens", 2000))
    except (TypeError, ValueError):
        raise BadRequest("max_tokens ต้องเป็นตัวเลข")
    if not 16 <= max_tokens <= 32000:
        raise BadRequest("max_tokens ต้องอยู่ระหว่าง 16-32000")

    seed_ok = any_supports("seed")
    # Some models cannot switch thinking off (catalog: reasoning.mandatory). Sending
    # reasoning.enabled=false to them is rejected, so use their lowest effort instead.
    mandatory = info["reasoning_mandatory"]
    thinking_off = bool(req.get("thinking_off")) and any_supports("reasoning") and not mandatory
    low_effort = _lowest_effort(info["efforts"]) if mandatory and any_supports("reasoning") else None
    if mandatory:
        warnings.append("โมเดลนี้ปิด thinking ไม่ได้ (บังคับคิดก่อนตอบ) "
                        + (f"จึงตั้ง effort ต่ำสุด = {low_effort} " if low_effort else "")
                        + "token ที่ใช้คิดนับรวมใน max_tokens ถ้าตั้งต่ำไปคำตอบอาจว่าง และเทียบกับโมเดลที่ปิด thinking ได้ไม่ตรงนัก")

    provider_cfg: dict[str, Any] = {"require_parameters": True}
    if ep:
        provider_cfg = {"order": [ep["slug"]], "allow_fallbacks": False, "require_parameters": True}
        if ep["quant"]:
            provider_cfg["quantizations"] = [ep["quant"]]
    extra_body: dict[str, Any] = {}
    if thinking_off:
        extra_body["reasoning"] = {"enabled": False}
    elif low_effort:
        extra_body["reasoning"] = {"effort": low_effort}
    extra_body["provider"] = provider_cfg

    # Price for the note: the pinned endpoint's, or the cheapest listed when not pinned.
    cheapest = min(pool, key=lambda e: (e["in"] or 0) + (e["out"] or 0))
    where = (f"pinned to {ep['provider']} {ep['quant'] or 'quantization unknown'}" if ep
             else "provider NOT pinned (OpenRouter routes; weights may differ between calls)")
    note = (f"OpenRouter {info['id']} {where}; ${_fmt_price(cheapest['in'])}/M in, "
            f"${_fmt_price(cheapest['out'])}/M out (checked {date.today().isoformat()}); "
            f"thinking {'OFF' if thinking_off else ('MANDATORY (effort=' + (low_effort or 'default') + ')') if mandatory else 'default (not disabled)'}"
            f"{'' if seed_ok else '; NO seed support (runs are not reproducible by seed)'}")
    rpm = 15 if info["id"].endswith(":free") else 60   # free tiers are rate limited hard

    lines = [
        f"  # >>> webui-added {key} ({date.today().isoformat()})",
        f"  - key: {_q(key)}",
        "    provider: openai_compatible",
        f"    model: {_q(info['id'])}",
        f"    base_url: {OPENROUTER_API}",
        f"    api_key_env: {API_KEY_ENV}",
        f"    rpm: {rpm}",
        "    concurrency: 2",
        f"    temperature: {_q(temperature)}",
        f"    max_tokens: {max_tokens}",
        "    timeout_s: 180",
    ]
    if seed_ok:
        lines.append("    seed_base: 1000")
    lines += [f"    extra_body: {_q(extra_body)}", f"    note: {_q(note)}",
              f"  # <<< webui-added {key}"]
    block = "\n".join(lines) + "\n"

    with CONFIG_LOCK:
        text = read_config_text()
        try:
            existing = {m.get("key") for m in (yaml.safe_load(text) or {}).get("models", [])}
        except yaml.YAMLError:
            raise BadRequest("อ่านไฟล์ config เดิมไม่ได้ ไม่ได้แก้อะไร")
        if key in existing:
            raise BadRequest(f"key '{key}' มีอยู่แล้ว ใช้ชื่ออื่น")
        _validate_and_write(text.rstrip("\n") + "\n\n" + block)
    return {"key": key, "note": note, "warnings": warnings, "seed": seed_ok, "thinking_off": thinking_off}


# ----------------------------------------------------------------------------------------
# Project state (read-only)
# ----------------------------------------------------------------------------------------
def prompt_files() -> list[dict]:
    """*.jsonl files in the project root with their record counts (comments/blank skipped)."""
    files = []
    for p in sorted(ROOT.glob("*.jsonl")):
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        n = sum(1 for ln in lines if ln.strip() and not ln.strip().startswith("#"))
        files.append({"name": p.name, "count": n})
    return files


def label_csvs() -> list[str]:
    return sorted(p.name for p in ROOT.glob("*.csv") if LABEL_CSV_RE.match(p.name))


def list_runs() -> list[dict]:
    """Runs in both databases with ok/error counts per model (opened read-only)."""
    runs = []
    for kind, path in DBS.items():
        if not path.exists():
            continue
        try:
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            meta = con.execute(
                "SELECT run_id, created_at, n_samples FROM runs ORDER BY created_at DESC").fetchall()
            counts = con.execute(
                "SELECT run_id, model_key, status, COUNT(*) FROM generations "
                "GROUP BY run_id, model_key, status").fetchall()
            con.close()
        except sqlite3.Error:
            continue
        per: dict = {}
        for run_id, model_key, status, n in counts:
            per.setdefault(run_id, {}).setdefault(model_key, {})[status] = n
        for run_id, created_at, n_samples in meta:
            runs.append({"db": kind, "run_id": run_id, "created_at": created_at,
                         "n_samples": n_samples, "models": per.get(run_id, {})})
    return runs


def state() -> dict:
    return {"models": load_models(), "prompt_files": prompt_files(), "label_csvs": label_csvs(),
            "runs": list_runs(), "job": JOB.snapshot(0), "api_key_env": API_KEY_ENV}


# ----------------------------------------------------------------------------------------
# Human labelling ("ติดป้าย"): reads/writes the labels table through spike.py's functions
# ----------------------------------------------------------------------------------------
LABEL_TASKS = {"control": "human_control_correct"}   # task name -> rubric column it records


def _open_label_db(kind: str) -> sqlite3.Connection:
    if kind not in DBS:
        raise BadRequest("db ต้องเป็น real หรือ dry")
    if not DBS[kind].exists():
        raise BadRequest("ยังไม่มีไฟล์ฐานข้อมูลนี้")
    db = spike.open_db(str(DBS[kind]))   # creates the labels table on first use
    db.execute("PRAGMA busy_timeout = 10000")   # a spike.py run may be writing at the same time
    return db


def _blind_order(run_id: str, gen_id: int) -> str:
    """Fixed pseudo-random order per run, so both models' answers are interleaved (blind mode)
    and the order is the same every time the page is opened."""
    return hashlib.sha256(f"{run_id}:{gen_id}".encode()).hexdigest()


def label_items(kind: str, run_id: str, task: str) -> dict:
    if not RUN_ID_RE.match(run_id or ""):
        raise BadRequest("run id ไม่ถูกรูปแบบ")
    field = LABEL_TASKS.get(task)
    if field is None:
        raise BadRequest("ไม่รู้จักงานติดป้ายนี้")
    db = _open_label_db(kind)
    try:
        current = spike.current_labels(db, run_id)
        counts = {(p, m, i): n for p, m, i, n in db.execute(
            "SELECT prompt_id, model_key, sample_idx, COUNT(*) FROM labels WHERE run_id=? "
            "GROUP BY prompt_id, model_key, sample_idx", (run_id,))}
        rows = db.execute(
            "SELECT id, prompt_id, model_key, sample_idx, status, response_text, prompt_json "
            "FROM generations WHERE run_id=?", (run_id,)).fetchall()
    finally:
        db.close()
    items = []
    for gid, prompt_id, model_key, idx, status, text, pj in rows:
        if status != "ok":
            continue
        prompt = json.loads(pj)
        analysed = spike.analyse_generation(prompt, status, text)
        if not spike.label_applies(field, prompt, analysed):
            continue
        lab = current.get((prompt_id, model_key, idx), {})
        items.append({
            "gen_id": gid, "prompt_id": prompt_id, "model_key": model_key, "sample_idx": idx,
            "question": prompt.get("text", ""), "expected": prompt.get("expected_answer"),
            "response": text or "", "hint": analysed.get("control_auto_correct"),
            "label": lab.get(field), "note": lab.get("human_notes", ""),
            "n_history": counts.get((prompt_id, model_key, idx), 0),
            "_order": _blind_order(run_id, gid),
        })
    items.sort(key=lambda x: x.pop("_order"))
    return {"run_id": run_id, "db": kind, "field": field, "items": items}


def _generation_key(db: sqlite3.Connection, run_id: str, gen_id: Any) -> tuple:
    try:
        gid = int(gen_id)
    except (TypeError, ValueError):
        raise BadRequest("gen_id ไม่ถูกต้อง")
    row = db.execute("SELECT prompt_id, model_key, sample_idx FROM generations WHERE id=? AND run_id=?",
                     (gid, run_id)).fetchone()
    if row is None:
        raise BadRequest("ไม่พบคำตอบนี้ใน run ที่เลือก")
    return row


def set_label(req: dict) -> dict:
    run_id = str(req.get("run_id", ""))
    if not RUN_ID_RE.match(run_id):
        raise BadRequest("run id ไม่ถูกรูปแบบ")
    field = str(req.get("field", ""))
    if field not in set(LABEL_TASKS.values()) | {"human_notes"}:
        raise BadRequest("ไม่รับป้ายชนิดนี้จากหน้าเว็บ")
    value = req.get("value")
    if value is not None and not isinstance(value, str):
        raise BadRequest("ค่าป้ายต้องเป็นข้อความ")
    if isinstance(value, str) and len(value) > 2000:
        raise BadRequest("บันทึกยาวเกิน 2000 ตัวอักษร")
    db = _open_label_db(str(req.get("db", "")))
    try:
        p, m, i = _generation_key(db, run_id, req.get("gen_id"))
        try:
            current = spike.write_label(db, run_id, p, m, i, field, value, "web")
        except ValueError as exc:   # spike's validation: allowed values, applies_to, failed call
            raise BadRequest(str(exc))
        n = db.execute("SELECT COUNT(*) FROM labels WHERE run_id=? AND prompt_id=? AND model_key=? "
                       "AND sample_idx=?", (run_id, p, m, i)).fetchone()[0]
    finally:
        db.close()
    return {"value": current, "n_history": n}


def get_history(kind: str, run_id: str, gen_id: str) -> list:
    if not RUN_ID_RE.match(run_id or ""):
        raise BadRequest("run id ไม่ถูกรูปแบบ")
    db = _open_label_db(kind)
    try:
        p, m, i = _generation_key(db, run_id, gen_id)
        return spike.label_history(db, run_id, p, m, i)
    finally:
        db.close()


# ----------------------------------------------------------------------------------------
# Form -> spike.py argv (validated)
# ----------------------------------------------------------------------------------------
def _run_id(req: dict) -> str:
    rid = str(req.get("run_id", "")).strip()
    if not rid:
        raise BadRequest("ยังไม่ได้กรอก run id (ช่องเทาๆ ที่เห็นคือตัวอย่าง ต้องพิมพ์เอง เช่น v3-or-ctrl-001)")
    if not RUN_ID_RE.match(rid):
        bad = sorted({c for c in rid if not re.match(r"[A-Za-z0-9._-]", c)})
        detail = f"ตัวอักษรที่ใช้ไม่ได้: {' '.join(repr(c) for c in bad)}" if bad else "ยาวเกิน 64 ตัว"
        raise BadRequest(f"run id ใช้ได้เฉพาะ A-Z a-z 0-9 . _ - (ไม่เกิน 64 ตัว) — {detail}")
    return rid


def _prompts(req: dict) -> str:
    name = str(req.get("prompts", ""))
    if name not in {f["name"] for f in prompt_files()}:
        raise BadRequest(f"ไม่พบไฟล์ prompt: {name}")
    return name


def _model_keys(req: dict) -> list[str]:
    valid = {m["key"] for m in load_models()}
    keys = [str(k) for k in (req.get("models") or [])]
    bad = [k for k in keys if k not in valid]
    if bad:
        raise BadRequest(f"ไม่รู้จัก model: {', '.join(bad)}")
    return keys


def _int(req: dict, name: str, lo: int, hi: int, default: int) -> int:
    try:
        v = int(req.get(name, default))
    except (TypeError, ValueError):
        raise BadRequest(f"{name} ต้องเป็นตัวเลข")
    if not lo <= v <= hi:
        raise BadRequest(f"{name} ต้องอยู่ระหว่าง {lo}-{hi}")
    return v


def export_name(run_id: str) -> str:
    """CSV file name that the .gitignore patterns (v3-*.csv / spike-*.csv) already cover."""
    return f"{run_id}.csv" if run_id.startswith(("v3-", "spike-")) else f"v3-{run_id}.csv"


def config_args() -> list[str]:
    """--config only when a non-default file is in use (keeps the shown command clean)."""
    return [] if CONFIG == DEFAULT_CONFIG else ["--config", str(CONFIG)]


def build_argv(req: dict) -> list[str]:
    cmd = req.get("command")
    dry = bool(req.get("dry_run"))
    common = lambda rid: ["--run-id", rid] + (["--dry-run"] if dry else [])  # noqa: E731

    if cmd == "selftest":
        return ["selftest"]

    if cmd == "run":
        rid = _run_id(req)
        prompts = _prompts(req)
        models = _model_keys(req)
        if not models:
            raise BadRequest("เลือกอย่างน้อย 1 โมเดล")
        samples = _int(req, "samples", 1, 20, 1)
        max_calls = _int(req, "max_calls", 1, 100_000, 200)
        # spike.py asks on stdin before paid calls; the page asks instead and sends
        # confirm_cost=true, so --yes is only added after that confirmation.
        if not dry and req.get("confirm_cost") is not True:
            raise BadRequest("ต้องยืนยันค่าใช้จ่ายก่อนรันจริง")
        argv = ["run", *common(rid), *config_args(), "--prompts", prompts,
                "--samples", str(samples), "--max-calls", str(max_calls), "--yes"]
        for k in models:
            argv += ["--model", k]
        return argv

    if cmd in ("summary", "score"):
        argv = [cmd, *common(_run_id(req))]
        labels = str(req.get("labels") or "")
        if cmd == "score" and labels:
            if labels not in label_csvs():
                raise BadRequest(f"ไม่พบไฟล์ label: {labels}")
            argv += ["--labels", labels]
        return argv

    if cmd == "show":
        argv = ["show", *common(_run_id(req))]
        model = str(req.get("model") or "")
        if model:
            argv += ["--model", _model_keys({"models": [model]})[0]]
        prompt = str(req.get("prompt") or "").strip()
        if prompt:
            if not PROMPT_ID_RE.match(prompt):
                raise BadRequest("prompt id ไม่ถูกรูปแบบ")
            argv += ["--prompt", prompt]
        return argv

    if cmd == "export-csv":
        rid = _run_id(req)
        return ["export-csv", *common(rid), "--out", export_name(rid)]

    if cmd == "repro-check":
        model = str(req.get("model") or "")
        if not model:
            raise BadRequest("เลือก model ก่อน")
        argv = ["repro-check", *config_args(), "--model", _model_keys({"models": [model]})[0],
                "--prompts", _prompts(req), "--repeats", str(_int(req, "repeats", 2, 5, 2))]
        prompt = str(req.get("prompt") or "").strip()
        if prompt:
            if not PROMPT_ID_RE.match(prompt):
                raise BadRequest("prompt id ไม่ถูกรูปแบบ")
            argv += ["--prompt", prompt]
        return argv

    raise BadRequest(f"ไม่รู้จักคำสั่ง: {cmd}")


def estimate(req: dict) -> dict:
    """
    Rough USD cost of a run BEFORE it starts. Prices come from the "$x/M in, $y/M out" text in
    each model's note (written when the model was added; the two core entries have it too).
    "typical" assumes short answers; "worst" assumes every reply runs to max_tokens. Both are
    estimates: token counts are guessed from character counts, not measured.
    """
    keys = _model_keys(req)
    name = _prompts(req)
    samples = _int(req, "samples", 1, 20, 1)
    try:
        prompts = spike.load_prompts(ROOT / name)
    except SystemExit as exc:
        raise BadRequest(str(exc))
    in_tokens = sum(math.ceil(len(p["text"]) / CHARS_PER_TOKEN) for p in prompts)
    by_key = {m["key"]: m for m in load_models()}
    rows, total_typ, total_max, unknown = [], 0.0, 0.0, []
    for k in keys:
        m = by_key[k]
        calls = len(prompts) * samples
        row = {"key": k, "calls": calls, "typical": None, "worst": None}
        if m["price_in"] is None or m["price_out"] is None:
            unknown.append(k)
        else:
            tin = in_tokens * samples
            row["typical"] = (tin * m["price_in"] + calls * TYPICAL_OUTPUT_TOKENS * m["price_out"]) / 1e6
            row["worst"] = (tin * m["price_in"] + calls * int(m["max_tokens"]) * m["price_out"]) / 1e6
            total_typ += row["typical"]
            total_max += row["worst"]
        rows.append(row)
    return {"rows": rows, "typical": total_typ, "worst": total_max, "unknown_price": unknown,
            "calls": sum(r["calls"] for r in rows)}


# ----------------------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "bias-poc-webui"

    def log_message(self, fmt: str, *args: Any) -> None:  # keep the terminal quiet
        pass

    # --- helpers ---
    def _host_ok(self) -> bool:
        port = self.server.server_address[1]
        return self.headers.get("Host", "") in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _token_ok(self) -> bool:
        return secrets.compare_digest(self.headers.get("X-Token", ""), TOKEN)

    def _send(self, status: int, body: bytes, ctype: str, extra: Optional[dict] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, status: int = 200) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    # --- routes ---
    def do_GET(self) -> None:
        if not self._host_ok():
            return self._send(HTTPStatus.FORBIDDEN, b"bad host", "text/plain")
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        one = lambda name, default="": (qs.get(name) or [default])[0]  # noqa: E731
        if url.path == "/":
            html = PAGE.replace("__TOKEN__", TOKEN).encode("utf-8")
            return self._send(200, html, "text/html; charset=utf-8")
        if url.path == "/api/state":
            return self._json(state())
        if url.path == "/api/job":
            since = int(one("since", "0") or 0)
            return self._json(JOB.snapshot(max(0, since)))
        if url.path in ("/api/label/items", "/api/label/history"):
            if not self._token_ok():
                return self._json({"error": "token ไม่ถูกต้อง (รีเฟรชหน้าเว็บ)"}, 403)
            try:
                if url.path == "/api/label/items":
                    return self._json(label_items(one("db"), one("run_id"), one("task", "control")))
                return self._json(get_history(one("db"), one("run_id"), one("gen_id")))
            except BadRequest as exc:
                return self._json({"error": str(exc)}, 400)
        if url.path in ("/api/catalog", "/api/endpoints"):
            # These make outbound requests, so they need the token like a POST does.
            if not self._token_ok():
                return self._json({"error": "token ไม่ถูกต้อง (รีเฟรชหน้าเว็บ)"}, 403)
            try:
                if url.path == "/api/catalog":
                    limit = max(1, min(int(one("limit", "60") or 60), 200))
                    return self._json(search_catalog(one("q"), one("sort", "new"),
                                                     one("free") == "1", one("seed") == "1", limit,
                                                     one("scope", "featured")))
                return self._json(fetch_endpoints(one("id").strip()))
            except BadRequest as exc:
                return self._json({"error": str(exc)}, 400)
            except (RuntimeError, ValueError) as exc:
                return self._json({"error": str(exc)}, 502)
        if url.path == "/download":
            # GET is fine without the token: it only reads a git-ignored label CSV, and other
            # origins cannot read the response. The name must match the allow-list exactly.
            name = one("file")
            if name not in label_csvs():
                return self._send(404, b"not found", "text/plain")
            data = (ROOT / name).read_bytes()
            return self._send(200, data, "text/csv; charset=utf-8",
                              {"Content-Disposition": f'attachment; filename="{name}"'})
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if not self._host_ok():
            return self._send(HTTPStatus.FORBIDDEN, b"bad host", "text/plain")
        if not self._token_ok():
            return self._json({"error": "token ไม่ถูกต้อง (รีเฟรชหน้าเว็บ)"}, 403)
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 100_000)
            req = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._json({"error": "JSON ไม่ถูกต้อง"}, 400)
        path = urlparse(self.path).path
        try:
            if path == "/api/job":
                argv = build_argv(req)
                JOB.start(argv)
                return self._json({"ok": True, "argv": argv})
            if path == "/api/job/stop":
                return self._json({"stopped": JOB.stop()})
            if path == "/api/estimate":
                return self._json(estimate(req))
            if path == "/api/models/add":
                return self._json(add_model(req))
            if path == "/api/label/set":
                return self._json(set_label(req))
            if path == "/api/models/remove":
                remove_model(str(req.get("key", "")))
                return self._json({"ok": True})
        except BadRequest as exc:
            return self._json({"error": str(exc)}, 400)
        except RuntimeError as exc:
            return self._json({"error": str(exc)}, 409)
        self._send(404, b"not found", "text/plain")


# ----------------------------------------------------------------------------------------
# Page (single file; no external resources)
# ----------------------------------------------------------------------------------------
PAGE = r"""<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bias POC Runner</title>
<style>
:root {
  --bg: #f6f5f2; --panel: #ffffff; --ink: #1d1d1b; --muted: #6b6a66; --line: #e3e1db;
  --accent: #2f5d8a; --accent-ink: #ffffff; --ok: #2e7d4f; --warn: #a15c00; --bad: #b3261e;
  --code-bg: #1e1f22; --code-ink: #e6e6e3; --hover: #f0eee8;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #161615; --panel: #1f1f1d; --ink: #ecebe7; --muted: #a09e98; --line: #34332f;
    --accent: #7fb0e0; --accent-ink: #10202f; --ok: #6cc28f; --warn: #e0a24a; --bad: #f08a80;
    --code-bg: #0f0f0e; --code-ink: #dcdcd8; --hover: #2a2926;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Sukhumvit Set", "Thonburi", "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 20px 16px 60px; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 0 0 12px; }
.sub { color: var(--muted); font-size: 13px; margin: 0 0 14px; }
.grid { display: grid; gap: 16px; grid-template-columns: 1fr; }
@media (min-width: 900px) { .grid { grid-template-columns: 380px 1fr; } }
@media (min-width: 900px) { .grid.wide { grid-template-columns: 1.15fr 1fr; } }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px; min-width: 0; }
.stack > * + * { margin-top: 16px; }
label { display: block; font-size: 13px; color: var(--muted); margin: 10px 0 4px; }
input[type=text], input[type=number], input[type=search], select {
  width: 100%; padding: 7px 9px; border: 1px solid var(--line); border-radius: 7px;
  background: var(--bg); color: var(--ink); font: inherit; }
.check { display: flex; align-items: center; gap: 8px; color: var(--ink); margin: 6px 0; font-size: 14px; }
.check label { margin: 0; color: inherit; }
.tabs { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
.tabs button { background: transparent; color: var(--ink); border: 1px solid var(--line); }
.tabs button.on { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
.nav { display: flex; gap: 8px; margin: 0 0 16px; }
.nav button { padding: 8px 18px; }
button { padding: 7px 14px; border: 0; border-radius: 7px; font: inherit; cursor: pointer;
  background: var(--accent); color: var(--accent-ink); }
button.ghost { background: transparent; color: var(--ink); border: 1px solid var(--line); }
button.danger { background: var(--bad); color: #fff; }
button.small { padding: 2px 9px; font-size: 12.5px; }
button:disabled { opacity: .45; cursor: default; }
.row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-top: 14px; }
.between { justify-content: space-between; }
.pill { display: inline-block; font-size: 12px; padding: 1px 8px; border-radius: 99px; border: 1px solid currentColor; }
.ok { color: var(--ok); } .warn { color: var(--warn); } .bad { color: var(--bad); } .muted { color: var(--muted); }
.hint { font-size: 12.5px; color: var(--muted); margin-top: 8px; }
.box { border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; font-size: 13px; margin-top: 10px; }
.box.warnbox { border-color: var(--warn); }
table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
th, td { text-align: left; padding: 7px 6px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 500; }
td.actions { white-space: nowrap; }
td.actions button { padding: 3px 9px; font-size: 12.5px; }
td.num, th.num { text-align: right; white-space: nowrap; }
tr.pick { cursor: pointer; }
tr.pick:hover { background: var(--hover); }
tr.sel { background: var(--hover); outline: 2px solid var(--accent); outline-offset: -2px; }
.scroll { overflow-x: auto; }
.list { max-height: 520px; overflow: auto; }
pre#out { background: var(--code-bg); color: var(--code-ink); margin: 0; padding: 12px; border-radius: 8px;
  height: 420px; overflow: auto; font: 12.5px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace;
  white-space: pre-wrap; word-break: break-word; }
.cmdline { font: 12px ui-monospace, Menlo, monospace; color: var(--muted); word-break: break-all; }
.models div.m { margin: 8px 0; font-size: 13.5px; }
.tag { display: inline-block; font-size: 11.5px; padding: 0 6px; border-radius: 5px; border: 1px solid var(--line); color: var(--muted); margin-right: 3px; }
.tag.g { color: var(--ok); border-color: var(--ok); }
.mono { font: 12.5px ui-monospace, Menlo, monospace; word-break: break-all; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.chips button { padding: 3px 10px; font-size: 12.5px; background: transparent; color: var(--ink); border: 1px solid var(--line); }
.chips button.on { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
tr.grp td { background: var(--hover); font-weight: 600; font-size: 12.5px; padding: 5px 6px; position: sticky; top: 0; }
.filters { display: flex; gap: 14px; flex-wrap: wrap; align-items: center; margin-top: 8px; }
.filters .check { margin: 0; }
.filters select { width: auto; }
.lbar { display: grid; gap: 12px; grid-template-columns: 1fr; }
@media (min-width: 760px) { .lbar { grid-template-columns: 1.4fr 1fr 1.3fr; } }
.progress { height: 6px; background: var(--line); border-radius: 99px; margin-top: 14px; overflow: hidden; }
.progress div { height: 100%; width: 0; background: var(--ok); transition: width .2s; }
.qa { border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; margin-top: 12px; }
.qa.expected { border-color: var(--ok); }
.qlabel { font-size: 12px; color: var(--muted); margin-bottom: 4px; }
.qtext { white-space: pre-wrap; word-break: break-word; font-size: 14.5px; }
.qtext.resp { max-height: 420px; overflow: auto; }
.okbtn { background: var(--ok); color: #fff; }
.badbtn { background: var(--bad); color: #fff; }
.kbd { display: inline-block; font: 11px ui-monospace, Menlo, monospace; border: 1px solid currentColor; border-radius: 4px; padding: 0 4px; margin-left: 4px; opacity: .8; }
textarea { width: 100%; padding: 7px 9px; border: 1px solid var(--line); border-radius: 7px; background: var(--bg); color: var(--ink); font: inherit; resize: vertical; }
[hidden] { display: none !important; }
</style>
</head>
<body>
<main>
  <h1>Bias POC Runner</h1>
  <p class="sub">หน้านี้เรียก <code>spike.py</code> บนเครื่องนี้ ทุกปุ่มเท่ากับรันคำสั่งเดียวกับ CLI ผลที่ได้จึงเหมือนกันทุกประการ</p>

  <div class="nav" id="nav">
    <button data-view="run" class="on">รันทดสอบ</button>
    <button data-view="label" class="ghost">ติดป้าย</button>
    <button data-view="models" class="ghost">เลือกโมเดล</button>
  </div>

  <!-- ============================ RUN VIEW ============================ -->
  <div id="view_run" class="grid">
    <div class="stack">
      <section class="card">
        <div class="row between" style="margin-top:0">
          <h2 style="margin:0">โมเดลใน config</h2>
          <button class="ghost small" id="go_models">+ เพิ่มโมเดล</button>
        </div>
        <div id="models" class="models muted" style="margin-top:8px">กำลังโหลด…</div>
      </section>

      <section class="card">
        <div class="tabs" id="tabs">
          <button data-cmd="run" class="on">รัน</button>
          <button data-cmd="score">Score</button>
          <button data-cmd="summary">Summary</button>
          <button data-cmd="show">ดูคำตอบ</button>
          <button data-cmd="export-csv">Export CSV</button>
          <button data-cmd="repro-check">Repro</button>
          <button data-cmd="selftest">Selftest</button>
        </div>

        <form id="form" autocomplete="off">
          <div data-for="run score summary show export-csv">
            <label for="run_id">Run id</label>
            <input type="text" id="run_id" placeholder="เช่น v3-or-ctrl-001" list="run_ids">
            <datalist id="run_ids"></datalist>
          </div>
          <div data-for="run repro-check">
            <label for="prompts">ไฟล์ prompt</label>
            <select id="prompts"></select>
          </div>
          <div data-for="run">
            <label>โมเดลที่จะรัน</label>
            <div id="model_checks"></div>
            <label for="samples">Samples ต่อ prompt ต่อโมเดล</label>
            <input type="number" id="samples" min="1" max="20" value="1">
          </div>
          <div data-for="show repro-check">
            <label for="model_one">โมเดล</label>
            <select id="model_one"></select>
            <label for="prompt_id">Prompt id <span class="muted">(ไม่ใส่ = ทั้งหมด / ข้อแรก)</span></label>
            <input type="text" id="prompt_id" placeholder="เช่น CTRL_fr_license">
          </div>
          <div data-for="repro-check">
            <label for="repeats">จำนวนครั้งที่เรียกซ้ำ</label>
            <input type="number" id="repeats" min="2" max="5" value="2">
          </div>
          <div data-for="score">
            <label for="labels">ไฟล์ label CSV เพิ่มเติม <span class="muted">(ไม่บังคับ — ป้ายจากแท็บติดป้ายใช้อัตโนมัติ)</span></label>
            <select id="labels"></select>
          </div>
          <div data-for="run score summary show export-csv">
            <div class="check"><input type="checkbox" id="dry_run"><label for="dry_run">Dry run (โมเดลจำลอง, ไม่เสียเงิน, ใช้ bias_poc_dry.db)</label></div>
          </div>
          <p class="hint" id="cmd_hint"></p>
          <div id="estimate" class="box" hidden></div>
          <div class="row">
            <button type="submit" id="go">เริ่ม</button>
          </div>
        </form>
      </section>
    </div>

    <div class="stack">
      <section class="card">
        <div class="row between" style="margin-top:0">
          <h2 style="margin:0">ผลลัพธ์</h2>
          <div class="row" style="margin-top:0">
            <span id="job_status" class="pill muted">ว่าง</span>
            <button class="danger" id="stop" disabled>หยุด</button>
          </div>
        </div>
        <p class="cmdline" id="job_cmd"></p>
        <pre id="out"></pre>
      </section>

      <section class="card">
        <div class="row between" style="margin-top:0">
          <h2 style="margin:0">Runs ในฐานข้อมูล</h2>
          <button class="ghost" id="refresh">รีเฟรช</button>
        </div>
        <div class="scroll"><table>
          <thead><tr><th>Run id</th><th>DB</th><th>โมเดล: ok / error</th><th></th></tr></thead>
          <tbody id="runs"></tbody>
        </table></div>
        <div id="csvs" class="hint"></div>
      </section>
    </div>
  </div>

  <!-- ============================ LABEL VIEW ============================ -->
  <div id="view_label" hidden>
    <section class="card">
      <div class="lbar">
        <div><label for="l_run">Run</label><select id="l_run"></select></div>
        <div><label for="l_mode">แสดง</label>
          <select id="l_mode"><option value="todo">ข้อที่ยังไม่ติดป้ายก่อน</option><option value="all">ทุกข้อตามลำดับ</option></select></div>
        <div class="check" style="margin-top:26px"><input type="checkbox" id="l_blind" checked><label for="l_blind">โหมดปิดตา (ซ่อนชื่อโมเดลและคำใบ้)</label></div>
      </div>
      <div class="progress"><div id="l_bar"></div></div>
      <p class="hint" id="l_info">เลือก run เพื่อเริ่ม ติดป้ายเฉพาะข้อสอบควบคุมแบบถามตรง (ข้อ A/B ระบบตรวจเองได้)</p>
    </section>

    <section class="card" id="l_card" hidden style="margin-top:16px">
      <div class="row between" style="margin-top:0">
        <b id="l_pos"></b>
        <span id="l_who" class="muted"></span>
      </div>
      <div class="qa"><div class="qlabel">คำถามที่ส่งให้โมเดล</div><div id="l_q" class="qtext"></div></div>
      <div class="qa expected"><div class="qlabel">เฉลย (expected_answer)</div><div id="l_exp" class="qtext"></div></div>
      <div class="qa"><div class="qlabel">คำตอบของโมเดล (เต็ม)</div><div id="l_resp" class="qtext resp"></div></div>
      <div class="row">
        <button class="okbtn" id="l_y">✓ ถูก <span class="kbd">Y</span></button>
        <button class="badbtn" id="l_n">✗ ผิด <span class="kbd">N</span></button>
        <button class="ghost" id="l_prev">← ก่อนหน้า</button>
        <button class="ghost" id="l_next">ข้าม →</button>
        <span id="l_cur" class="pill muted">ยังไม่ติดป้าย</span>
        <button class="ghost small" id="l_clear" hidden>ล้างป้าย</button>
      </div>
      <label for="l_note">บันทึก (ไม่บังคับ — บันทึกอัตโนมัติเมื่อออกจากช่อง)</label>
      <textarea id="l_note" rows="2" placeholder="เช่น ตอบถูก แต่แต่งรายละเอียดผิด"></textarea>
      <div class="row">
        <button class="ghost small" id="l_hist_btn">ดูประวัติป้ายของข้อนี้</button>
        <span class="hint" id="l_msg"></span>
      </div>
      <div id="l_hist" class="hint" hidden></div>
    </section>

    <section class="card" id="l_done" hidden style="margin-top:16px">
      <b>ติดป้ายครบทุกข้อแล้ว</b>
      <p class="hint">score จะใช้ป้ายเหล่านี้อัตโนมัติ ไม่ต้องเลือกไฟล์ CSV</p>
      <div class="row"><button id="l_score">Score ด้วยป้ายนี้</button><button class="ghost" id="l_review">ทบทวนทุกข้ออีกครั้ง</button></div>
    </section>
  </div>

  <!-- ============================ MODELS VIEW ============================ -->
  <div id="view_models" class="grid wide" hidden>
    <section class="card">
      <h2>ค้นหาโมเดลจาก OpenRouter</h2>
      <div class="chips" id="chips"></div>
      <input type="search" id="q" placeholder="พิมพ์ค้นหา เช่น qwen, gemma, claude, llama, gpt">
      <div class="filters">
        <select id="sort">
          <option value="new">ใหม่ก่อน</option>
          <option value="price">ราคาถูกก่อน</option>
          <option value="name">ชื่อ A-Z</option>
          <option value="ctx">context ยาวก่อน</option>
        </select>
        <div class="check"><input type="checkbox" id="f_free"><label for="f_free">ฟรีเท่านั้น</label></div>
        <div class="check"><input type="checkbox" id="f_seed"><label for="f_seed">รองรับ seed</label></div>
      </div>
      <p class="hint" id="cat_info">กำลังโหลดรายการ…</p>
      <div class="list scroll"><table>
        <thead><tr><th>โมเดล</th><th class="num">เข้า $/M</th><th class="num">ออก $/M</th><th class="num">context</th></tr></thead>
        <tbody id="cat_rows"></tbody>
      </table></div>
      <label for="manual_id">หรือพิมพ์ model id เอง <span class="muted">(รูป ผู้พัฒนา/ชื่อรุ่น)</span></label>
      <div class="row" style="margin-top:0">
        <input type="text" id="manual_id" placeholder="เช่น qwen/qwen3.5-9b" style="flex:1">
        <button class="ghost" id="manual_go">โหลด</button>
      </div>
    </section>

    <section class="card" id="detail">
      <h2>ตั้งค่าโมเดล</h2>
      <div id="detail_body" class="muted">เลือกโมเดลจากรายการทางซ้าย</div>
    </section>
  </div>
</main>

<script>
const TOKEN = "__TOKEN__";
let S = null;                 // last /api/state
let cmd = "run";              // selected command tab
let outLen = 0;               // chars of job output already shown
let polling = null;
const SEL = new Set();        // model keys ticked in the run form (kept across refreshes)
let CAT = {loaded: false, scope: "featured"};    // catalog view state (scope: featured | all | v:<slug>)
let D = null;                 // model shown in the detail panel: {id, name, endpoints}
let VIEW = "run";
let L = null;                 // label view: {db, run_id, field, items, i}
let LSEEN = new Set();        // blind mode: items whose model name the user chose to reveal

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const price = (x) => x == null ? "?" : x === 0 ? "ฟรี" : "$" + (x < 0.01 ? x.toFixed(4) : x < 1 ? x.toFixed(3) : x.toFixed(2));
const usd = (x) => x == null ? "?" : x < 0.01 ? "<$0.01" : "$" + x.toFixed(2);
const ctxTxt = (c) => c ? (c >= 1e6 ? (c / 1e6).toFixed(1) + "M" : Math.round(c / 1000) + "k") : "?";

async function api(path, body) {
  const opt = {headers: {"X-Token": TOKEN}};
  if (body !== undefined) {
    opt.method = "POST";
    opt.headers["Content-Type"] = "application/json";
    opt.body = JSON.stringify(body);
  }
  const r = await fetch(path, opt);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || ("HTTP " + r.status));
  return j;
}

/* ------------------------------- views ------------------------------- */
function setView(v) {
  $("view_run").hidden = v !== "run";
  $("view_models").hidden = v !== "models";
  $("view_label").hidden = v !== "label";
  VIEW = v;
  if (v === "label") fillLabelRuns();
  document.querySelectorAll("#nav button").forEach(b => b.className = b.dataset.view === v ? "" : "ghost");
  if (v === "models" && !CAT.loaded) loadCatalog();
}

/* ------------------------------- run view ------------------------------- */
function renderModels() {
  $("models").innerHTML = S.models.length ? S.models.map(m =>
    `<div class="m"><b>${esc(m.key)}</b> <span class="muted">${esc(m.model)}</span>
     ${m.price_in != null ? `<span class="muted"> · ${price(m.price_in)} / ${price(m.price_out)} ต่อล้าน token</span>` : ""}<br>
     ${m.key_set ? `<span class="pill ok">${esc(m.api_key_env)} ตั้งแล้ว</span>`
                 : `<span class="pill bad">${esc(m.api_key_env || "ไม่มี api_key_env")} ยังไม่ได้ตั้ง</span>`}
     ${m.removable ? `<button class="ghost small" data-remove="${esc(m.key)}">ลบ</button>` : ""}</div>`
  ).join("") : `<span class="bad">อ่าน config ไม่ได้ หรือยังไม่มีโมเดล</span>`;
}

let selInit = false;
function renderModelChecks() {
  if (!selInit) {   // first load: tick the core models (not the ones added here), as before
    S.models.filter(m => !m.removable).forEach(m => SEL.add(m.key));
    selInit = true;
  }
  for (const k of [...SEL]) if (!S.models.some(m => m.key === k)) SEL.delete(k);   // removed models
  $("model_checks").innerHTML = S.models.length ? S.models.map(m =>
    `<div class="check"><input type="checkbox" name="mk" value="${esc(m.key)}" id="mk_${esc(m.key)}" ${SEL.has(m.key) ? "checked" : ""}>
     <label for="mk_${esc(m.key)}">${esc(m.key)} <span class="muted">${m.price_in != null ? price(m.price_in) + " / " + price(m.price_out) : ""}</span></label></div>`
  ).join("") : `<span class="muted">ยังไม่มีโมเดล — กด "+ เพิ่มโมเดล"</span>`;
}

function renderState() {
  renderModels();
  renderModelChecks();
  const keep = (id) => $(id).value;
  const pv = keep("prompts"), mv = keep("model_one"), lv = keep("labels");
  $("prompts").innerHTML = S.prompt_files.map(f =>
    `<option value="${esc(f.name)}">${esc(f.name)} (${f.count} ข้อ)</option>`).join("");
  if (pv) $("prompts").value = pv;
  else if (S.prompt_files.some(f => f.name === "controls_only.jsonl")) $("prompts").value = "controls_only.jsonl";
  $("model_one").innerHTML = S.models.map(m => `<option>${esc(m.key)}</option>`).join("");
  if (cmd === "show") $("model_one").insertAdjacentHTML("afterbegin", `<option value="">(ทุกโมเดล)</option>`);
  if (mv) $("model_one").value = mv;
  $("labels").innerHTML = `<option value="">(ไม่ใช้ — ใช้ป้ายในฐานข้อมูล)</option>` +
    S.label_csvs.map(f => `<option>${esc(f)}</option>`).join("");
  if (lv) $("labels").value = lv;

  $("run_ids").innerHTML = S.runs.map(r => `<option value="${esc(r.run_id)}">`).join("");
  $("runs").innerHTML = S.runs.length ? S.runs.map(r => {
    const models = Object.entries(r.models).map(([k, st]) =>
      `${esc(k)}: <span class="ok">${st.ok || 0}</span> / <span class="${(st.error||0) ? "bad" : "muted"}">${st.error || 0}</span>`).join("<br>");
    const b = (c, t) => `<button class="ghost" data-quick="${c}" data-run="${esc(r.run_id)}" data-dry="${r.db === "dry"}">${t}</button>`;
    return `<tr><td><b>${esc(r.run_id)}</b><br><span class="muted">${esc((r.created_at||"").slice(0,16).replace("T"," "))}</span></td>
      <td>${r.db === "dry" ? '<span class="pill warn">dry</span>' : '<span class="pill ok">จริง</span>'}</td>
      <td>${models || '<span class="muted">-</span>'}</td>
      <td class="actions">${b("score","Score")} ${b("summary","Summary")} ${b("export-csv","CSV")} <button class="ghost" data-label="${r.db}|${esc(r.run_id)}">ติดป้าย</button></td></tr>`;
  }).join("") : `<tr><td colspan="4" class="muted">ยังไม่มี run</td></tr>`;

  if (VIEW === "label") fillLabelRuns();
  $("csvs").innerHTML = S.label_csvs.length
    ? "ดาวน์โหลด CSV: " + S.label_csvs.map(f => `<a href="/download?file=${encodeURIComponent(f)}">${esc(f)}</a>`).join(" · ")
    : "";
  updateHint();
}

function setTab(c) {
  cmd = c;
  document.querySelectorAll("#tabs button").forEach(b => b.classList.toggle("on", b.dataset.cmd === c));
  document.querySelectorAll("[data-for]").forEach(el => el.hidden = !el.dataset.for.split(" ").includes(c));
  if (S) renderState();
}

function selectedKeys() { return [...document.querySelectorAll('input[name=mk]:checked')].map(x => x.value); }

function plannedCalls() {
  const f = S.prompt_files.find(p => p.name === $("prompts").value);
  return (f ? f.count : 0) * (parseInt($("samples").value) || 1) * selectedKeys().length;
}

let estTimer = null, estSeq = 0;
function updateHint() {
  let t = "";
  $("estimate").hidden = true;
  if (cmd === "run" && S) {
    const n = plannedCalls();
    t = $("dry_run").checked ? `สูงสุด ${n} calls (โมเดลจำลอง ไม่เสียเงิน)`
      : `สูงสุด ${n} API calls ที่เสียเงินจริง (ข้อที่ทำเสร็จแล้วใน run id เดิมจะข้าม) · run id เดิมต้องใช้โมเดลชุดเดียวกับที่เคยรัน ไม่งั้น spike.py จะไม่ยอมรัน`;
    if (!$("dry_run").checked && n > 0) { clearTimeout(estTimer); estTimer = setTimeout(refreshEstimate, 250); }
  } else if (cmd === "repro-check") {
    t = "เรียก API จริงด้วย prompt และ seed เดิมซ้ำ เพื่อดูว่าได้คำตอบเหมือนเดิมไหม (เสียเงินเล็กน้อย)";
  } else if (cmd === "export-csv") {
    t = "ไฟล์จะถูกบันทึกเป็น v3-*.csv ในโฟลเดอร์โปรเจกต์ (อยู่ใน .gitignore) แล้วดาวน์โหลดได้จากด้านขวา";
  } else if (cmd === "score") {
    t = "ป้ายจากแท็บ \"ติดป้าย\" ถูกใช้อัตโนมัติ ไฟล์ CSV ด้านบนเป็นตัวเลือกเสริม (ถ้าขัดกับป้ายในฐานข้อมูล score จะหยุดและแจ้ง) ข้อที่ยังไม่มีป้ายจะใช้คำใบ้อัตโนมัติ (PROVISIONAL)";
  }
  $("cmd_hint").textContent = t;
}

function estimateHtml(e) {
  const rows = e.rows.map(r => `<div>${esc(r.key)}: ${r.calls} calls · ${usd(r.typical)} (สูงสุด ${usd(r.worst)})</div>`).join("");
  return `<b>ค่าใช้จ่ายโดยประมาณ ${usd(e.typical)}</b> <span class="muted">(ถ้าตอบยาวเต็ม max_tokens ทุกครั้ง สูงสุด ${usd(e.worst)})</span>${rows}` +
    (e.unknown_price.length ? `<div class="warn">ไม่มีราคาของ ${esc(e.unknown_price.join(", "))} ใน note จึงไม่ได้รวม</div>` : "") +
    `<div class="muted">ประมาณจากจำนวนตัวอักษร ไม่ใช่ตัวเลขจริง ค่าจริงดูได้ที่หน้า OpenRouter</div>`;
}

async function refreshEstimate() {
  const seq = ++estSeq;
  const keys = selectedKeys();
  if (cmd !== "run" || $("dry_run").checked || !keys.length) return;
  try {
    const e = await api("/api/estimate", {models: keys, prompts: $("prompts").value, samples: parseInt($("samples").value) || 1});
    if (seq !== estSeq) return;
    $("estimate").innerHTML = estimateHtml(e); $("estimate").hidden = false;
  } catch (err) { /* estimate is optional; the run itself validates everything */ }
}

function collect() {
  return {
    command: cmd,
    run_id: $("run_id").value.trim(),
    prompts: $("prompts").value,
    models: selectedKeys(),
    samples: parseInt($("samples").value) || 1,
    model: $("model_one").value,
    prompt: $("prompt_id").value.trim(),
    repeats: parseInt($("repeats").value) || 2,
    labels: $("labels").value,
    dry_run: $("dry_run").checked,
  };
}

async function submit(req) {
  if (req.command === "run") {
    if (!req.models.length) return alert("เลือกอย่างน้อย 1 โมเดล");
    const n = plannedCalls();
    req.max_calls = Math.max(1, n);
    if (!req.dry_run) {
      let est = "";
      try { const e = await api("/api/estimate", {models: req.models, prompts: req.prompts, samples: req.samples});
            est = `\nประมาณ ${usd(e.typical)} (สูงสุด ${usd(e.worst)})`; } catch (err) {}
      if (!confirm(`จะเรียก API จริงสูงสุด ${n} ครั้ง ซึ่งมีค่าใช้จ่าย${est}\nrun id: ${req.run_id}\nโมเดล: ${req.models.join(", ")}\nยืนยันไหม?`)) return;
      req.confirm_cost = true;
    }
  }
  if (req.command === "repro-check" && !confirm("repro-check เรียก API จริง (เสียเงินเล็กน้อย) ยืนยันไหม?")) return;
  try {
    await api("/api/job", req);
    outLen = 0; $("out").textContent = "";
    startPolling();
  } catch (e) { alert(e.message); }
}

async function poll() {
  const j = await api("/api/job?since=" + outLen).catch(() => null);
  if (!j) return;
  if (j.length < outLen) { outLen = 0; $("out").textContent = ""; return; }  // new job replaced the log
  if (j.output) {
    const out = $("out");
    const atBottom = out.scrollTop + out.clientHeight >= out.scrollHeight - 30;
    out.textContent += j.output;
    outLen = j.length;
    if (atBottom) out.scrollTop = out.scrollHeight;
  }
  $("job_cmd").textContent = j.argv.length ? "python3 spike.py " + j.argv.join(" ") : "";
  const st = $("job_status");
  if (j.running) { st.className = "pill warn"; st.textContent = "กำลังรัน…"; }
  else if (j.exit_code === null) { st.className = "pill muted"; st.textContent = "ว่าง"; }
  else if (j.exit_code === 0) { st.className = "pill ok"; st.textContent = "เสร็จ"; }
  // spike.py score exits 3 when a model fails the pre-registered control gate: that is a
  // result to read, not a crash. Negative codes mean the process was killed (Stop button).
  else if (j.exit_code === 3 && j.argv[0] === "score") { st.className = "pill warn"; st.textContent = "เสร็จ: มีโมเดลไม่ผ่านเกณฑ์ control (ดูผลด้านล่าง)"; }
  else if (j.exit_code === 2 && j.argv[0] === "repro-check") { st.className = "pill warn"; st.textContent = "เสร็จ: คำตอบไม่เหมือนกัน (seed ไม่ทำให้ซ้ำได้)"; }
  else if (j.exit_code < 0) { st.className = "pill muted"; st.textContent = "หยุดแล้ว"; }
  else { st.className = "pill bad"; st.textContent = "จบด้วย exit " + j.exit_code; }
  $("stop").disabled = !j.running;
  $("go").disabled = j.running;
  if (!j.running && polling) {
    clearInterval(polling); polling = null;
    refresh();  // run counts / new CSVs
  }
}

function startPolling() { if (!polling) polling = setInterval(poll, 700); poll(); }
async function refresh() { S = await api("/api/state"); renderState(); }

/* ------------------------------- label view ------------------------------- */
function fillLabelRuns() {
  if (!S) return;
  const cur = $("l_run").value;
  $("l_run").innerHTML = `<option value="">— เลือก run —</option>` + S.runs.map(r =>
    `<option value="${r.db}|${esc(r.run_id)}">${esc(r.run_id)}${r.db === "dry" ? " (dry)" : ""}</option>`).join("");
  if (cur) $("l_run").value = cur;
}

async function loadLabels(keepIndex) {
  const v = $("l_run").value;
  if (!v) { L = null; $("l_card").hidden = true; $("l_done").hidden = true; return; }
  const [db, run_id] = [v.slice(0, v.indexOf("|")), v.slice(v.indexOf("|") + 1)];
  $("l_info").textContent = "กำลังโหลด…";
  try {
    const r = await api("/api/label/items?" + new URLSearchParams({db, run_id, task: "control"}));
    const i = keepIndex && L && L.run_id === run_id ? L.i : 0;
    L = {db, run_id, field: r.field, items: r.items, i};
    LSEEN = new Set();
    if (!L.items.length) {
      $("l_info").textContent = "run นี้ไม่มีข้อสอบควบคุมแบบถามตรงที่ต้องติดป้าย";
      $("l_card").hidden = true; $("l_done").hidden = true; updateProgress(); return;
    }
    if (!keepIndex && $("l_mode").value === "todo") gotoNext(-1);
    renderLabel();
  } catch (e) { $("l_info").innerHTML = `<span class="bad">${esc(e.message)}</span>`; }
}

function labelledCount() { return L ? L.items.filter(x => x.label).length : 0; }

function updateProgress() {
  const n = L ? L.items.length : 0, k = labelledCount();
  $("l_bar").style.width = n ? (100 * k / n) + "%" : "0";
  if (L && n) $("l_info").textContent = `ติดป้ายแล้ว ${k}/${n} ข้อ · ${L.db === "dry" ? "ฐานข้อมูล dry run" : "ฐานข้อมูลจริง"} · คีย์ลัด Y = ถูก, N = ผิด, ← → = เลื่อน`;
}

// Move to the next item after `from` (wrapping); in "todo" mode prefer unlabelled ones.
function gotoNext(from) {
  const n = L.items.length;
  if ($("l_mode").value === "todo") {
    for (let s = 1; s <= n; s++) { const j = (from + s + n) % n; if (!L.items[j].label) { L.i = j; return true; } }
    return false;   // everything is labelled
  }
  L.i = Math.min(n - 1, from + 1);
  return true;
}

function renderLabel() {
  updateProgress();
  const it = L.items[L.i];
  const allDone = labelledCount() === L.items.length;
  $("l_done").hidden = !allDone;
  $("l_card").hidden = false;
  $("l_pos").textContent = `ข้อ ${L.i + 1} / ${L.items.length}  ·  ${it.prompt_id}${it.sample_idx ? " #" + it.sample_idx : ""}`;
  const blind = $("l_blind").checked;
  const reveal = !blind || LSEEN.has(it.gen_id);
  $("l_who").innerHTML = reveal
    ? `โมเดล: <b>${esc(it.model_key)}</b>${blind ? "" : ` · คำใบ้อัตโนมัติ: ${it.hint === true ? "ถูก" : it.hint === false ? "ผิด" : "-"}`}`
    : (it.label ? `<button class="ghost small" id="l_reveal">ดูชื่อโมเดล</button>` : `โมเดล: ซ่อนอยู่ (ติดป้ายก่อนถึงดูได้)`);
  $("l_q").textContent = it.question;
  $("l_exp").textContent = it.expected ?? "(ไม่มีเฉลย)";
  $("l_resp").textContent = it.response;
  $("l_resp").scrollTop = 0;
  const cur = $("l_cur");
  cur.className = "pill " + (it.label === "y" ? "ok" : it.label === "n" ? "bad" : "muted");
  cur.textContent = it.label === "y" ? "ป้ายปัจจุบัน: ถูก" : it.label === "n" ? "ป้ายปัจจุบัน: ผิด" : "ยังไม่ติดป้าย";
  $("l_clear").hidden = !it.label;
  $("l_note").value = it.note || "";
  $("l_hist").hidden = true;
  $("l_hist_btn").textContent = `ดูประวัติป้ายของข้อนี้ (${it.n_history})`;
  $("l_prev").disabled = L.i === 0;
  $("l_msg").textContent = "";
}

async function saveLabel(value) {
  const it = L.items[L.i];
  try {
    await saveNote();   // keep a note typed just before pressing Y/N
    const r = await api("/api/label/set", {db: L.db, run_id: L.run_id, gen_id: it.gen_id, field: L.field, value});
    it.label = r.value; it.n_history = r.n_history;
    if (value !== null && gotoNext(L.i) === false) L.i = L.items.indexOf(it);   // all done: stay
    renderLabel();
  } catch (e) { $("l_msg").innerHTML = `<span class="bad">${esc(e.message)}</span>`; }
}

async function saveNote() {
  if (!L) return;
  const it = L.items[L.i], v = $("l_note").value.trim();
  if (v === (it.note || "")) return;
  const r = await api("/api/label/set", {db: L.db, run_id: L.run_id, gen_id: it.gen_id, field: "human_notes", value: v || null});
  it.note = r.value || ""; it.n_history = r.n_history;
  $("l_msg").innerHTML = `<span class="ok">บันทึกโน้ตแล้ว</span>`;
  $("l_hist_btn").textContent = `ดูประวัติป้ายของข้อนี้ (${it.n_history})`;
}

async function showHistory() {
  const it = L.items[L.i];
  try {
    const h = await api("/api/label/history?" + new URLSearchParams({db: L.db, run_id: L.run_id, gen_id: it.gen_id}));
    const txt = (f, v) => f === "human_control_correct" ? (v === "y" ? "ถูก" : v === "n" ? "ผิด" : "ล้าง") : (v ? `โน้ต: ${v}` : "ล้างโน้ต");
    $("l_hist").innerHTML = h.length ? h.map(x => `<div>${esc((x.at || "").replace("T", " ").slice(0, 19))} · ${esc(x.source)} · ${esc(txt(x.field, x.value))}</div>`).join("") : "ยังไม่มีประวัติ";
    $("l_hist").hidden = false;
  } catch (e) { $("l_msg").innerHTML = `<span class="bad">${esc(e.message)}</span>`; }
}

/* ------------------------------- models view ------------------------------- */
let catTimer = null, catSeq = 0;
async function loadCatalog() {
  const seq = ++catSeq;
  const qs = new URLSearchParams({q: $("q").value.trim(), sort: $("sort").value, scope: CAT.scope,
    free: $("f_free").checked ? "1" : "0", seed: $("f_seed").checked ? "1" : "0", limit: "60"});
  $("cat_info").textContent = "กำลังโหลด…";
  try {
    const r = await api("/api/catalog?" + qs);
    if (seq !== catSeq) return;
    CAT.loaded = true;
    const overview = CAT.scope === "featured" && !$("q").value.trim();
    $("chips").innerHTML = [["featured", "⭐ โมเดลดัง"], ...r.chips.map(c => ["v:" + c.key, `${c.label} (${c.count})`]), ["all", `ทั้งหมด (${r.catalog_size})`]]
      .map(([k, t]) => `<button type="button" data-scope="${esc(k)}" class="${CAT.scope === k ? "on" : ""}">${esc(t)}</button>`).join("");
    $("cat_info").textContent = (overview ? `โมเดลใหม่ล่าสุดของค่ายดังแต่ละค่าย (ไม่รวมรุ่นฟรีและรุ่นเฉพาะทาง) · กดชื่อค่ายด้านบนเพื่อดูทั้งหมดของค่ายนั้น`
      : `พบ ${r.total} โมเดล (แสดง ${r.items.length} อันแรก)`) + ` · ข้อมูลสาธารณะจาก openrouter.ai ไม่ส่ง key`;
    let lastVendor = null;
    $("cat_rows").innerHTML = r.items.map(m => {
      const head = overview && m.vendor !== lastVendor ? `<tr class="grp"><td colspan="4">${esc(m.vendor)}</td></tr>` : "";
      lastVendor = m.vendor;
      return head + `<tr class="pick ${D && D.id === m.id ? "sel" : ""}" data-id="${esc(m.id)}">
        <td><b>${esc(m.name)}</b><br><span class="mono muted">${esc(m.id)}</span><br>
          ${m.free ? '<span class="tag g">ฟรี</span>' : ""}${m.out >= 10 ? '<span class="tag" style="color:var(--warn);border-color:var(--warn)">ราคาสูง</span>' : ""}${m.seed ? '<span class="tag">seed</span>' : ""}${m.reasoning ? '<span class="tag">reasoning</span>' : ""}</td>
        <td class="num">${price(m.in)}</td><td class="num">${price(m.out)}</td><td class="num">${ctxTxt(m.ctx)}</td></tr>`;
    }).join("") || `<tr><td colspan="4" class="muted">ไม่พบโมเดลที่ตรงกับเงื่อนไข</td></tr>`;
  } catch (e) {
    if (seq !== catSeq) return;
    $("cat_info").innerHTML = `<span class="bad">โหลดรายการไม่ได้: ${esc(e.message)}</span> — ยังพิมพ์ model id เองด้านล่างได้`;
    $("cat_rows").innerHTML = "";
  }
}

function slugKey(id, free) {
  let s = id.split("/")[1].replace(/:free$/, "").toLowerCase().replace(/\./g, "").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  s = ("or-" + s + (free ? "-free" : "")).slice(0, 48).replace(/-$/, "");
  let k = s, i = 2;
  while (S.models.some(m => m.key === k)) k = s.slice(0, 44) + "-" + i++;
  return k;
}

async function pickModel(id) {
  $("detail_body").innerHTML = `<span class="muted">กำลังโหลดรายชื่อ provider ของ ${esc(id)}…</span>`;
  try {
    D = await api("/api/endpoints?id=" + encodeURIComponent(id));
  } catch (e) { D = null; $("detail_body").innerHTML = `<span class="bad">${esc(e.message)}</span>`; return; }
  document.querySelectorAll("#cat_rows tr").forEach(tr => tr.classList.toggle("sel", tr.dataset.id === D.id));
  renderDetail();
}

function renderDetail() {
  if (!D.endpoints.length) { $("detail_body").innerHTML = `<b>${esc(D.name)}</b><p class="bad">โมเดลนี้ไม่มี provider ให้ใช้งานตอนนี้</p>`; return; }
  const opts = D.endpoints.map((e, i) =>
    `<option value="${i}">${esc(e.provider)} · ${esc(e.quant || "ไม่ระบุ quantization")} · ${price(e.in)} / ${price(e.out)} · ${ctxTxt(e.ctx)}${e.seed ? " · seed" : ""}</option>`).join("");
  const free = D.id.endsWith(":free");
  $("detail_body").innerHTML = `
    <div><b>${esc(D.name)}</b><br><span class="mono muted">${esc(D.id)}</span></div>
    <label class="check"><input type="checkbox" id="d_pin" checked><span>ล็อก provider (แนะนำ ทุกคำตอบมาจาก weights ชุดเดียวกัน)</span></label>
    <div id="d_ep_wrap">
      <label for="d_ep">Provider และ quantization</label>
      <select id="d_ep">${opts}</select>
    </div>
    <label for="d_key">ชื่อ key ใน config</label>
    <input type="text" id="d_key" value="${esc(slugKey(D.id, free))}">
    <div class="check" style="margin-top:12px"><input type="checkbox" id="d_think" checked><label for="d_think">ปิด thinking (เหมือนโมเดลอื่นในโปรเจกต์)</label></div>
    <label for="d_temp">temperature</label>
    <input type="number" id="d_temp" min="0" max="2" step="0.1" value="0.7">
    <label for="d_max">max_tokens</label>
    <input type="number" id="d_max" min="16" max="32000" value="2000">
    <div id="d_warn"></div>
    <div class="row"><button id="d_add">เพิ่มเข้า config</button><span class="hint" id="d_msg"></span></div>
    <p class="hint">จะบันทึกลง config/models.yaml เป็นบล็อกที่ลบได้จากหน้านี้ ยังไม่เสียเงิน จนกว่าจะกดรัน แนะนำให้รันข้อสอบควบคุม (controls_only.jsonl) ก่อนเสมอ</p>`;
  // Default to the cheapest provider that states its quantization and supports seed (a
  // pinned run is only meaningful if we know which weights it pins); fall back gracefully.
  const best = [e => e.max_tokens && e.quant && e.seed, e => e.max_tokens && e.quant, e => e.max_tokens && e.seed, e => e.max_tokens]
    .map(f => D.endpoints.findIndex(f)).find(i => i >= 0);
  if (D.reasoning_mandatory) $("d_max").value = "4000";
  $("d_ep").value = String(best ?? 0);
  updateDetailWarnings();
}

function curEp() { return $("d_pin").checked ? D.endpoints[parseInt($("d_ep").value) || 0] : null; }

function updateDetailWarnings() {
  $("d_ep_wrap").hidden = !$("d_pin").checked;
  const pool = $("d_pin").checked ? [curEp()] : D.endpoints;
  const any = (f) => pool.some(e => e[f]);
  const w = [];
  if (!$("d_pin").checked) w.push("ไม่ล็อก provider: OpenRouter อาจส่งแต่ละ call ไปคนละเจ้า/คนละ quantization ผลเทียบกันไม่ได้อย่างเข้มงวด");
  if (!any("seed")) w.push("provider นี้ไม่รองรับ seed: รันซ้ำแล้วคำตอบไม่ซ้ำเดิม และ Repro ใช้ไม่ได้");
  if (!any("temperature")) w.push("provider นี้ไม่รับ temperature: จะไม่ส่งค่า ทำให้เทียบกับโมเดลอื่นที่ใช้ 0.7 ตรงๆ ไม่ได้");
  let blocked = false;
  if (!any("max_tokens")) { blocked = true; w.push("provider นี้ไม่รับ max_tokens ซึ่ง spike.py ส่งทุกครั้ง: จะ error ทุก call จึงเพิ่มไม่ได้ เลือก provider อื่น"); }
  if (D.reasoning_mandatory) {
    $("d_think").checked = false;
    w.push("โมเดลนี้ปิด thinking ไม่ได้: จะตั้ง effort ต่ำสุดให้แทน token ที่ใช้คิดนับรวมใน max_tokens (แนะนำ 4000 ขึ้นไป) และผลเทียบกับโมเดลที่ปิด thinking ได้ไม่ตรงนัก");
  } else if (!any("reasoning")) { $("d_think").checked = false; w.push("โมเดลนี้ไม่มีตัวเลือก thinking จึงไม่ส่ง reasoning"); }
  $("d_think").disabled = !any("reasoning") || D.reasoning_mandatory;
  $("d_add").disabled = blocked;
  $("d_temp").disabled = !any("temperature");
  if ($("d_pin").checked) {
    const e = curEp(), same = D.endpoints.filter(x => x.slug === e.slug && x.quant === e.quant).length;
    if (same > 1) w.push("มีหลาย endpoint ของ provider นี้ที่ quantization เดียวกัน OpenRouter อาจเลือกตัวใดตัวหนึ่ง");
    if (!e.quant) w.push("provider นี้ไม่ระบุ quantization จึงล็อกได้แค่ชื่อ provider");
  }
  $("d_warn").innerHTML = w.map(t => `<div class="box warnbox warn">${esc(t)}</div>`).join("");
}

async function addModel() {
  const e = curEp();
  $("d_add").disabled = true; $("d_msg").textContent = "กำลังบันทึก…";
  try {
    const r = await api("/api/models/add", {
      model: D.id, key: $("d_key").value.trim(), pin: $("d_pin").checked, tag: e ? e.tag : "",
      thinking_off: $("d_think").checked && !$("d_think").disabled,
      temperature: $("d_temp").disabled ? null : $("d_temp").value,
      max_tokens: parseInt($("d_max").value) || 2000});
    await refresh();
    SEL.add(r.key);
    renderModelChecks();
    $("d_msg").innerHTML = `<span class="ok">เพิ่ม ${esc(r.key)} แล้ว</span> — ติ๊กเลือกให้ในหน้ารันแล้ว` +
      (r.warnings.length ? `<br><span class="warn">${r.warnings.map(esc).join("<br>")}</span>` : "");
    setTimeout(() => setView("run"), r.warnings.length ? 4000 : 900);
  } catch (err) { $("d_msg").innerHTML = `<span class="bad">${esc(err.message)}</span>`; }
  $("d_add").disabled = false;
}

/* ------------------------------- events ------------------------------- */
$("nav").addEventListener("click", e => { if (e.target.dataset.view) setView(e.target.dataset.view); });
$("l_run").addEventListener("change", () => loadLabels(false));
$("l_mode").addEventListener("change", () => { if (L) { if ($("l_mode").value === "todo") gotoNext(L.i - 1); renderLabel(); } });
$("l_blind").addEventListener("change", () => { if (L) renderLabel(); });
$("l_y").addEventListener("click", () => saveLabel("y"));
$("l_n").addEventListener("click", () => saveLabel("n"));
$("l_clear").addEventListener("click", () => { if (confirm("ล้างป้ายของข้อนี้? (ประวัติยังเก็บไว้)")) saveLabel(null); });
$("l_prev").addEventListener("click", async () => { await saveNote().catch(() => {}); if (L.i > 0) { L.i--; renderLabel(); } });
$("l_next").addEventListener("click", async () => {
  await saveNote().catch(() => {});
  const was = L.i;
  if ($("l_mode").value === "todo") { if (!gotoNext(L.i)) L.i = Math.min(L.items.length - 1, was + 1); }
  else L.i = Math.min(L.items.length - 1, L.i + 1);
  renderLabel();
});
$("l_note").addEventListener("blur", () => saveNote().catch(e => { $("l_msg").innerHTML = `<span class="bad">${esc(e.message)}</span>`; }));
$("l_hist_btn").addEventListener("click", showHistory);
$("l_who").addEventListener("click", e => { if (e.target.id === "l_reveal") { LSEEN.add(L.items[L.i].gen_id); renderLabel(); } });
$("l_review").addEventListener("click", () => { $("l_mode").value = "all"; L.i = 0; renderLabel(); });
$("l_score").addEventListener("click", () => {
  setView("run"); setTab("score");
  submit({command: "score", run_id: L.run_id, dry_run: L.db === "dry"});
});
document.addEventListener("keydown", e => {
  if (VIEW !== "label" || !L || $("l_card").hidden) return;
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName) || e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === "y") { e.preventDefault(); saveLabel("y"); }
  else if (k === "n") { e.preventDefault(); saveLabel("n"); }
  else if (e.key === "ArrowRight") { e.preventDefault(); $("l_next").click(); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); $("l_prev").click(); }
});
$("go_models").addEventListener("click", () => setView("models"));
$("tabs").addEventListener("click", e => { if (e.target.dataset.cmd) setTab(e.target.dataset.cmd); });
$("form").addEventListener("submit", e => { e.preventDefault(); submit(collect()); });
$("form").addEventListener("input", updateHint);
$("model_checks").addEventListener("change", () => { SEL.clear(); selectedKeys().forEach(k => SEL.add(k)); updateHint(); });
$("stop").addEventListener("click", () => api("/api/job/stop", {}).catch(e => alert(e.message)));
$("refresh").addEventListener("click", refresh);
$("runs").addEventListener("click", e => {
  const lb = e.target.closest("button[data-label]");
  if (lb) { setView("label"); $("l_run").value = lb.dataset.label; loadLabels(false); return; }
  const b = e.target.closest("button[data-quick]"); if (!b) return;
  submit({command: b.dataset.quick, run_id: b.dataset.run, dry_run: b.dataset.dry === "true"});
});
// A run id that already exists must be resumed with the SAME models: tick exactly those.
$("run_id").addEventListener("change", () => {
  if (cmd !== "run" || !S) return;
  const r = S.runs.find(x => x.run_id === $("run_id").value.trim() && (x.db === "dry") === $("dry_run").checked);
  if (!r) return;
  SEL.clear(); Object.keys(r.models).filter(k => S.models.some(m => m.key === k)).forEach(k => SEL.add(k));
  renderModelChecks(); updateHint();
  $("cmd_hint").textContent = "run id นี้มีอยู่แล้ว: ติ๊กโมเดลที่เคยรันไว้ให้อัตโนมัติ (ต้องใช้ชุดเดียวกันถึงจะรันต่อได้)";
});
$("models").addEventListener("click", async e => {
  const k = e.target.dataset.remove; if (!k) return;
  if (!confirm(`ลบ ${k} ออกจาก config?\nข้อมูลใน run เก่าที่ใช้โมเดลนี้ยังอยู่ แต่จะรันต่อจาก run นั้นไม่ได้`)) return;
  try { await api("/api/models/remove", {key: k}); await refresh(); } catch (err) { alert(err.message); }
});
const reloadCat = () => { clearTimeout(catTimer); catTimer = setTimeout(loadCatalog, 300); };
$("q").addEventListener("input", reloadCat);
$("chips").addEventListener("click", e => { const k = e.target.dataset.scope; if (k) { CAT.scope = k; loadCatalog(); } });
["sort", "f_free", "f_seed"].forEach(id => $(id).addEventListener("change", loadCatalog));
$("cat_rows").addEventListener("click", e => { const tr = e.target.closest("tr[data-id]"); if (tr) pickModel(tr.dataset.id); });
$("manual_go").addEventListener("click", () => { const v = $("manual_id").value.trim(); if (v) pickModel(v); });
$("manual_id").addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); $("manual_go").click(); } });
$("detail_body").addEventListener("change", e => { if (["d_pin", "d_ep"].includes(e.target.id)) updateDetailWarnings(); });
$("detail_body").addEventListener("click", e => { if (e.target.id === "d_add") addModel(); });

setTab("run");
refresh().then(() => { if (S.job.running || S.job.argv.length) startPolling(); });
</script>
</body>
</html>
"""


def main() -> int:
    global CONFIG
    ap = argparse.ArgumentParser(description="Local web UI for spike.py")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="models config to read and edit (default config/models.yaml)")
    args = ap.parse_args()
    CONFIG = Path(args.config).resolve()
    if not CONFIG.exists():
        sys.exit(f"Config file not found: {CONFIG}")
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        if exc.errno == 48 or "in use" in str(exc).lower():  # 48 = EADDRINUSE on macOS
            sys.exit(f"Port {args.port} is already in use (another webui.py still running?).\n"
                     f"Stop it, or pick another port: python3 webui.py --port {args.port + 1}")
        raise
    print(f"Bias POC web UI: http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    if CONFIG != DEFAULT_CONFIG:
        print(f"config: {CONFIG}")
    missing = [m["api_key_env"] for m in load_models() if not m["key_set"]]
    if missing:
        print(f"note: {', '.join(sorted(set(filter(None, missing))))} not set in this shell; "
              f"only dry runs and read-only commands will work.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        JOB.stop()
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
