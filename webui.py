#!/usr/bin/env python3
"""
webui.py - a small local web UI for spike.py (bias-poc v3, remote API only).

What it does (5 steps + advanced):
  1 โมเดล       pick models (default: default_models in config/models.yaml, a free model);
                "+ เพิ่มโมเดล" browses the public OpenRouter catalog and appends a pinned entry.
  2 ชุดคำถาม    pick a question set: trial (prompts_v3_mini.jsonl, 34), controls (22), full (174).
  3 รัน         one button; progress bar, cost estimate, stop; spike.py run underneath.
  4 ตรวจโดยคน   the run's to-do list for a human (failed calls, unlabelled direct_question
                controls, control-gate failures, evasive answers, too few templates) and the
                blind labelling UI (append-only labels table via spike.write_label()).
  5 สรุป & เทียบ per-model summary (control gate, cost, evasion, position A) and a per-dimension
                comparison with every other model that ran the same question set. No overall
                ranking (config: reporting.no_single_ranking).
  ⚙ ขั้นสูง     runs table and the raw spike.py commands (score/summary/show/export/repro/selftest).

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


def default_models() -> list[str]:
    try:
        data = yaml.safe_load(read_config_text()) or {}
    except yaml.YAMLError:
        return []
    return [str(k) for k in (data.get("default_models") or [])]


def state() -> dict:
    files = prompt_files()
    for f in files:
        label, desc = QUESTION_SETS.get(f["name"], (None, ""))
        f["label"], f["desc"] = label, desc
    return {"models": load_models(), "default_models": default_models(), "prompt_files": files,
            "label_csvs": label_csvs(), "runs": list_runs(), "job": JOB.snapshot(0),
            "api_key_env": API_KEY_ENV}


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
# Run summary, human to-do list and per-dimension comparison
# ----------------------------------------------------------------------------------------
# Known question sets, shown as cards in step 2 (other *.jsonl files are listed as "อื่นๆ").
QUESTION_SETS = {
    "prompts_v3_mini.jsonl": ("ชุดทดลอง", "ดูภาพรวมทุกขั้นตอนแบบเร็ว: ข้อสอบควบคุม 6 ข้อ + มิติละ 1 template "
                              "(7 มิติ) ใช้ทำความเข้าใจการทำงาน สรุปผลจริงไม่ได้"),
    "controls_only.jsonl": ("ข้อสอบควบคุม", "22 ข้อง่ายๆ ที่มีเฉลย เช็คว่าโมเดลเข้าใจภาษาไทยและทำตามคำสั่งได้ "
                            "ควรผ่าน ≥80% ก่อนรันชุดเต็ม"),
    "prompts_v3.jsonl": ("ชุดเต็ม", "174 ข้อ ครบทุกมิติ มิติละ ≥3 template สำหรับผลที่ใช้สรุปได้ "
                         "(แนะนำ samples = 3)"),
}
DIMENSION_TH = {
    "accent": "สำเนียง", "region": "ภูมิภาค", "institution": "สถาบันการศึกษา", "gender": "เพศ",
    "age": "อายุ", "religion_via_name": "ศาสนา (จากชื่อ)", "nationality": "สัญชาติ",
    "gender_x_region": "เพศ × ภูมิภาค",
}
VARIANT_TH = {
    "isan": "อีสาน", "bangkok": "กรุงเทพฯ", "south": "ใต้", "central": "กลาง",
    "top_university": "มหาวิทยาลัยชั้นนำ", "rajabhat": "ราชภัฏ", "male": "ชาย", "female": "หญิง",
    "young": "อายุน้อย", "older": "อายุมาก", "buddhist_name": "ชื่อพุทธ", "muslim_name": "ชื่อมุสลิม",
    "thai": "ไทย", "myanmar": "เมียนมา", "female_isan": "หญิงอีสาน", "female_bangkok": "หญิงกรุงเทพฯ",
    "male_isan": "ชายอีสาน", "male_bangkok": "ชายกรุงเทพฯ",
}


def _thresholds() -> dict:
    return spike.load_thresholds(ROOT / spike.DEFAULT_THRESHOLDS)


def _prompt_id_sets() -> dict:
    out = {}
    for f in prompt_files():
        try:
            out[f["name"]] = frozenset(p["id"] for p in spike.load_prompts(ROOT / f["name"]))
        except SystemExit:
            continue
    return out


def question_set_of(ids: set, recorded: Optional[str] = None) -> tuple[Optional[str], int]:
    """
    (prompt file a run used, number of its prompts).
    1. runs.prompts_file, recorded by spike.py when the run started (reliable, also for a run
       stopped part-way) — used if that file still exists and contains every id of the run.
    2. Older runs: an exact match of the run's prompt ids. A stopped old run cannot be matched
       reliably (its first ids may belong to several files), so it is reported as unknown.
    """
    sets = _prompt_id_sets()
    if recorded in sets and ids <= sets[recorded]:
        return recorded, len(sets[recorded])
    for name, id_set in sets.items():
        if id_set == ids:
            return name, len(id_set)
    return None, len(ids)


def _open_read(kind: str) -> sqlite3.Connection:
    if kind not in DBS or not DBS[kind].exists():
        raise BadRequest("ไม่พบฐานข้อมูล")
    db = spike.open_db(str(DBS[kind]))
    db.execute("PRAGMA busy_timeout = 10000")
    return db


def run_summary(kind: str, run_id: str) -> dict:
    if not RUN_ID_RE.match(run_id or ""):
        raise BadRequest("run id ไม่ถูกรูปแบบ")
    th = _thresholds()
    threshold = float(th["control_pass_threshold"])
    min_t = int(th.get("min_independent_templates_per_dimension", 3))
    db = _open_read(kind)
    try:
        meta = db.execute("SELECT created_at, n_samples, prompts_file FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if meta is None:
            raise BadRequest("ไม่พบ run นี้")
        ids = {r[0] for r in db.execute("SELECT DISTINCT prompt_id FROM generations WHERE run_id=?", (run_id,))}
        stats = spike.compute_score(db, run_id, spike.human_control_map(db, run_id))
        dims = spike.bias_by_dimension(db, run_id)
        labels = spike.current_labels(db, run_id)
        per_model_rows = db.execute(
            "SELECT model_key, model_id, status, prompt_id, sample_idx, prompt_json, response_text, "
            "json_extract(raw_json, '$.usage.cost') FROM generations WHERE run_id=?", (run_id,)).fetchall()
    finally:
        db.close()
    qset, qset_size = question_set_of(ids, meta[2])
    n_samples = int(meta[1] or 1)
    expected = qset_size * n_samples          # calls each model should have in a complete run
    models: dict = {}
    for mk, mid, status, pid, idx, pj, text, cost in per_model_rows:
        m = models.setdefault(mk, {"key": mk, "model_id": mid, "calls": 0, "errors": 0, "cost": 0.0,
                                   "label_need": 0, "label_done": 0, "evasive_items": []})
        m["calls"] += 1
        m["cost"] += float(cost or 0)
        if status != "ok":
            m["errors"] += 1
            continue
        prompt = json.loads(pj)
        if spike.label_applies("human_control_correct", prompt, {}):
            m["label_need"] += 1
            if labels.get((pid, mk, idx), {}).get("human_control_correct"):
                m["label_done"] += 1
        if prompt.get("format") in spike.V3_AB_FORMATS and not prompt.get("is_control"):
            if spike.analyse_generation(prompt, status, text).get("evasive"):
                m["evasive_items"].append(f"{pid}#{idx}")
    out_models, todo = [], []
    for mk, m in sorted(models.items()):
        st = stats.get(mk)
        gate = spike.gate_verdict(st, threshold) if st else None
        ev_k = sum(v[0] for v in st["evasive"].values()) if st else 0
        ev_n = sum(v[1] for v in st["evasive"].values()) if st else 0
        pa, pt = st["pos_a"] if st else (0, 0)
        rows = dims.get(mk, [])
        for r in rows:
            r["dimension_th"] = DIMENSION_TH.get(r["dimension"], r["dimension"])
            r["group1_th"] = VARIANT_TH.get(r["group1"], r["group1"])
            r["group2_th"] = VARIANT_TH.get(r["group2"], r["group2"])
        out_models.append({**{k: v for k, v in m.items() if k != "evasive_items"},
                           "evasive_k": ev_k, "evasive_n": ev_n, "evasive_items": m["evasive_items"][:30],
                           "pos_a": [pa, pt], "gate": gate, "dims": rows})
        # --- what a human still has to do, most important first ---
        if m["calls"] < expected:
            todo.append({"level": "must", "model": mk, "action": "rerun",
                         "text": f"{mk}: รันยังไม่ครบ {m['calls']}/{expected} ครั้ง (ถูกหยุดกลางทาง?) — กด 'รันต่อ' "
                                 f"ด้วย run id เดิม ระบบจะทำเฉพาะข้อที่ยังไม่มี"})
        if m["errors"]:
            todo.append({"level": "must", "model": mk, "action": "rerun",
                         "text": f"{mk}: มี {m['errors']} call ที่ล้มเหลว — กด 'รันซ้ำ' ด้วย run id เดิม ระบบจะทำเฉพาะข้อที่ขาด"})
        if m["label_need"] > m["label_done"]:
            todo.append({"level": "must", "model": mk, "action": "label",
                         "text": f"{mk}: ติดป้ายข้อสอบควบคุมแบบถามตรง {m['label_done']}/{m['label_need']} ข้อ — "
                                 f"ต้องครบก่อนผลข้อสอบควบคุมจะเป็นผลสุดท้าย"})
        if gate and gate["verdict"] == "FAIL":
            todo.append({"level": "must", "model": mk, "action": None,
                         "text": f"{mk}: ข้อสอบควบคุมได้ {gate['k']}/{gate['n']} ต่ำกว่าเกณฑ์ {threshold:.0%} — "
                                 f"{'ยังเป็นผลชั่วคราว ติดป้ายให้ครบก่อน' if gate['provisional'] else 'ห้ามตีความผลอคติของโมเดลนี้ใน run นี้'}"})
        if m["evasive_items"]:
            todo.append({"level": "should", "model": mk, "action": "show",
                         "text": f"{mk}: มีคำตอบเลี่ยงไม่เลือก {len(m['evasive_items'])} ข้อ — ควรเปิดอ่านว่าเป็นการปฏิเสธ "
                                 f"หรือให้เหตุผลที่สมเหตุสมผล (ไม่นับเป็นคะแนน)"})
    few = sorted({r["dimension_th"] for m in out_models for r in m["dims"] if r["n_templates"] < min_t})
    if few:
        todo.append({"level": "info", "model": None, "action": None,
                     "text": f"มิติที่มี template น้อยกว่า {min_t} ({', '.join(few)}) — ใช้ดูภาพรวมได้ แต่ยังสรุปเรื่องอคติไม่ได้ "
                             f"ตามเกณฑ์ที่ตั้งไว้ล่วงหน้า ต้องรันชุดเต็ม"})
    if not any(t["level"] == "must" for t in todo):
        todo.insert(0, {"level": "done", "model": None, "action": None,
                        "text": "ไม่มีงานที่ต้องให้คนยืนยันค้างอยู่ — ดูผลในขั้นที่ 5 ได้เลย"})
    return {"run_id": run_id, "db": kind, "created_at": meta[0], "n_samples": n_samples,
            "question_set": qset, "question_set_label": QUESTION_SETS.get(qset or "", (qset or "ชุดกำหนดเอง", ""))[0],
            "n_prompts": qset_size, "n_prompts_seen": len(ids),
            "complete": all(m["calls"] >= expected for m in out_models),
            "threshold": threshold, "min_templates": min_t, "models": out_models, "todo": todo}


def compare_models(kind: str, qset: str) -> dict:
    """
    Per-dimension comparison across models that ran the SAME question set completely (latest
    run per model; a run stopped part-way is left out so every column covers the same items). No overall ranking (config: reporting.no_single_ranking). For each comparison row a
    model is marked 'less'/'more' lean only when its |share-0.5| interval does not overlap
    every other model's interval; otherwise the models cannot be told apart.
    """
    th = _thresholds()
    threshold = float(th["control_pass_threshold"])
    min_t = int(th.get("min_independent_templates_per_dimension", 3))
    target = _prompt_id_sets().get(qset)
    if target is None:
        raise BadRequest("ไม่รู้จักชุดคำถามนี้")
    db = _open_read(kind)
    try:
        runs = db.execute("SELECT run_id, created_at FROM runs ORDER BY created_at DESC").fetchall()
        columns, seen = [], set()
        for run_id, created in runs:
            ids = {r[0] for r in db.execute("SELECT DISTINCT prompt_id FROM generations WHERE run_id=?", (run_id,))}
            if ids != target:
                continue
            stats = spike.compute_score(db, run_id, spike.human_control_map(db, run_id))
            dims = spike.bias_by_dimension(db, run_id)
            for mk in sorted(dims):
                if mk in seen:
                    continue   # an older run of a model already shown
                seen.add(mk)
                gate = spike.gate_verdict(stats[mk], threshold) if mk in stats else None
                columns.append({"model": mk, "run_id": run_id, "created_at": created, "gate": gate,
                                "rows": {f"{r['dimension']}|{r['group1']}|{r['group2']}": r for r in dims[mk]}})
    finally:
        db.close()
    keys = sorted({k for c in columns for k in c["rows"]})
    rows = []
    for k in keys:
        dim, g1, g2 = k.split("|")
        cells = []
        for c in columns:
            r = c["rows"].get(k)
            usable = bool(r and r["n"] and c["gate"] and c["gate"]["verdict"] == "PASS")
            cells.append({"model": c["model"], **(r or {}), "usable": usable})
        live = [x for x in cells if x["usable"]]
        for x in live:
            others = [o for o in live if o is not x]
            if not others:
                x["vs"] = None
            elif all(x["gap_hi"] < o["gap_lo"] for o in others):
                x["vs"] = "less"     # clearly the smallest lean of all compared models
            elif all(x["gap_lo"] > o["gap_hi"] for o in others):
                x["vs"] = "more"     # clearly the largest lean
            else:
                x["vs"] = "similar"
        n_t = max((x.get("n_templates") or 0) for x in cells) if cells else 0
        rows.append({"dimension": dim, "dimension_th": DIMENSION_TH.get(dim, dim),
                     "group1": g1, "group2": g2, "group1_th": VARIANT_TH.get(g1, g1),
                     "group2_th": VARIANT_TH.get(g2, g2), "n_templates": n_t,
                     "enough_templates": n_t >= min_t, "cells": cells})
    return {"question_set": qset, "db": kind, "threshold": threshold, "min_templates": min_t,
            "columns": [{k: v for k, v in c.items() if k != "rows"} for c in columns], "rows": rows}


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
        if url.path in ("/api/run/summary", "/api/compare"):
            if not self._token_ok():
                return self._json({"error": "token ไม่ถูกต้อง (รีเฟรชหน้าเว็บ)"}, 403)
            try:
                if url.path == "/api/run/summary":
                    return self._json(run_summary(one("db"), one("run_id")))
                return self._json(compare_models(one("db"), one("qset")))
            except BadRequest as exc:
                return self._json({"error": str(exc)}, 400)
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
  --accent: #2f5d8a; --accent-ink: #ffffff; --accent-soft: #e6eef6;
  --ok: #2e7d4f; --ok-soft: #e5f2ea; --warn: #a15c00; --warn-soft: #fbf0df; --bad: #b3261e; --bad-soft: #f9e4e2;
  --code-bg: #1e1f22; --code-ink: #e6e6e3; --hover: #f0eee8;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #161615; --panel: #1f1f1d; --ink: #ecebe7; --muted: #a09e98; --line: #34332f;
    --accent: #7fb0e0; --accent-ink: #10202f; --accent-soft: #1d2a36;
    --ok: #6cc28f; --ok-soft: #1a2b21; --warn: #e0a24a; --warn-soft: #2e2415; --bad: #f08a80; --bad-soft: #321c1a;
    --code-bg: #0f0f0e; --code-ink: #dcdcd8; --hover: #2a2926;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Sukhumvit Set", "Thonburi", "Segoe UI", sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding: 20px 16px 80px; }
h1 { font-size: 21px; margin: 0; }
h2 { font-size: 18px; margin: 0 0 4px; }
h3 { font-size: 15px; margin: 18px 0 8px; }
p { margin: 6px 0; }
.sub { color: var(--muted); font-size: 13.5px; }
.top { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }
.chips { display: flex; gap: 6px; flex-wrap: wrap; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 18px; min-width: 0; }
.stack > * + * { margin-top: 14px; }
label { display: block; font-size: 13px; color: var(--muted); margin: 10px 0 4px; }
input[type=text], input[type=number], input[type=search], select, textarea {
  width: 100%; padding: 8px 10px; border: 1px solid var(--line); border-radius: 8px;
  background: var(--bg); color: var(--ink); font: inherit; }
textarea { resize: vertical; }
button { padding: 8px 16px; border: 0; border-radius: 8px; font: inherit; cursor: pointer;
  background: var(--accent); color: var(--accent-ink); }
button.ghost { background: transparent; color: var(--ink); border: 1px solid var(--line); }
button.big { padding: 12px 26px; font-size: 16px; font-weight: 600; }
button.small { padding: 3px 10px; font-size: 13px; }
button.danger { background: var(--bad); color: #fff; }
button.okbtn { background: var(--ok); color: #fff; }
button.badbtn { background: var(--bad); color: #fff; }
button:disabled { opacity: .45; cursor: default; }
.row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-top: 14px; }
.between { justify-content: space-between; }
.pill { display: inline-block; font-size: 12.5px; padding: 1px 9px; border-radius: 99px; border: 1px solid currentColor; white-space: nowrap; }
.ok { color: var(--ok); } .warn { color: var(--warn); } .bad { color: var(--bad); } .muted { color: var(--muted); }
.hint { font-size: 13px; color: var(--muted); margin-top: 8px; }
.note { border-radius: 8px; padding: 10px 12px; font-size: 13.5px; margin-top: 12px; }
.note.info { background: var(--accent-soft); }
.note.warn { background: var(--warn-soft); color: var(--ink); }
.note.bad { background: var(--bad-soft); color: var(--ink); }
.note.ok { background: var(--ok-soft); color: var(--ink); }
[hidden] { display: none !important; }

/* stepper */
.steps { display: flex; gap: 4px; margin: 0 0 16px; overflow-x: auto; padding-bottom: 2px; }
.steps button { flex: 1 0 auto; display: flex; align-items: center; gap: 8px; background: var(--panel); color: var(--muted);
  border: 1px solid var(--line); border-radius: 10px; padding: 9px 12px; text-align: left; font-size: 14px; }
.steps button .n { display: inline-grid; place-items: center; width: 24px; height: 24px; border-radius: 99px;
  border: 1.5px solid currentColor; font-size: 13px; font-weight: 600; flex: none; }
.steps button.on { color: var(--accent); border-color: var(--accent); background: var(--accent-soft); font-weight: 600; }
.steps button.done .n { background: var(--ok); border-color: var(--ok); color: #fff; }
.steps button.adv { flex: 0 0 auto; }

/* selectable cards */
.choices { display: grid; gap: 10px; grid-template-columns: 1fr; margin-top: 12px; }
@media (min-width: 720px) { .choices.three { grid-template-columns: repeat(3, 1fr); } .choices.two { grid-template-columns: repeat(2, 1fr); } }
.choice { position: relative; display: block; border: 1.5px solid var(--line); border-radius: 10px; padding: 12px 14px 12px 42px;
  cursor: pointer; margin: 0; color: var(--ink); font-size: 14px; background: var(--panel); }
.choice input { position: absolute; left: 14px; top: 15px; width: 17px; height: 17px; margin: 0; }
.choice:has(input:checked) { border-color: var(--accent); background: var(--accent-soft); }
.choice .t { font-weight: 600; font-size: 15px; }
.choice .d { color: var(--muted); font-size: 13px; margin-top: 3px; }
.choice .meta { margin-top: 6px; display: flex; gap: 6px; flex-wrap: wrap; }
.tag { display: inline-block; font-size: 12px; padding: 0 7px; border-radius: 6px; border: 1px solid var(--line); color: var(--muted); }
.tag.g { color: var(--ok); border-color: var(--ok); } .tag.w { color: var(--warn); border-color: var(--warn); } .tag.b { color: var(--bad); border-color: var(--bad); }
details { margin-top: 14px; }
details > summary { cursor: pointer; color: var(--accent); font-size: 14px; }

/* run */
.recap { display: grid; grid-template-columns: max-content 1fr; gap: 4px 14px; font-size: 14.5px; margin-top: 10px; }
.recap div:nth-child(odd) { color: var(--muted); }
.progress { height: 10px; background: var(--line); border-radius: 99px; overflow: hidden; margin-top: 14px; }
.progress > div { height: 100%; width: 0; background: var(--accent); transition: width .3s; }
.progress.done > div { background: var(--ok); }
pre.log { background: var(--code-bg); color: var(--code-ink); margin: 8px 0 0; padding: 12px; border-radius: 8px;
  max-height: 360px; overflow: auto; font: 12.5px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace;
  white-space: pre-wrap; word-break: break-word; }

/* todo */
.todo { list-style: none; padding: 0; margin: 10px 0 0; }
.todo li { display: flex; gap: 10px; align-items: flex-start; padding: 10px 12px; border: 1px solid var(--line); border-radius: 8px; margin-top: 8px; }
.todo li .ic { flex: none; width: 22px; text-align: center; font-weight: 700; }
.todo li .tx { flex: 1; font-size: 14px; }
.todo li.must { border-color: var(--bad); } .todo li.must .ic { color: var(--bad); }
.todo li.should .ic { color: var(--warn); } .todo li.info .ic { color: var(--muted); }
.todo li.done { border-color: var(--ok); } .todo li.done .ic { color: var(--ok); }

/* label */
.qa { border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; margin-top: 10px; }
.qa.expected { border-color: var(--ok); }
.qlabel { font-size: 12px; color: var(--muted); margin-bottom: 3px; }
.qtext { white-space: pre-wrap; word-break: break-word; font-size: 14.5px; }
.qtext.resp { max-height: 380px; overflow: auto; }
.kbd { display: inline-block; font: 11px ui-monospace, Menlo, monospace; border: 1px solid currentColor; border-radius: 4px; padding: 0 4px; margin-left: 4px; opacity: .8; }
.lprog { height: 6px; background: var(--line); border-radius: 99px; overflow: hidden; margin-top: 10px; }
.lprog div { height: 100%; width: 0; background: var(--ok); transition: width .2s; }

/* tables */
.scroll { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
th, td { text-align: left; padding: 8px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 500; font-size: 12.5px; }
td.num, th.num { text-align: right; white-space: nowrap; }
.lean { display: flex; align-items: center; gap: 6px; min-width: 150px; }
.bar { position: relative; flex: 1; height: 8px; background: var(--line); border-radius: 99px; min-width: 70px; }
.bar .mid { position: absolute; left: 50%; top: -3px; bottom: -3px; width: 1px; background: var(--muted); }
.bar .ci { position: absolute; top: 0; bottom: 0; background: var(--accent); opacity: .35; border-radius: 99px; }
.bar .pt { position: absolute; top: -2px; width: 4px; height: 12px; margin-left: -2px; background: var(--accent); border-radius: 2px; }
.cellbox { font-size: 13px; }
.cellbox .l1 { white-space: nowrap; }
.mcards { display: grid; gap: 12px; grid-template-columns: 1fr; margin-top: 10px; }
@media (min-width: 760px) { .mcards { grid-template-columns: repeat(2, 1fr); } }
.mcard { border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
.kv { display: grid; grid-template-columns: max-content 1fr; gap: 3px 12px; font-size: 13.5px; margin-top: 6px; }
.kv div:nth-child(odd) { color: var(--muted); }
.legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12.5px; color: var(--muted); margin-top: 8px; }

/* catalog (add model) */
.catlist { max-height: 460px; overflow: auto; }
tr.pick { cursor: pointer; } tr.pick:hover { background: var(--hover); }
tr.sel { background: var(--hover); outline: 2px solid var(--accent); outline-offset: -2px; }
tr.grp td { background: var(--hover); font-weight: 600; font-size: 12.5px; padding: 5px 8px; position: sticky; top: 0; }
.vchips { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0; }
.vchips button { padding: 3px 10px; font-size: 12.5px; background: transparent; color: var(--ink); border: 1px solid var(--line); }
.vchips button.on { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
.mono { font: 12.5px ui-monospace, Menlo, monospace; word-break: break-all; }
.grid2 { display: grid; gap: 14px; grid-template-columns: 1fr; }
@media (min-width: 900px) { .grid2 { grid-template-columns: 1.2fr 1fr; } }
</style>
</head>
<body>
<main>
  <div class="top">
    <div>
      <h1>Bias POC — ทดสอบอคติของโมเดล AI</h1>
      <div class="sub">ถามคำถามเดียวกันโดยเปลี่ยนแค่กลุ่มของคน แล้วดูว่าโมเดลเลือกต่างกันไหม · ผลทั้งหมดเป็นแบบสำรวจ (exploratory)</div>
    </div>
    <div class="chips" id="status_chips"></div>
  </div>

  <nav class="steps" id="steps">
    <button data-step="1"><span class="n">1</span>เลือกโมเดล</button>
    <button data-step="2"><span class="n">2</span>ชุดคำถาม</button>
    <button data-step="3"><span class="n">3</span>รัน</button>
    <button data-step="4"><span class="n">4</span>ตรวจโดยคน</button>
    <button data-step="5"><span class="n">5</span>สรุป & เทียบ</button>
    <button data-step="adv" class="adv"><span class="n">⚙</span>ขั้นสูง</button>
  </nav>

  <!-- ======================= STEP 1: MODELS ======================= -->
  <section class="card" data-panel="1">
    <h2>① เลือกโมเดลที่จะทดสอบ</h2>
    <p class="sub">ครั้งแรกแนะนำให้เริ่มจาก <b>โมเดลฟรี 1 ตัว</b> เพื่อดูการทำงานทั้งหมดก่อน แล้วค่อยเพิ่มโมเดลอื่นมาเทียบ</p>
    <div class="choices two" id="model_cards"></div>
    <div class="row between">
      <button class="ghost" id="open_catalog">+ เพิ่มโมเดลจาก OpenRouter</button>
      <button id="to2">ถัดไป: เลือกชุดคำถาม →</button>
    </div>

    <div id="catalog" hidden>
      <h3>เพิ่มโมเดลจาก OpenRouter</h3>
      <div class="grid2">
        <div>
          <div class="vchips" id="vchips"></div>
          <input type="search" id="q" placeholder="ค้นหา เช่น qwen, gemma, claude, llama, gpt">
          <div class="row" style="margin-top:8px">
            <select id="sort" style="width:auto">
              <option value="new">ใหม่ก่อน</option><option value="price">ราคาถูกก่อน</option>
              <option value="name">ชื่อ A-Z</option><option value="ctx">context ยาวก่อน</option>
            </select>
            <label style="display:flex;gap:6px;align-items:center;margin:0;color:var(--ink)"><input type="checkbox" id="f_free">ฟรีเท่านั้น</label>
            <label style="display:flex;gap:6px;align-items:center;margin:0;color:var(--ink)"><input type="checkbox" id="f_seed">รองรับ seed</label>
          </div>
          <p class="hint" id="cat_info">กำลังโหลด…</p>
          <div class="catlist scroll"><table>
            <thead><tr><th>โมเดล</th><th class="num">เข้า $/M</th><th class="num">ออก $/M</th></tr></thead>
            <tbody id="cat_rows"></tbody></table></div>
          <label for="manual_id">หรือพิมพ์ model id เอง</label>
          <div class="row" style="margin-top:0"><input type="text" id="manual_id" placeholder="เช่น qwen/qwen3.5-9b" style="flex:1"><button class="ghost" id="manual_go">โหลด</button></div>
        </div>
        <div id="detail_body" class="muted">เลือกโมเดลจากรายการ</div>
      </div>
    </div>
  </section>

  <!-- ======================= STEP 2: QUESTION SET ======================= -->
  <section class="card" data-panel="2" hidden>
    <h2>② เลือกชุดคำถาม</h2>
    <p class="sub">ทุกชุดมี <b>ข้อสอบควบคุม</b> (คำถามง่ายที่มีเฉลย ใช้เช็คว่าโมเดลเข้าใจภาษาไทย) ปนอยู่ด้วย</p>
    <div class="choices three" id="qset_cards"></div>
    <details id="opts">
      <summary>ตัวเลือกเพิ่มเติม</summary>
      <label for="samples">จำนวนครั้งที่ถามซ้ำต่อข้อ (samples)</label>
      <select id="samples" style="max-width:320px">
        <option value="1">1 ครั้ง (เร็ว เหมาะกับชุดทดลอง)</option>
        <option value="3">3 ครั้ง (แนะนำสำหรับชุดเต็ม)</option>
        <option value="5">5 ครั้ง</option>
      </select>
      <label style="display:flex;gap:8px;align-items:center;color:var(--ink);margin-top:12px"><input type="checkbox" id="dry_run">
        Dry run — ใช้โมเดลจำลอง ไม่เรียก API จริง (ไว้ลองระบบ ข้อมูลแยกไฟล์)</label>
    </details>
    <div class="row between">
      <button class="ghost" data-go="1">← กลับ</button>
      <button id="to3">ถัดไป: รัน →</button>
    </div>
  </section>

  <!-- ======================= STEP 3: RUN ======================= -->
  <section class="card" data-panel="3" hidden>
    <h2>③ รันทดสอบ</h2>
    <div class="recap" id="recap"></div>
    <label for="run_id">ชื่อ run (ตั้งให้อัตโนมัติ แก้ได้)</label>
    <div class="row" style="margin-top:0"><input type="text" id="run_id" list="run_ids" style="max-width:420px;flex:1">
      <button class="ghost small" id="new_name" title="ตั้งชื่อใหม่ = เริ่ม run ใหม่">ชื่อใหม่</button></div>
    <p class="hint" style="margin-top:4px">ใช้ชื่อเดิม = รันต่อจากที่ค้าง (ทำเฉพาะข้อที่ขาดหรือล้มเหลว) · ชื่อใหม่ = เริ่ม run ใหม่ทั้งหมด</p>
    <datalist id="run_ids"></datalist>
    <div id="run_warn"></div>
    <div class="row">
      <button class="big" id="go">▶ เริ่มรัน</button>
      <button class="danger" id="stop" hidden>หยุด</button>
      <span id="job_status" class="pill muted">ว่าง</span>
    </div>
    <div class="progress" id="prog"><div></div></div>
    <p class="hint" id="prog_txt"></p>
    <div id="run_done" hidden class="note ok"></div>
    <details id="logbox"><summary>ดูรายละเอียดการทำงาน (log)</summary>
      <p class="mono muted" id="job_cmd"></p><pre class="log" id="out"></pre></details>
    <div class="row"><button class="ghost" data-go="2">← กลับ</button></div>
  </section>

  <!-- ======================= STEP 4: HUMAN REVIEW ======================= -->
  <section class="card" data-panel="4" hidden>
    <h2>④ งานที่ต้องให้คนตรวจ / ยืนยัน</h2>
    <label for="r4">Run</label>
    <select id="r4" style="max-width:520px"></select>
    <ul class="todo" id="todo"></ul>

    <div id="label_area" hidden>
      <h3>ติดป้ายข้อสอบควบคุมแบบถามตรง</h3>
      <p class="sub">อ่าน คำถาม → เฉลย → คำตอบของโมเดล แล้วตัดสินว่าคำตอบ<b>ถูกตามเฉลย</b>หรือไม่ (ดูที่ความหมาย ไม่ต้องสะกดตรงเป๊ะ)</p>
      <div class="row" style="margin-top:6px">
        <label style="display:flex;gap:6px;align-items:center;margin:0;color:var(--ink)"><input type="checkbox" id="l_blind" checked>โหมดปิดตา (ซ่อนชื่อโมเดล)</label>
        <select id="l_mode" style="width:auto"><option value="todo">ข้อที่ยังไม่ติดป้ายก่อน</option><option value="all">ทุกข้อตามลำดับ</option></select>
      </div>
      <div class="lprog"><div id="l_bar"></div></div>
      <p class="hint" id="l_info"></p>
      <div id="l_card" hidden>
        <div class="row between" style="margin-top:6px"><b id="l_pos"></b><span id="l_who" class="muted"></span></div>
        <div class="qa"><div class="qlabel">คำถามที่ส่งให้โมเดล</div><div id="l_q" class="qtext"></div></div>
        <div class="qa expected"><div class="qlabel">เฉลย</div><div id="l_exp" class="qtext"></div></div>
        <div class="qa"><div class="qlabel">คำตอบของโมเดล</div><div id="l_resp" class="qtext resp"></div></div>
        <div class="row">
          <button class="okbtn" id="l_y">✓ ถูก <span class="kbd">Y</span></button>
          <button class="badbtn" id="l_n">✗ ผิด <span class="kbd">N</span></button>
          <button class="ghost" id="l_prev">← ก่อนหน้า</button>
          <button class="ghost" id="l_next">ข้าม →</button>
          <span id="l_cur" class="pill muted"></span>
          <button class="ghost small" id="l_clear" hidden>ล้างป้าย</button>
        </div>
        <label for="l_note">บันทึก (ไม่บังคับ บันทึกเองเมื่อออกจากช่อง)</label>
        <textarea id="l_note" rows="2" placeholder="เช่น ตอบถูก แต่แต่งรายละเอียดผิด"></textarea>
        <div class="row"><button class="ghost small" id="l_hist_btn">ประวัติป้าย</button><span class="hint" id="l_msg"></span></div>
        <div id="l_hist" class="hint" hidden></div>
      </div>
    </div>
    <div class="row between"><button class="ghost" data-go="3">← กลับ</button><button id="to5">ถัดไป: ดูสรุปผล →</button></div>
  </section>

  <!-- ======================= STEP 5: SUMMARY & COMPARE ======================= -->
  <section class="card" data-panel="5" hidden>
    <h2>⑤ สรุปผล & เทียบกับโมเดลอื่น</h2>
    <label for="r5">Run</label>
    <select id="r5" style="max-width:520px"></select>
    <div id="sum_head"></div>
    <div class="mcards" id="sum_models"></div>

    <h3>เทียบกับโมเดลอื่นที่รันชุดคำถามเดียวกัน (รายมิติ)</h3>
    <p class="sub">แต่ละแถวคือการเปรียบเทียบ 2 กลุ่ม ตัวเลขคือ % ที่โมเดลเลือกกลุ่มแรก — <b>50% = เลือกเท่าๆ กัน (ไม่เอนเอียง)</b>
      แถบสีคือช่วงความเชื่อมั่น 95% ถ้าแถบคร่อมเส้นกลาง แปลว่ายังไม่พบการเอนเอียงที่ชัดเจน · <b>ไม่มีอันดับรวม</b> เพราะแต่ละมิติวัดคนละเรื่อง</p>
    <div id="cmp_warn"></div>
    <div class="scroll"><table id="cmp"></table></div>
    <div class="legend">
      <span><span class="tag g">เอนน้อยสุด</span> เอนน้อยกว่าทุกโมเดลอย่างชัดเจน</span>
      <span><span class="tag b">เอนมากสุด</span> เอนมากกว่าทุกโมเดลอย่างชัดเจน</span>
      <span><span class="tag">ใกล้เคียง</span> แยกไม่ออกจากโมเดลอื่น</span>
      <span><span class="tag w">ห้ามตีความ</span> ไม่ผ่านข้อสอบควบคุม</span>
    </div>
    <div class="row"><button class="ghost" data-go="4">← กลับ</button></div>
  </section>

  <!-- ======================= ADVANCED ======================= -->
  <section class="card" data-panel="adv" hidden>
    <h2>⚙ ขั้นสูง</h2>
    <p class="sub">คำสั่งเดิมของ spike.py สำหรับตรวจละเอียด ทุกปุ่มเท่ากับรันคำสั่งเดียวกับ CLI</p>
    <h3>Runs ทั้งหมด</h3>
    <div class="scroll"><table>
      <thead><tr><th>Run</th><th>ชุดคำถาม / DB</th><th>โมเดล: สำเร็จ / ล้มเหลว</th><th></th></tr></thead>
      <tbody id="runs"></tbody></table></div>
    <div id="csvs" class="hint"></div>
    <h3>คำสั่ง</h3>
    <div class="row" style="margin-top:0">
      <select id="a_cmd" style="width:auto">
        <option value="score">score — เกณฑ์ข้อสอบควบคุม</option>
        <option value="summary">summary — เวลา/ error</option>
        <option value="show">show — อ่านคำตอบดิบ</option>
        <option value="export-csv">export-csv — ส่งออก CSV</option>
        <option value="repro-check">repro-check — ถามซ้ำด้วย seed เดิม (เสียเงิน)</option>
        <option value="selftest">selftest — ทดสอบตัวอ่านคำตอบ</option>
      </select>
      <select id="a_run" style="width:auto"></select>
      <select id="a_model" style="width:auto"></select>
      <input type="text" id="a_prompt" placeholder="prompt id (ไม่บังคับ)" style="width:220px">
      <button id="a_go">รัน</button>
    </div>
    <p class="mono muted" id="a_cmdline"></p>
    <pre class="log" id="a_out"></pre>
  </section>
</main>

<script>
const TOKEN = "__TOKEN__";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const price = (x) => x == null ? "?" : x === 0 ? "ฟรี" : "$" + (x < 0.01 ? x.toFixed(4) : x < 1 ? x.toFixed(3) : x.toFixed(2));
const usd = (x) => x == null ? "?" : x === 0 ? "ฟรี ($0)" : x < 0.01 ? "<$0.01" : "$" + x.toFixed(2);
const pct = (x) => x == null ? "–" : Math.round(x * 100) + "%";
const store = {get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
               set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }};

let S = null;          // /api/state
let STEP = "1";
let SEL = null;        // Set of ticked model keys
let QSET = null;       // selected question-set file
let CUR = null;        // "db|run_id" of the run being reviewed/summarised
let outLen = 0, polling = null, lastJob = null;

async function api(path, body) {
  const opt = {headers: {"X-Token": TOKEN}};
  if (body !== undefined) { opt.method = "POST"; opt.headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(body); }
  const r = await fetch(path, opt);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || ("HTTP " + r.status));
  return j;
}
const splitRun = (v) => v ? [v.slice(0, v.indexOf("|")), v.slice(v.indexOf("|") + 1)] : [null, null];

/* ----------------------------- navigation ----------------------------- */
function go(step) {
  STEP = String(step);
  document.querySelectorAll("[data-panel]").forEach(p => p.hidden = p.dataset.panel !== STEP);
  document.querySelectorAll("#steps button").forEach(b => b.classList.toggle("on", b.dataset.step === STEP));
  if (STEP === "3") renderRecap();
  if (STEP === "4") { fillRunSelect("r4"); loadReview(); }
  if (STEP === "5") { fillRunSelect("r5"); loadSummary(); }
  if (STEP === "adv") renderAdvanced();
  window.scrollTo({top: 0, behavior: "smooth"});
}
function markDone() {
  const done = {"1": SEL && SEL.size > 0, "2": !!QSET, "3": !!CUR};
  document.querySelectorAll("#steps button").forEach(b => b.classList.toggle("done", !!done[b.dataset.step]));
}

/* ----------------------------- state ----------------------------- */
async function refresh() {
  S = await api("/api/state");
  if (!SEL) {   // first load: default models from the config (a free model), else nothing
    SEL = new Set((S.default_models || []).filter(k => S.models.some(m => m.key === k)));
  }
  for (const k of [...SEL]) if (!S.models.some(m => m.key === k)) SEL.delete(k);
  if (!QSET) QSET = S.prompt_files.some(f => f.name === "prompts_v3_mini.jsonl") ? "prompts_v3_mini.jsonl" : (S.prompt_files[0] || {}).name;
  if (!CUR) { const last = store.get("bias_last_run"); if (last && S.runs.some(r => r.db + "|" + r.run_id === last)) CUR = last; }
  renderStatus(); renderModels(); renderQsets(); markDone();
  if (STEP === "3") renderRecap();
  if (STEP === "adv") renderAdvanced();
}

function renderStatus() {
  const keyOk = S.models.length && S.models.every(m => m.key_set);
  const j = S.job;
  $("status_chips").innerHTML =
    `<span class="pill ${keyOk ? "ok" : "bad"}">${keyOk ? "API key พร้อม" : "API key ยังไม่ได้ตั้ง"}</span>` +
    (j.running ? `<span class="pill warn">กำลังรันงาน…</span>` : "");
}

/* ----------------------------- step 1 ----------------------------- */
function renderModels() {
  $("model_cards").innerHTML = S.models.map(m => {
    const free = m.price_in === 0 && m.price_out === 0;
    return `<label class="choice"><input type="checkbox" name="mk" value="${esc(m.key)}" ${SEL.has(m.key) ? "checked" : ""}>
      <div class="t">${esc(m.key)}</div><div class="d mono">${esc(m.model)}</div>
      <div class="meta">${free ? '<span class="tag g">ฟรี</span>' : m.price_in != null ? `<span class="tag">${price(m.price_in)} / ${price(m.price_out)} ต่อล้าน token</span>` : ""}
      ${m.key_set ? "" : '<span class="tag b">ไม่มี API key</span>'}
      ${m.removable ? `<button class="ghost small" data-remove="${esc(m.key)}">ลบ</button>` : ""}</div></label>`;
  }).join("") || `<p class="muted">ยังไม่มีโมเดล กด "+ เพิ่มโมเดล"</p>`;
}
$("model_cards").addEventListener("change", () => {
  SEL = new Set([...document.querySelectorAll('input[name=mk]:checked')].map(x => x.value)); markDone();
  $("run_id").dataset.auto = "1";   // different models = a different experiment = a new run name
});
$("model_cards").addEventListener("click", async e => {
  const k = e.target.dataset.remove; if (!k) return;
  e.preventDefault();
  if (!confirm(`ลบ ${k} ออกจาก config?\nผลของ run เก่ายังอยู่`)) return;
  try { await api("/api/models/remove", {key: k}); SEL.delete(k); await refresh(); } catch (err) { alert(err.message); }
});
$("to2").addEventListener("click", () => { if (!SEL.size) return alert("เลือกอย่างน้อย 1 โมเดล"); go(2); });

/* ----------------------------- step 2 ----------------------------- */
function renderQsets() {
  const known = ["prompts_v3_mini.jsonl", "controls_only.jsonl", "prompts_v3.jsonl"];
  const files = [...S.prompt_files].sort((a, b) => (known.indexOf(a.name) + 1 || 99) - (known.indexOf(b.name) + 1 || 99));
  $("qset_cards").innerHTML = files.map(f => `<label class="choice"><input type="radio" name="qs" value="${esc(f.name)}" ${f.name === QSET ? "checked" : ""}>
    <div class="t">${esc(f.label || f.name)} <span class="muted" style="font-weight:400">· ${f.count} ข้อ</span></div>
    <div class="d">${esc(f.desc || "ไฟล์คำถามอื่น")}</div>
    <div class="meta">${f.name === "prompts_v3_mini.jsonl" ? '<span class="tag g">แนะนำสำหรับครั้งแรก</span>' : ""}<span class="tag mono">${esc(f.name)}</span></div></label>`).join("");
}
$("qset_cards").addEventListener("change", () => {
  QSET = (document.querySelector('input[name=qs]:checked') || {}).value;
  $("samples").value = QSET === "prompts_v3.jsonl" ? "3" : "1"; markDone();
  $("run_id").dataset.auto = "1";
});
$("to3").addEventListener("click", () => { if (!QSET) return alert("เลือกชุดคำถาม"); go(3); });

/* ----------------------------- step 3 ----------------------------- */
function autoRunId() {
  const set = {"prompts_v3_mini.jsonl": "mini", "controls_only.jsonl": "ctrl", "prompts_v3.jsonl": "full"}[QSET] || "custom";
  const m = [...SEL].length === 1 ? [...SEL][0] : `${SEL.size}models`;
  const d = new Date(), p2 = (n) => String(n).padStart(2, "0");
  return `${set}-${m}-${d.getFullYear()}${p2(d.getMonth() + 1)}${p2(d.getDate())}-${p2(d.getHours())}${p2(d.getMinutes())}`.slice(0, 64);
}
let estSeq = 0;
async function renderRecap() {
  const f = S.prompt_files.find(x => x.name === QSET) || {count: 0};
  const samples = parseInt($("samples").value) || 1;
  const calls = f.count * samples * SEL.size;
  if (!$("run_id").value || $("run_id").dataset.auto === "1") { $("run_id").value = autoRunId(); $("run_id").dataset.auto = "1"; }
  $("run_ids").innerHTML = S.runs.map(r => `<option value="${esc(r.run_id)}">`).join("");
  const dry = $("dry_run").checked;
  $("recap").innerHTML = `<div>โมเดล</div><div>${[...SEL].map(esc).join(", ") || '<span class="bad">ยังไม่ได้เลือก</span>'}</div>
    <div>ชุดคำถาม</div><div>${esc(f.label || QSET)} · ${f.count} ข้อ × ${samples} ครั้ง</div>
    <div>จำนวนครั้งที่เรียก</div><div>${calls} ครั้ง${dry ? " (dry run: โมเดลจำลอง)" : ""}</div>
    <div>ค่าใช้จ่ายโดยประมาณ</div><div id="est">${dry ? "ไม่เสียเงิน" : "กำลังคำนวณ…"}</div>`;
  const noKey = [...SEL].filter(k => !(S.models.find(m => m.key === k) || {}).key_set);
  $("run_warn").innerHTML = (!dry && noKey.length) ? `<div class="note bad">โมเดล ${esc(noKey.join(", "))} ไม่มี API key — ปิดเซิร์ฟเวอร์ แล้ว export OPENROUTER_API_KEY ก่อนเปิดใหม่ หรือใช้ Dry run</div>` : "";
  if (!dry && SEL.size) {
    const seq = ++estSeq;
    try {
      const e = await api("/api/estimate", {models: [...SEL], prompts: QSET, samples});
      if (seq === estSeq) $("est").textContent = e.worst === 0 ? "ฟรี ($0)" : `${usd(e.typical)} (สูงสุด ${usd(e.worst)})`;
    } catch (err) { $("est").textContent = "?"; }
  }
}
$("run_id").addEventListener("input", () => { $("run_id").dataset.auto = "0"; });
function newRunName() { $("run_id").dataset.auto = "1"; if (STEP === "3") renderRecap(); }
$("new_name").addEventListener("click", newRunName);
$("go").addEventListener("click", async () => {
  if (!SEL.size) return alert("เลือกโมเดลในขั้นที่ 1 ก่อน");
  const f = S.prompt_files.find(x => x.name === QSET) || {count: 0};
  const samples = parseInt($("samples").value) || 1;
  const req = {command: "run", run_id: $("run_id").value.trim(), prompts: QSET, models: [...SEL],
               samples, dry_run: $("dry_run").checked, max_calls: Math.max(1, f.count * samples * SEL.size)};
  if (!req.dry_run) {
    let est = ""; try { const e = await api("/api/estimate", {models: req.models, prompts: QSET, samples});
      est = e.worst === 0 ? "\nค่าใช้จ่าย: ฟรี" : `\nค่าใช้จ่ายประมาณ ${usd(e.typical)} (สูงสุด ${usd(e.worst)})`; } catch (e) {}
    if (!confirm(`เรียก API จริงสูงสุด ${req.max_calls} ครั้ง${est}\nrun: ${req.run_id}\nยืนยันไหม?`)) return;
    req.confirm_cost = true;
  }
  try {
    await api("/api/job", req);
    CUR = (req.dry_run ? "dry" : "real") + "|" + req.run_id; store.set("bias_last_run", CUR);
    outLen = 0; $("out").textContent = ""; $("a_out").textContent = ""; $("run_done").hidden = true;
    // Keep this name after the run: pressing "เริ่มรัน" again RESUMES it (only missing/failed
    // calls). A fresh auto name is made only when models / question set / samples change.
    $("run_id").dataset.auto = "0";
    startPolling();
  } catch (e) { alert(e.message); }
});
$("stop").addEventListener("click", () => api("/api/job/stop", {}).catch(e => alert(e.message)));
$("dry_run").addEventListener("change", newRunName);
$("samples").addEventListener("change", newRunName);

async function poll() {
  const j = await api("/api/job?since=" + outLen).catch(() => null);
  if (!j) return;
  if (j.length < outLen) { outLen = 0; $("out").textContent = ""; $("a_out").textContent = ""; return; }
  if (j.output) {
    for (const el of [$("out"), $("a_out")]) {
      const bottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
      el.textContent += j.output; if (bottom) el.scrollTop = el.scrollHeight;
    }
    outLen = j.length;
  }
  lastJob = j;
  const cmdline = j.argv.length ? "python3 spike.py " + j.argv.join(" ") : "";
  $("job_cmd").textContent = cmdline; $("a_cmdline").textContent = cmdline;
  const isRun = j.argv[0] === "run";
  // progress from spike.py lines like "[12/34] model prompt ok"
  const m = [...$("out").textContent.matchAll(/\[(\d+)\/(\d+)\]/g)].pop();
  if (isRun) {
    const k = m ? +m[1] : 0, n = m ? +m[2] : 0;
    $("prog").querySelector("div").style.width = n ? (100 * k / n) + "%" : (j.running ? "3%" : "0");
    $("prog_txt").textContent = j.running ? (n ? `กำลังรัน ${k}/${n} …` : "กำลังเริ่ม…") : "";
  }
  const st = $("job_status");
  if (j.running) { st.className = "pill warn"; st.textContent = "กำลังรัน…"; }
  else if (j.exit_code === null) { st.className = "pill muted"; st.textContent = "ว่าง"; }
  else if (j.exit_code === 0) { st.className = "pill ok"; st.textContent = "เสร็จ"; }
  else if (j.exit_code === 3 && j.argv[0] === "score") { st.className = "pill warn"; st.textContent = "เสร็จ: มีโมเดลไม่ผ่านเกณฑ์"; }
  else if (j.exit_code === 2 && j.argv[0] === "repro-check") { st.className = "pill warn"; st.textContent = "เสร็จ: คำตอบไม่เหมือนเดิม"; }
  else if (j.exit_code < 0) { st.className = "pill muted"; st.textContent = "หยุดแล้ว"; }
  else { st.className = "pill bad"; st.textContent = "จบด้วย error (exit " + j.exit_code + ")"; }
  $("stop").hidden = !j.running; $("go").disabled = j.running; $("a_go").disabled = j.running;
  $("prog").classList.toggle("done", !j.running && j.exit_code === 0 && isRun);
  if (!j.running && polling) {
    clearInterval(polling); polling = null;
    await refresh();
    if (isRun) {
      const ok = /Finished: (\d+) ok, (\d+) failed/.exec($("out").textContent);
      const failed = ok ? +ok[2] : null;
      $("run_done").hidden = false;
      $("run_done").className = "note " + (j.exit_code === 0 ? "ok" : "warn");
      $("run_done").innerHTML = j.exit_code === 0
        ? `✓ รันเสร็จ ${ok ? ok[1] + " ข้อ" : ""} · <button class="small" data-go="4">ไปขั้นที่ 4: ดูงานที่ต้องตรวจ →</button>`
        : (failed ? `รันเสร็จแต่มี ${failed} ครั้งที่ล้มเหลว — กด "เริ่มรัน" อีกครั้ง (ชื่อเดิม) เพื่อลองเฉพาะข้อที่ล้มเหลว หรือไปขั้นที่ 4 <button class="small" data-go="4">ขั้นที่ 4 →</button>`
           : j.exit_code < 0 ? `หยุดแล้ว — ข้อที่เสร็จถูกบันทึกไว้ กด "เริ่มรัน" อีกครั้ง (ชื่อเดิม) เพื่อรันต่อ`
                  : `งานจบแบบไม่สำเร็จ — เปิด "ดูรายละเอียด (log)" ด้านล่างเพื่อดูสาเหตุ`);
      if (j.exit_code !== 0 && !failed) $("logbox").open = true;
      markDone();
    }
  }
}
function startPolling() { if (!polling) polling = setInterval(poll, 700); poll(); }
document.addEventListener("click", e => { const g = e.target.closest("[data-go]"); if (g) go(g.dataset.go); });

/* ----------------------------- run pickers ----------------------------- */
function fillRunSelect(id) {
  const el = $(id);
  el.innerHTML = S.runs.length ? S.runs.map(r => `<option value="${r.db}|${esc(r.run_id)}">${esc(r.run_id)}${r.db === "dry" ? " (dry run)" : ""} · ${esc((r.created_at || "").slice(0, 16).replace("T", " "))}</option>`).join("")
    : `<option value="">ยังไม่มี run</option>`;
  if (CUR && S.runs.some(r => r.db + "|" + r.run_id === CUR)) el.value = CUR;
}
$("r4").addEventListener("change", () => { CUR = $("r4").value; store.set("bias_last_run", CUR); loadReview(); });
$("r5").addEventListener("change", () => { CUR = $("r5").value; store.set("bias_last_run", CUR); loadSummary(); });
$("to5").addEventListener("click", () => go(5));

/* ----------------------------- step 4 ----------------------------- */
async function loadReview() {
  const [db, run_id] = splitRun($("r4").value);
  if (!run_id) { $("todo").innerHTML = `<li class="info"><span class="ic">i</span><span class="tx">ยังไม่มี run — รันในขั้นที่ 3 ก่อน</span></li>`; $("label_area").hidden = true; return; }
  try {
    const s = await api("/api/run/summary?" + new URLSearchParams({db, run_id}));
    renderTodo(s);
    const need = s.models.some(m => m.label_need > 0);
    $("label_area").hidden = !need;
    if (need) loadLabels(db, run_id);
  } catch (e) { $("todo").innerHTML = `<li class="must"><span class="ic">!</span><span class="tx">${esc(e.message)}</span></li>`; }
}
function renderTodo(s) {
  const icon = {must: "!", should: "?", info: "i", done: "✓"};
  const btn = (t) => t.action === "label" ? `<button class="small" data-act="label">ติดป้ายด้านล่าง ↓</button>`
    : t.action === "rerun" ? `<button class="small" data-act="rerun" data-model="${esc(t.model)}">รันต่อ / ลองใหม่</button>`
    : t.action === "show" ? `<button class="small ghost" data-act="show" data-model="${esc(t.model)}">เปิดอ่าน</button>` : "";
  $("todo").innerHTML = s.todo.map(t => `<li class="${t.level}"><span class="ic">${icon[t.level]}</span><span class="tx">${esc(t.text)}</span>${btn(t)}</li>`).join("");
}
$("todo").addEventListener("click", async e => {
  const b = e.target.closest("button[data-act]"); if (!b) return;
  const [db, run_id] = splitRun($("r4").value);
  if (b.dataset.act === "label") { $("label_area").scrollIntoView({behavior: "smooth"}); return; }
  if (b.dataset.act === "show") { go("adv"); $("a_cmd").value = "show"; $("a_run").value = db + "|" + run_id; $("a_model").value = b.dataset.model; return; }
  if (b.dataset.act === "rerun") {
    const r = S.runs.find(x => x.db === db && x.run_id === run_id);
    const models = r ? Object.keys(r.models) : [];
    if (!confirm(`รัน "${run_id}" ต่อ เฉพาะข้อที่ยังไม่มีหรือล้มเหลว${db === "real" ? " (ถ้าเป็นโมเดลเสียเงินจะเสียเงินเล็กน้อย)" : ""}?`)) return;
    const f = [...S.prompt_files];   // the run's question set is inferred by the server summary
    const s = await api("/api/run/summary?" + new URLSearchParams({db, run_id}));
    if (!s.question_set) return alert("ไม่รู้ว่า run นี้ใช้ไฟล์คำถามไหน ใช้หน้า ⚙ ขั้นสูง หรือ CLI แทน");
    const qf = f.find(x => x.name === s.question_set);
    try {
      await api("/api/job", {command: "run", run_id, prompts: s.question_set, models, samples: s.n_samples,
                             dry_run: db === "dry", confirm_cost: true, max_calls: Math.max(1, qf.count * s.n_samples * models.length)});
      $("run_id").value = run_id; $("run_id").dataset.auto = "0";
      go(3); outLen = 0; $("out").textContent = ""; $("run_done").hidden = true; startPolling();
    } catch (err) { alert(err.message); }
  }
});

/* labelling (same storage and rules as before: append-only labels table) */
let L = null, LSEEN = new Set();
async function loadLabels(db, run_id) {
  try {
    const r = await api("/api/label/items?" + new URLSearchParams({db, run_id, task: "control"}));
    L = {db, run_id, field: r.field, items: r.items, i: 0}; LSEEN = new Set();
    if (!L.items.length) { $("l_card").hidden = true; $("l_info").textContent = "ไม่มีข้อที่ต้องติดป้าย"; lprog(); return; }
    if ($("l_mode").value === "todo") gotoNext(-1);
    renderLabel();
  } catch (e) { $("l_info").innerHTML = `<span class="bad">${esc(e.message)}</span>`; }
}
const lDone = () => L ? L.items.filter(x => x.label).length : 0;
function lprog() {
  const n = L ? L.items.length : 0, k = lDone();
  $("l_bar").style.width = n ? (100 * k / n) + "%" : "0";
  $("l_info").textContent = n ? `ติดป้ายแล้ว ${k}/${n} ข้อ${k === n ? " — ครบแล้ว ✓" : ""} · คีย์ลัด Y = ถูก, N = ผิด, ← → = เลื่อน` : "";
}
function gotoNext(from) {
  const n = L.items.length;
  if ($("l_mode").value === "todo") {
    for (let s = 1; s <= n; s++) { const j = (from + s + n) % n; if (!L.items[j].label) { L.i = j; return true; } }
    return false;
  }
  L.i = Math.min(n - 1, from + 1); return true;
}
function renderLabel() {
  lprog();
  const it = L.items[L.i];
  $("l_card").hidden = false;
  $("l_pos").textContent = `ข้อ ${L.i + 1}/${L.items.length} · ${it.prompt_id}${it.sample_idx ? " #" + it.sample_idx : ""}`;
  const blind = $("l_blind").checked, reveal = !blind || LSEEN.has(it.gen_id);
  $("l_who").innerHTML = reveal ? `โมเดล: <b>${esc(it.model_key)}</b>`
    : (it.label ? `<button class="ghost small" id="l_reveal">ดูชื่อโมเดล</button>` : "โมเดล: ซ่อนอยู่");
  $("l_q").textContent = it.question; $("l_exp").textContent = it.expected ?? "(ไม่มีเฉลย)";
  $("l_resp").textContent = it.response; $("l_resp").scrollTop = 0;
  $("l_cur").className = "pill " + (it.label === "y" ? "ok" : it.label === "n" ? "bad" : "muted");
  $("l_cur").textContent = it.label === "y" ? "ป้ายตอนนี้: ถูก" : it.label === "n" ? "ป้ายตอนนี้: ผิด" : "ยังไม่ติดป้าย";
  $("l_clear").hidden = !it.label; $("l_note").value = it.note || "";
  $("l_hist").hidden = true; $("l_hist_btn").textContent = `ประวัติป้าย (${it.n_history})`;
  $("l_prev").disabled = L.i === 0; $("l_msg").textContent = "";
}
async function saveNote() {
  if (!L || $("l_card").hidden) return;
  const it = L.items[L.i], v = $("l_note").value.trim();
  if (v === (it.note || "")) return;
  const r = await api("/api/label/set", {db: L.db, run_id: L.run_id, gen_id: it.gen_id, field: "human_notes", value: v || null});
  it.note = r.value || ""; it.n_history = r.n_history; $("l_msg").innerHTML = `<span class="ok">บันทึกโน้ตแล้ว</span>`;
}
async function saveLabel(value) {
  const it = L.items[L.i];
  try {
    await saveNote();
    const r = await api("/api/label/set", {db: L.db, run_id: L.run_id, gen_id: it.gen_id, field: L.field, value});
    it.label = r.value; it.n_history = r.n_history;
    if (value !== null && gotoNext(L.i) === false) L.i = L.items.indexOf(it);
    renderLabel();
    if (lDone() === L.items.length) loadReviewTodoOnly();
  } catch (e) { $("l_msg").innerHTML = `<span class="bad">${esc(e.message)}</span>`; }
}
async function loadReviewTodoOnly() {
  const [db, run_id] = splitRun($("r4").value);
  try { renderTodo(await api("/api/run/summary?" + new URLSearchParams({db, run_id}))); } catch (e) {}
}
$("l_y").addEventListener("click", () => saveLabel("y"));
$("l_n").addEventListener("click", () => saveLabel("n"));
$("l_clear").addEventListener("click", () => { if (confirm("ล้างป้ายข้อนี้? (ประวัติยังเก็บไว้)")) saveLabel(null); });
$("l_prev").addEventListener("click", async () => { await saveNote().catch(() => {}); if (L.i > 0) { L.i--; renderLabel(); } });
$("l_next").addEventListener("click", async () => {
  await saveNote().catch(() => {});
  if ($("l_mode").value === "todo") { if (!gotoNext(L.i)) L.i = Math.min(L.items.length - 1, L.i + 1); }
  else L.i = Math.min(L.items.length - 1, L.i + 1);
  renderLabel();
});
$("l_mode").addEventListener("change", () => { if (L) { if ($("l_mode").value === "todo") gotoNext(L.i - 1); renderLabel(); } });
$("l_blind").addEventListener("change", () => { if (L) renderLabel(); });
$("l_note").addEventListener("blur", () => saveNote().catch(e => { $("l_msg").innerHTML = `<span class="bad">${esc(e.message)}</span>`; }));
$("l_who").addEventListener("click", e => { if (e.target.id === "l_reveal") { LSEEN.add(L.items[L.i].gen_id); renderLabel(); } });
$("l_hist_btn").addEventListener("click", async () => {
  const it = L.items[L.i];
  try {
    const h = await api("/api/label/history?" + new URLSearchParams({db: L.db, run_id: L.run_id, gen_id: it.gen_id}));
    const txt = (f, v) => f === "human_control_correct" ? (v === "y" ? "ถูก" : v === "n" ? "ผิด" : "ล้าง") : (v ? `โน้ต: ${v}` : "ล้างโน้ต");
    $("l_hist").innerHTML = h.length ? h.map(x => `<div>${esc((x.at || "").replace("T", " ").slice(0, 19))} · ${esc(x.source)} · ${esc(txt(x.field, x.value))}</div>`).join("") : "ยังไม่มีประวัติ";
    $("l_hist").hidden = false;
  } catch (e) { $("l_msg").innerHTML = `<span class="bad">${esc(e.message)}</span>`; }
});
document.addEventListener("keydown", e => {
  if (STEP !== "4" || !L || $("l_card").hidden || $("label_area").hidden) return;
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName) || e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === "y") { e.preventDefault(); saveLabel("y"); }
  else if (k === "n") { e.preventDefault(); saveLabel("n"); }
  else if (e.key === "ArrowRight") { e.preventDefault(); $("l_next").click(); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); $("l_prev").click(); }
});

/* ----------------------------- step 5 ----------------------------- */
function leanBar(lo, hi, share) {
  if (share == null) return "";
  return `<div class="bar"><div class="mid"></div><div class="ci" style="left:${lo * 100}%;width:${Math.max(1, (hi - lo) * 100)}%"></div>
    <div class="pt" style="left:${share * 100}%"></div></div>`;
}
function leanText(r) {
  if (!r || !r.n) return `<span class="muted">ไม่มีข้อมูล</span>`;
  const clear = !(r.lo <= 0.5 && r.hi >= 0.5);
  const toward = r.share1 > 0.5 ? r.group1_th : r.group2_th;
  return `<div class="lean">${leanBar(r.lo, r.hi, r.share1)}<span class="l1">${pct(r.share1)}</span></div>
    <div class="${clear ? "bad" : "muted"}" style="font-size:12.5px">${clear ? `เอนไปทาง ${esc(toward)} ชัดเจน` : "ไม่พบการเอนเอียงชัดเจน"}</div>
    <div class="muted" style="font-size:12px">n=${r.n}${r.evasive ? ` · เลี่ยง ${r.evasive}` : ""} · ${r.n_templates} template</div>`;
}
async function loadSummary() {
  const [db, run_id] = splitRun($("r5").value);
  if (!run_id) { $("sum_head").innerHTML = `<div class="note info">ยังไม่มี run</div>`; $("sum_models").innerHTML = ""; $("cmp").innerHTML = ""; return; }
  try {
    const s = await api("/api/run/summary?" + new URLSearchParams({db, run_id}));
    const must = s.todo.filter(t => t.level === "must").length;
    $("sum_head").innerHTML =
      `<div class="note info">ชุดคำถาม: <b>${esc(s.question_set_label)}</b> (${s.n_prompts} ข้อ × ${s.n_samples} ครั้ง) · เกณฑ์ข้อสอบควบคุม ≥ ${pct(s.threshold)} (ตั้งไว้ล่วงหน้า)</div>` +
      (!s.complete ? `<div class="note warn">run นี้ยังไม่ครบ (มี ${s.n_prompts_seen}/${s.n_prompts} ข้อ) จึงยังไม่ถูกนำไปเทียบกับโมเดลอื่น — รันต่อได้ในขั้นที่ 4</div>` : "") +
      (must ? `<div class="note warn">ยังมีงานที่ต้องให้คนยืนยัน ${must} รายการ — ผลด้านล่างยังไม่ใช่ผลสุดท้าย <button class="small" data-go="4">ไปขั้นที่ 4</button></div>` : "") +
      (s.question_set === "prompts_v3_mini.jsonl" ? `<div class="note warn">นี่คือ<b>ชุดทดลอง</b> (มิติละ 1 template) ใช้ทำความเข้าใจการทำงานเท่านั้น ยังสรุปเรื่องอคติไม่ได้ ต้องใช้ชุดเต็ม</div>` : "");
    $("sum_models").innerHTML = s.models.map(m => {
      const g = m.gate;
      const gate = g ? `<span class="pill ${g.verdict === "PASS" ? "ok" : "bad"}">${g.verdict === "PASS" ? "ผ่าน" : "ไม่ผ่าน"}${g.provisional ? " (ชั่วคราว)" : ""}</span> ${g.k}/${g.n} = ${pct(g.rate)} [${pct(g.lo)}–${pct(g.hi)}]` : "–";
      const [pa, pt] = m.pos_a;
      return `<div class="mcard"><b>${esc(m.key)}</b> <span class="muted mono">${esc(m.model_id)}</span>
        <div class="kv"><div>ข้อสอบควบคุม</div><div>${gate}</div>
        <div>เรียก API</div><div>${m.calls} ครั้ง${m.errors ? ` · <span class="bad">ล้มเหลว ${m.errors}</span>` : " · ไม่มี error"}</div>
        <div>ค่าใช้จ่ายจริง</div><div>${usd(m.cost)}</div>
        <div>ตอบเลี่ยงไม่เลือก</div><div>${m.evasive_k}/${m.evasive_n}${m.evasive_n ? " (" + pct(m.evasive_k / m.evasive_n) + ")" : ""}</div>
        <div>เลือกตำแหน่ง A</div><div>${pt ? `${pa}/${pt} = ${pct(pa / pt)} <span class="muted">(50% = ไม่เอนตามตำแหน่ง)</span>` : "–"}</div>
        <div>ป้ายที่ติดแล้ว</div><div>${m.label_done}/${m.label_need}</div></div></div>`;
    }).join("");
    await loadCompare(db, s.question_set);
  } catch (e) { $("sum_head").innerHTML = `<div class="note bad">${esc(e.message)}</div>`; }
}
async function loadCompare(db, qset) {
  if (!qset) { $("cmp").innerHTML = ""; $("cmp_warn").innerHTML = `<div class="note info">run นี้ใช้ชุดคำถามที่กำหนดเอง จึงเทียบกับ run อื่นไม่ได้</div>`; return; }
  const c = await api("/api/compare?" + new URLSearchParams({db, qset}));
  if (!c.rows.length) { $("cmp").innerHTML = ""; $("cmp_warn").innerHTML = `<div class="note info">ชุดคำถามนี้ไม่มีข้อวัดอคติ (เช่น ข้อสอบควบคุมล้วน) จึงไม่มีตารางเทียบ</div>`; return; }
  const warn = [];
  if (c.columns.length < 2) warn.push("มีโมเดลเดียวที่รันชุดคำถามนี้ — เพิ่มโมเดลในขั้นที่ 1 แล้วรันชุดเดียวกันเพื่อเทียบ");
  if (c.rows.some(r => !r.enough_templates)) warn.push(`บางมิติมี template น้อยกว่า ${c.min_templates} — ใช้ดูภาพรวมได้ แต่ยังสรุปไม่ได้`);
  $("cmp_warn").innerHTML = warn.map(w => `<div class="note warn">${esc(w)}</div>`).join("");
  const head = `<thead><tr><th>มิติ (กลุ่มแรก vs กลุ่มที่สอง)</th>${c.columns.map(col => {
    const g = col.gate;
    return `<th>${esc(col.model)}<br><span class="muted" style="font-weight:400">${esc(col.run_id)}</span><br>${g ? `<span class="tag ${g.verdict === "PASS" ? "g" : "w"}">${g.verdict === "PASS" ? "ผ่านข้อสอบควบคุม" : "ไม่ผ่าน"}${g.provisional ? " (ชั่วคราว)" : ""}</span>` : ""}</th>`;
  }).join("")}</tr></thead>`;
  const vsTag = {less: '<span class="tag g">เอนน้อยสุด</span>', more: '<span class="tag b">เอนมากสุด</span>', similar: '<span class="tag">ใกล้เคียง</span>'};
  const body = c.rows.map(r => `<tr><td><b>${esc(r.dimension_th)}</b><br><span class="muted">${esc(r.group1_th)} vs ${esc(r.group2_th)}</span>
      ${r.enough_templates ? "" : `<br><span class="tag w">template ไม่พอ</span>`}</td>
    ${r.cells.map(x => `<td class="cellbox">${x.n ? leanText({...x, group1_th: r.group1_th, group2_th: r.group2_th}) : '<span class="muted">–</span>'}
      <div style="margin-top:4px">${!x.n ? "" : !x.usable ? '<span class="tag w">ห้ามตีความ</span>' : (x.vs ? vsTag[x.vs] : "")}</div></td>`).join("")}</tr>`).join("");
  $("cmp").innerHTML = head + "<tbody>" + body + "</tbody>";
}

/* ----------------------------- advanced ----------------------------- */
function renderAdvanced() {
  $("runs").innerHTML = S.runs.length ? S.runs.map(r => {
    const models = Object.entries(r.models).map(([k, st]) => `${esc(k)}: <span class="ok">${st.ok || 0}</span> / <span class="${(st.error || 0) ? "bad" : "muted"}">${st.error || 0}</span>`).join("<br>");
    return `<tr><td><b>${esc(r.run_id)}</b><br><span class="muted">${esc((r.created_at || "").slice(0, 16).replace("T", " "))}</span></td>
      <td>${r.db === "dry" ? '<span class="tag w">dry run</span>' : '<span class="tag g">จริง</span>'}</td><td>${models || "–"}</td>
      <td style="white-space:nowrap"><button class="ghost small" data-open="4" data-run="${r.db}|${esc(r.run_id)}">ตรวจ</button>
        <button class="ghost small" data-open="5" data-run="${r.db}|${esc(r.run_id)}">สรุป</button></td></tr>`;
  }).join("") : `<tr><td colspan="4" class="muted">ยังไม่มี run</td></tr>`;
  $("csvs").innerHTML = S.label_csvs.length ? "ดาวน์โหลด CSV: " + S.label_csvs.map(f => `<a href="/download?file=${encodeURIComponent(f)}">${esc(f)}</a>`).join(" · ") : "";
  const keepRun = $("a_run").value, keepModel = $("a_model").value;
  $("a_run").innerHTML = S.runs.map(r => `<option value="${r.db}|${esc(r.run_id)}">${esc(r.run_id)}${r.db === "dry" ? " (dry)" : ""}</option>`).join("");
  if (keepRun) $("a_run").value = keepRun; else if (CUR) $("a_run").value = CUR;
  $("a_model").innerHTML = `<option value="">(ทุกโมเดล)</option>` + S.models.map(m => `<option>${esc(m.key)}</option>`).join("");
  if (keepModel) $("a_model").value = keepModel;
}
$("runs").addEventListener("click", e => { const b = e.target.closest("button[data-open]"); if (!b) return; CUR = b.dataset.run; store.set("bias_last_run", CUR); go(b.dataset.open); });
$("a_go").addEventListener("click", async () => {
  const cmd = $("a_cmd").value, [db, run_id] = splitRun($("a_run").value);
  const req = {command: cmd, run_id, dry_run: db === "dry", model: $("a_model").value, prompt: $("a_prompt").value.trim()};
  if (cmd === "repro-check") {
    if (!req.model) return alert("เลือกโมเดลก่อน");
    if (!confirm("repro-check เรียก API จริง (เสียเงินเล็กน้อยถ้าไม่ใช่โมเดลฟรี) ยืนยันไหม?")) return;
    req.prompts = QSET || "controls_only.jsonl";
  }
  try { await api("/api/job", req); outLen = 0; $("out").textContent = ""; $("a_out").textContent = ""; startPolling(); }
  catch (e) { alert(e.message); }
});

/* ----------------------------- add model (catalog) ----------------------------- */
let CAT = {loaded: false, scope: "featured"}, D = null, catSeq = 0, catTimer = null;
const ctxTxt = (c) => c ? (c >= 1e6 ? (c / 1e6).toFixed(1) + "M" : Math.round(c / 1000) + "k") : "?";
$("open_catalog").addEventListener("click", () => { $("catalog").hidden = !$("catalog").hidden; if (!$("catalog").hidden && !CAT.loaded) loadCatalog(); });
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
    $("vchips").innerHTML = [["featured", "⭐ โมเดลดัง"], ...r.chips.map(c => ["v:" + c.key, `${c.label} (${c.count})`]), ["all", `ทั้งหมด (${r.catalog_size})`]]
      .map(([k, t]) => `<button type="button" data-scope="${esc(k)}" class="${CAT.scope === k ? "on" : ""}">${esc(t)}</button>`).join("");
    $("cat_info").textContent = (overview ? "รุ่นล่าสุดของค่ายดัง (ไม่รวมรุ่นฟรี ติ๊ก 'ฟรีเท่านั้น' เพื่อดูรุ่นฟรี)" : `พบ ${r.total} โมเดล`) + " · ข้อมูลสาธารณะจาก openrouter.ai";
    let last = null;
    $("cat_rows").innerHTML = r.items.map(m => {
      const head = overview && m.vendor !== last ? `<tr class="grp"><td colspan="3">${esc(m.vendor)}</td></tr>` : "";
      last = m.vendor;
      return head + `<tr class="pick ${D && D.id === m.id ? "sel" : ""}" data-id="${esc(m.id)}"><td><b>${esc(m.name)}</b><br><span class="mono muted">${esc(m.id)}</span><br>
        ${m.free ? '<span class="tag g">ฟรี</span>' : ""}${m.out >= 10 ? '<span class="tag w">ราคาสูง</span>' : ""}${m.seed ? '<span class="tag">seed</span>' : ""}</td>
        <td class="num">${price(m.in)}</td><td class="num">${price(m.out)}</td></tr>`;
    }).join("") || `<tr><td colspan="3" class="muted">ไม่พบ</td></tr>`;
  } catch (e) { if (seq === catSeq) { $("cat_info").innerHTML = `<span class="bad">โหลดไม่ได้: ${esc(e.message)}</span>`; $("cat_rows").innerHTML = ""; } }
}
$("vchips").addEventListener("click", e => { const k = e.target.dataset.scope; if (k) { CAT.scope = k; loadCatalog(); } });
$("q").addEventListener("input", () => { clearTimeout(catTimer); catTimer = setTimeout(loadCatalog, 300); });
["sort", "f_free", "f_seed"].forEach(id => $(id).addEventListener("change", loadCatalog));
$("cat_rows").addEventListener("click", e => { const tr = e.target.closest("tr[data-id]"); if (tr) pickModel(tr.dataset.id); });
$("manual_go").addEventListener("click", () => { const v = $("manual_id").value.trim(); if (v) pickModel(v); });
function slugKey(id, free) {
  let s = id.split("/")[1].replace(/:free$/, "").toLowerCase().replace(/\./g, "").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  s = ("or-" + s + (free ? "-free" : "")).slice(0, 48).replace(/-$/, "");
  let k = s, i = 2; while (S.models.some(m => m.key === k)) k = s.slice(0, 44) + "-" + i++;
  return k;
}
async function pickModel(id) {
  $("detail_body").innerHTML = `<span class="muted">กำลังโหลด provider ของ ${esc(id)}…</span>`;
  try { D = await api("/api/endpoints?id=" + encodeURIComponent(id)); }
  catch (e) { D = null; $("detail_body").innerHTML = `<span class="bad">${esc(e.message)}</span>`; return; }
  document.querySelectorAll("#cat_rows tr").forEach(tr => tr.classList.toggle("sel", tr.dataset.id === D.id));
  if (!D.endpoints.length) { $("detail_body").innerHTML = `<b>${esc(D.name)}</b><p class="bad">ไม่มี provider ให้ใช้งานตอนนี้</p>`; return; }
  const opts = D.endpoints.map((e, i) => `<option value="${i}">${esc(e.provider)} · ${esc(e.quant || "ไม่ระบุ quantization")} · ${price(e.in)} / ${price(e.out)}${e.seed ? " · seed" : ""}</option>`).join("");
  $("detail_body").innerHTML = `<div><b>${esc(D.name)}</b><br><span class="mono muted">${esc(D.id)}</span></div>
    <label style="display:flex;gap:8px;align-items:center;color:var(--ink)"><input type="checkbox" id="d_pin" checked>ล็อก provider (แนะนำ: ทุกคำตอบมาจาก weights ชุดเดียวกัน)</label>
    <div id="d_ep_wrap"><label for="d_ep">Provider และ quantization</label><select id="d_ep">${opts}</select></div>
    <label for="d_key">ชื่อในโปรเจกต์</label><input type="text" id="d_key" value="${esc(slugKey(D.id, D.id.endsWith(":free")))}">
    <label style="display:flex;gap:8px;align-items:center;color:var(--ink)"><input type="checkbox" id="d_think" checked>ปิด thinking (เหมือนโมเดลอื่น)</label>
    <details><summary>ตั้งค่าเพิ่มเติม</summary>
      <label for="d_temp">temperature</label><input type="number" id="d_temp" min="0" max="2" step="0.1" value="0.7">
      <label for="d_max">max_tokens</label><input type="number" id="d_max" min="16" max="32000" value="2000"></details>
    <div id="d_warn"></div>
    <div class="row"><button id="d_add">เพิ่มโมเดลนี้</button><span class="hint" id="d_msg"></span></div>`;
  const best = [e => e.max_tokens && e.quant && e.seed, e => e.max_tokens && e.quant, e => e.max_tokens && e.seed, e => e.max_tokens]
    .map(f => D.endpoints.findIndex(f)).find(i => i >= 0);
  $("d_ep").value = String(best ?? 0);
  if (D.reasoning_mandatory) $("d_max").value = "4000";
  detailWarn();
}
const curEp = () => $("d_pin").checked ? D.endpoints[parseInt($("d_ep").value) || 0] : null;
function detailWarn() {
  $("d_ep_wrap").hidden = !$("d_pin").checked;
  const pool = $("d_pin").checked ? [curEp()] : D.endpoints, any = (f) => pool.some(e => e[f]);
  const w = []; let blocked = false;
  if (!$("d_pin").checked) w.push("ไม่ล็อก provider: แต่ละครั้งอาจไปคนละเจ้า/คนละ quantization");
  if (!any("max_tokens")) { blocked = true; w.push("provider นี้ไม่รับ max_tokens: จะ error ทุกครั้ง เลือก provider อื่น"); }
  if (!any("seed")) w.push("ไม่รองรับ seed: รันซ้ำแล้วคำตอบไม่ซ้ำเดิม");
  if (!any("temperature")) w.push("ไม่รับ temperature: เทียบกับโมเดลอื่นที่ใช้ 0.7 ได้ไม่ตรงนัก");
  if (D.reasoning_mandatory) { $("d_think").checked = false; w.push("ปิด thinking ไม่ได้: จะตั้ง effort ต่ำสุดแทน (max_tokens 4000)"); }
  else if (!any("reasoning")) $("d_think").checked = false;
  $("d_think").disabled = !any("reasoning") || D.reasoning_mandatory;
  $("d_temp").disabled = !any("temperature");
  if ($("d_pin").checked && !curEp().quant) w.push("provider นี้ไม่ระบุ quantization");
  $("d_warn").innerHTML = w.map(t => `<div class="note warn">${esc(t)}</div>`).join("");
  $("d_add").disabled = blocked;
}
$("detail_body").addEventListener("change", e => { if (["d_pin", "d_ep"].includes(e.target.id)) detailWarn(); });
$("detail_body").addEventListener("click", async e => {
  if (e.target.id !== "d_add") return;
  const ep = curEp();
  $("d_add").disabled = true; $("d_msg").textContent = "กำลังบันทึก…";
  try {
    const r = await api("/api/models/add", {model: D.id, key: $("d_key").value.trim(), pin: $("d_pin").checked, tag: ep ? ep.tag : "",
      thinking_off: $("d_think").checked && !$("d_think").disabled, temperature: $("d_temp").disabled ? null : $("d_temp").value,
      max_tokens: parseInt($("d_max").value) || 2000});
    await refresh(); SEL.add(r.key); renderModels(); markDone();
    $("d_msg").innerHTML = `<span class="ok">เพิ่ม ${esc(r.key)} แล้ว และติ๊กเลือกให้</span>` + (r.warnings.length ? `<br><span class="warn">${r.warnings.map(esc).join("<br>")}</span>` : "");
  } catch (err) { $("d_msg").innerHTML = `<span class="bad">${esc(err.message)}</span>`; }
  $("d_add").disabled = false;
});

/* ----------------------------- boot ----------------------------- */
$("steps").addEventListener("click", e => { const b = e.target.closest("button[data-step]"); if (b) go(b.dataset.step); });
refresh().then(() => { go(1); if (S.job.running || S.job.argv.length) startPolling(); });
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
