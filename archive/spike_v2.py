#!/usr/bin/env python3
"""
spike.py (v2, local-model support) - Phase-0 spike for the Thai-context LLM bias evaluation POC.

What this script is FOR (it does NOT measure bias):
  1. Confirm each model endpoint is reachable and stable from this machine.
  2. Measure latency, error rate and output length per model on a few Thai prompts.
  3. Give a rough first look at refusal behaviour (crude hint + CSV for manual labelling).
  4. Prove the plumbing the real study needs: resumable runs, retries, raw-response storage.
  5. (local models) Check whether a fixed seed really gives reproducible output: `repro-check`.

Bias scoring (rubric, labels, analysis) is a later stage and is intentionally NOT here.

Typical local setup (Ollama example; verify commands against the Ollama docs for your version):
  ollama serve                      # or just open the Ollama app; API at http://localhost:11434
  ollama pull llama3.1:8b
  pip install httpx pyyaml

Usage:
  # 1) Plumbing test with fake models (no network, no model needed, separate database file):
  python spike.py run --run-id t1 --dry-run --mock-fail-rate 0.8 --yes
  python spike.py run --run-id t1 --dry-run --yes        # re-run: only failed calls are retried

  # 2) Is the seed honoured by your local server? (calls the same prompt twice with the same seed)
  python spike.py repro-check --model llama31-8b

  # 3) Real run against the local server:
  python spike.py run --run-id spike-local-001
  python spike.py summary --run-id spike-local-001
  python spike.py show --run-id spike-local-001
  python spike.py export-csv --run-id spike-local-001 --out spike-local-001.csv

Never commit bias_poc*.db (raw outputs) unless you intend to publish them; see .gitignore.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import sqlite3
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml

# ----------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------
DEFAULT_CONFIG = "config/models.yaml"
DEFAULT_PROMPTS = "data/spike_prompts.jsonl"
DEFAULT_DB = "bias_poc.db"
DRY_RUN_DB = "bias_poc_dry.db"  # dry-run (mock) data never mixes with real evidence
ANTHROPIC_VERSION = "2023-06-01"
MAX_ATTEMPTS = 5  # per call, including the first try
DEFAULT_TIMEOUT_S = 120.0
FALLBACK_REPRO_SEED = 1234  # used by repro-check when the model has no seed_base

# CRUDE refusal hints. Only a pointer for manual review, NOT a measurement:
# it misses polite refusals and can misfire (e.g. a valid "cannot conclude" answer).
# The CSV export has empty columns so a human can label refusals properly.
REFUSAL_HINTS = [
    "ขออภัย", "ไม่สามารถช่วย", "ไม่ขอตอบ", "ขอไม่ตอบ", "ไม่สามารถให้ความเห็น",
    "i can't help", "i cannot help", "i can't assist", "i cannot assist",
    "i'm sorry", "i am sorry", "i won't", "i'm not able to",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    config_hash  TEXT NOT NULL,
    n_samples    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS generations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL,
    prompt_id      TEXT NOT NULL,
    model_key      TEXT NOT NULL,
    model_id       TEXT NOT NULL,
    sample_idx     INTEGER NOT NULL,
    status         TEXT NOT NULL,          -- 'ok' | 'error'
    response_text  TEXT,
    raw_json       TEXT,                   -- full server response, kept as evidence
    prompt_json    TEXT NOT NULL,          -- snapshot of the prompt object (text + metadata)
    http_status    INTEGER,
    error          TEXT,
    latency_ms     INTEGER,
    input_tokens   INTEGER,
    output_tokens  INTEGER,
    attempts       INTEGER,
    created_at     TEXT NOT NULL,
    seed           INTEGER,                -- seed sent with the request (if any)
    note           TEXT,                   -- free-text model provenance from the config
    UNIQUE (run_id, prompt_id, model_key, sample_idx)
);
"""

# Columns added after the first version; older database files are upgraded in place.
MIGRATION_COLUMNS = [("seed", "INTEGER"), ("note", "TEXT")]


# ----------------------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------------------
@dataclass
class ModelCfg:
    key: str                              # short name used in the DB, e.g. "typhoon2-8b"
    provider: str                         # "openai_compatible" | "anthropic"
    model: str                            # model id/tag sent to the API
    api_key_env: Optional[str] = None     # NAME of the env var holding the key; omit for local servers
    base_url: Optional[str] = None
    rpm: int = 30                         # max requests per minute (this script's own limiter)
    concurrency: int = 2                  # max in-flight requests for this model
    temperature: Optional[float] = None   # null in YAML -> omitted from the request
    max_tokens: int = 600
    max_tokens_param: str = "max_tokens"  # some newer OpenAI models want max_completion_tokens
    extra_body: Optional[dict] = None     # merged into the request body as-is
    timeout_s: float = DEFAULT_TIMEOUT_S  # per-request timeout; local models may need much longer
    seed_base: Optional[int] = None       # if set, sample i is sent with seed = seed_base + i
    note: Optional[str] = None            # provenance text (quantization, runtime version, digest)


@dataclass
class CallResult:
    text: str
    raw: Optional[dict]
    http_status: Optional[int]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    latency_ms: int


class CallError(Exception):
    """A failed API call. `retryable` decides whether the caller backs off and tries again."""

    def __init__(self, message: str, http_status: Optional[int] = None,
                 retryable: bool = False, retry_after: Optional[float] = None):
        super().__init__(message)
        self.http_status = http_status
        self.retryable = retryable
        self.retry_after = retry_after


@dataclass
class RunContext:
    client: Optional[httpx.AsyncClient]
    db: sqlite3.Connection
    run_id: str
    api_keys: dict            # model_key -> key string, or None for keyless (local) models
    limiters: dict
    semaphores: dict
    dry_run: bool
    mock_fail_rate: float
    total: int
    done: int = 0
    ok: int = 0
    failed: int = 0
    counters: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------
# Config / prompt loading
# ----------------------------------------------------------------------------------------
def load_models(path: Path) -> list[ModelCfg]:
    if not path.exists():
        sys.exit(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models: list[ModelCfg] = []
    seen: set = set()
    for item in data.get("models", []):
        try:
            m = ModelCfg(**item)
        except TypeError as exc:
            sys.exit(f"Bad model entry {item.get('key', item)}: {exc}")
        if m.provider not in ("openai_compatible", "anthropic"):
            sys.exit(f"Model '{m.key}': unknown provider '{m.provider}'")
        if m.provider == "openai_compatible" and not m.base_url:
            sys.exit(f"Model '{m.key}': base_url is required for openai_compatible")
        if m.provider == "anthropic" and not m.api_key_env:
            sys.exit(f"Model '{m.key}': api_key_env is required for the anthropic provider")
        if m.seed_base is not None and m.provider != "openai_compatible":
            sys.exit(f"Model '{m.key}': seed_base is only supported for openai_compatible")
        if m.key in seen:
            sys.exit(f"Duplicate model key: {m.key}")
        seen.add(m.key)
        models.append(m)
    if not models:
        sys.exit(f"No models defined in {path}")
    return models


def load_prompts(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"Prompt file not found: {path}")
    prompts: list[dict] = []
    seen: set = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):  # allow blank lines and '#' comment lines
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            sys.exit(f"{path}:{lineno}: invalid JSON: {exc}")
        if "id" not in obj or "text" not in obj:
            sys.exit(f"{path}:{lineno}: each prompt needs 'id' and 'text'")
        if obj["id"] in seen:
            sys.exit(f"{path}:{lineno}: duplicate prompt id '{obj['id']}'")
        seen.add(obj["id"])
        prompts.append(obj)
    if not prompts:
        sys.exit(f"No prompts found in {path}")
    return prompts


def compute_config_hash(models: list[ModelCfg], prompts: list[dict]) -> str:
    """Hash of everything that changes WHAT is asked or HOW (not rpm/concurrency/timeout/note).
    A run id can only be resumed if this hash matches, so one run = one consistent experiment."""
    payload = {
        "models": sorted(
            [
                {
                    "key": m.key, "provider": m.provider, "model": m.model,
                    "base_url": m.base_url, "temperature": m.temperature,
                    "max_tokens": m.max_tokens, "max_tokens_param": m.max_tokens_param,
                    "extra_body": m.extra_body, "seed_base": m.seed_base,
                }
                for m in models
            ],
            key=lambda d: d["key"],
        ),
        "prompts": prompts,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def seed_for(m: ModelCfg, sample_idx: int) -> Optional[int]:
    """Seed sent for a given sample: distinct per sample, but fixed across re-runs."""
    return None if m.seed_base is None else m.seed_base + sample_idx


# ----------------------------------------------------------------------------------------
# Rate limiting
# ----------------------------------------------------------------------------------------
class RateLimiter:
    """Spaces request START times at least 60/rpm seconds apart (one limiter per model).
    Create it inside the running event loop."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._next_slot = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_slot - now)
            self._next_slot = max(now, self._next_slot) + self.interval
        if delay > 0:
            await asyncio.sleep(delay)


# ----------------------------------------------------------------------------------------
# Provider adapters
# ----------------------------------------------------------------------------------------
def _parse_retry_after(resp: httpx.Response) -> Optional[float]:
    value = resp.headers.get("retry-after")
    try:
        return float(value) if value else None
    except ValueError:
        return None


async def _post_json(client: httpx.AsyncClient, url: str, headers: dict, body: dict,
                     timeout_s: float):
    """POST and classify failures: 429/5xx/network/timeout -> retryable; other 4xx -> permanent."""
    started = time.monotonic()
    try:
        resp = await client.post(url, headers=headers, json=body, timeout=timeout_s)
    except httpx.TransportError as exc:  # includes timeouts and connection errors
        raise CallError(f"transport error: {type(exc).__name__}: {exc}", retryable=True)
    latency_ms = int((time.monotonic() - started) * 1000)

    if resp.status_code == 429 or resp.status_code >= 500:
        raise CallError(f"HTTP {resp.status_code}: {resp.text[:300]}",
                        http_status=resp.status_code, retryable=True,
                        retry_after=_parse_retry_after(resp))
    if resp.status_code >= 400:
        raise CallError(f"HTTP {resp.status_code}: {resp.text[:500]}",
                        http_status=resp.status_code, retryable=False)
    try:
        data = resp.json()
    except ValueError:
        raise CallError("response body is not valid JSON",
                        http_status=resp.status_code, retryable=False)
    return data, resp.status_code, latency_ms


async def call_openai_compatible(client: httpx.AsyncClient, m: ModelCfg, api_key: Optional[str],
                                 prompt_text: str, seed: Optional[int] = None) -> CallResult:
    url = (m.base_url or "").rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:  # local servers usually need no key, so the header is only sent when a key exists
        headers["Authorization"] = f"Bearer {api_key}"
    body: dict[str, Any] = {
        "model": m.model,
        "messages": [{"role": "user", "content": prompt_text}],
        m.max_tokens_param: m.max_tokens,
    }
    if m.temperature is not None:
        body["temperature"] = m.temperature
    if seed is not None:
        body["seed"] = seed
    if m.extra_body:
        body.update(m.extra_body)

    data, status, latency_ms = await _post_json(client, url, headers, body, m.timeout_s)
    try:
        text = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        raise CallError(f"unexpected response shape: {json.dumps(data)[:300]}",
                        http_status=status, retryable=False)
    usage = data.get("usage") or {}
    return CallResult(text=text, raw=data, http_status=status,
                      input_tokens=usage.get("prompt_tokens"),
                      output_tokens=usage.get("completion_tokens"),
                      latency_ms=latency_ms)


async def call_anthropic(client: httpx.AsyncClient, m: ModelCfg,
                         api_key: str, prompt_text: str) -> CallResult:
    url = (m.base_url or "https://api.anthropic.com").rstrip("/") + "/v1/messages"
    headers = {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION,
               "content-type": "application/json"}
    body: dict[str, Any] = {
        "model": m.model,
        "max_tokens": m.max_tokens,
        "messages": [{"role": "user", "content": prompt_text}],
    }
    if m.temperature is not None:
        body["temperature"] = m.temperature
    if m.extra_body:
        body.update(m.extra_body)

    data, status, latency_ms = await _post_json(client, url, headers, body, m.timeout_s)
    blocks = data.get("content")
    if not isinstance(blocks, list):
        raise CallError(f"unexpected response shape: {json.dumps(data)[:300]}",
                        http_status=status, retryable=False)
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    usage = data.get("usage") or {}
    return CallResult(text=text, raw=data, http_status=status,
                      input_tokens=usage.get("input_tokens"),
                      output_tokens=usage.get("output_tokens"),
                      latency_ms=latency_ms)


async def call_mock(m: ModelCfg, prompt_text: str, fail_rate: float) -> CallResult:
    """Fake model for --dry-run: exercises limiter, retries, storage and resume without a model."""
    await asyncio.sleep(random.uniform(0.02, 0.15))
    if random.random() < fail_rate:
        raise CallError("mock transient failure", http_status=503,
                        retryable=True, retry_after=0.02)
    text = f"[mock:{m.key}] echo: {prompt_text[:40]}"
    return CallResult(text=text, raw={"mock": True}, http_status=200,
                      input_tokens=max(1, len(prompt_text) // 3),
                      output_tokens=max(1, len(text) // 3),
                      latency_ms=random.randint(100, 900))


# ----------------------------------------------------------------------------------------
# Persistence (every call is committed immediately so a crash loses at most in-flight calls)
# ----------------------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    # upgrade databases created by the first version of this script
    existing = {row[1] for row in conn.execute("PRAGMA table_info(generations)")}
    for column, coltype in MIGRATION_COLUMNS:
        if column not in existing:
            conn.execute(f"ALTER TABLE generations ADD COLUMN {column} {coltype}")
    conn.commit()
    return conn


def save_generation(ctx: RunContext, m: ModelCfg, prompt: dict, sample_idx: int, *,
                    status: str, attempts: int, seed: Optional[int],
                    result: Optional[CallResult] = None,
                    error: Optional[CallError] = None) -> None:
    ctx.db.execute(
        """INSERT OR REPLACE INTO generations
           (run_id, prompt_id, model_key, model_id, sample_idx, status, response_text, raw_json,
            prompt_json, http_status, error, latency_ms, input_tokens, output_tokens,
            attempts, created_at, seed, note)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            ctx.run_id, prompt["id"], m.key, m.model, sample_idx, status,
            result.text if result else None,
            json.dumps(result.raw, ensure_ascii=False) if result and result.raw else None,
            json.dumps(prompt, ensure_ascii=False),
            (result.http_status if result else (error.http_status if error else None)),
            str(error) if error else None,
            result.latency_ms if result else None,
            result.input_tokens if result else None,
            result.output_tokens if result else None,
            attempts, now_iso(), seed, m.note,
        ),
    )
    ctx.db.commit()


# ----------------------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------------------
async def execute_call(ctx: RunContext, m: ModelCfg, prompt: dict, sample_idx: int) -> bool:
    """One (model, prompt, sample) unit of work with rate limiting and retry/backoff."""
    last_error: Optional[CallError] = None
    attempts = 0
    seed = seed_for(m, sample_idx)
    async with ctx.semaphores[m.key]:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            attempts = attempt
            await ctx.limiters[m.key].wait()  # the limit applies to every attempt, retries included
            try:
                if ctx.dry_run:
                    result = await call_mock(m, prompt["text"], ctx.mock_fail_rate)
                elif m.provider == "anthropic":
                    result = await call_anthropic(ctx.client, m, ctx.api_keys[m.key], prompt["text"])
                else:
                    result = await call_openai_compatible(ctx.client, m, ctx.api_keys[m.key],
                                                          prompt["text"], seed)
                save_generation(ctx, m, prompt, sample_idx, status="ok", attempts=attempts,
                                seed=seed, result=result)
                ctx.done += 1
                ctx.ok += 1
                print(f"[{ctx.done}/{ctx.total}] {m.key:<18} {prompt['id']}#{sample_idx} "
                      f"ok {result.latency_ms}ms (attempts={attempts})", flush=True)
                return True
            except CallError as exc:
                last_error = exc
                if not exc.retryable or attempt == MAX_ATTEMPTS:
                    break
                # honour Retry-After when the server sends it, otherwise exponential backoff
                delay = exc.retry_after if exc.retry_after is not None \
                    else min(60.0, 2 ** attempt + random.random())
                await asyncio.sleep(delay)

    save_generation(ctx, m, prompt, sample_idx, status="error", attempts=attempts,
                    seed=seed, error=last_error)
    ctx.done += 1
    ctx.failed += 1
    print(f"[{ctx.done}/{ctx.total}] {m.key:<18} {prompt['id']}#{sample_idx} "
          f"ERROR after {attempts} attempts: {str(last_error)[:120]}", flush=True)
    return False


async def run_async(ctx_args: dict, todo: list) -> RunContext:
    active: list[ModelCfg] = ctx_args["active"]
    # limiters/semaphores must be created inside the running loop
    limiters = {m.key: RateLimiter(m.rpm) for m in active}
    semaphores = {m.key: asyncio.Semaphore(max(1, m.concurrency)) for m in active}
    async with httpx.AsyncClient() as client:
        ctx = RunContext(client=client, db=ctx_args["db"], run_id=ctx_args["run_id"],
                         api_keys=ctx_args["api_keys"], limiters=limiters,
                         semaphores=semaphores, dry_run=ctx_args["dry_run"],
                         mock_fail_rate=ctx_args["mock_fail_rate"], total=len(todo))
        await asyncio.gather(*(execute_call(ctx, m, p, i) for m, p, i in todo))
    return ctx


# ----------------------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------------------
def resolve_db(args: argparse.Namespace) -> str:
    if args.db:
        return args.db
    return DRY_RUN_DB if args.dry_run else DEFAULT_DB


def cmd_run(args: argparse.Namespace) -> int:
    models = load_models(Path(args.config))
    prompts = load_prompts(Path(args.prompts))

    if args.model:
        wanted = set(args.model)
        unknown = wanted - {m.key for m in models}
        if unknown:
            sys.exit(f"Unknown model key(s): {', '.join(sorted(unknown))}")
    else:
        wanted = {m.key for m in models}

    # Collect API keys from the environment (never printed, never stored).
    # Local models have no api_key_env: they run keyless (None).
    api_keys: dict = {}
    active: list[ModelCfg] = []
    for m in models:
        if m.key not in wanted:
            continue
        if args.dry_run or not m.api_key_env:
            api_keys[m.key] = None
            active.append(m)
            continue
        key = os.environ.get(m.api_key_env)
        if not key:
            print(f"WARNING: env var {m.api_key_env} is not set; skipping model '{m.key}'")
            continue
        api_keys[m.key] = key
        active.append(m)
    if not active:
        sys.exit("No runnable models (missing API keys?). Use --dry-run to test the plumbing.")

    config_hash = compute_config_hash(models, prompts)
    db = open_db(resolve_db(args))

    row = db.execute("SELECT config_hash FROM runs WHERE run_id=?", (args.run_id,)).fetchone()
    if row and row[0] != config_hash and not args.force:
        sys.exit(f"Run '{args.run_id}' already exists with a DIFFERENT models/prompts configuration.\n"
                 f"Use a new --run-id (recommended: one run = one consistent experiment), "
                 f"or pass --force to override.")
    if not row:
        db.execute("INSERT INTO runs (run_id, created_at, config_hash, n_samples) VALUES (?,?,?,?)",
                   (args.run_id, now_iso(), config_hash, args.samples))
        db.commit()

    # Work still to do = everything without an 'ok' row (errors are retried on re-run).
    done = {(r[0], r[1], r[2]) for r in db.execute(
        "SELECT prompt_id, model_key, sample_idx FROM generations WHERE run_id=? AND status='ok'",
        (args.run_id,))}
    todo = [(m, p, i) for m in active for p in prompts for i in range(args.samples)
            if (p["id"], m.key, i) not in done]

    print(f"Run '{args.run_id}' -> database {resolve_db(args)}"
          f"{'  (DRY RUN, mock models)' if args.dry_run else ''}")
    for m in active:
        n = sum(1 for t in todo if t[0].key == m.key)
        print(f"  {m.key:<18} {m.provider:<17} {m.model:<40} calls to make: {n}")
    print(f"  total calls to make: {len(todo)} (already done and skipped: {len(done)})")
    if not todo:
        print("Nothing to do.")
        return 0
    if len(todo) > args.max_calls:
        sys.exit(f"Planned {len(todo)} calls exceeds --max-calls {args.max_calls}. "
                 f"Raise the limit deliberately if that is what you want.")

    # Only paid (keyed) models need the confirmation prompt; local-only runs cost nothing.
    paid = [m for m in active if m.api_key_env]
    if paid and not args.dry_run and not args.yes:
        print("Real API calls cost money; check each provider's current pricing yourself.")
        if not sys.stdin.isatty():
            sys.exit("Not an interactive terminal; pass --yes to confirm.")
        if input("Proceed? [y/N] ").strip().lower() != "y":
            print("Aborted.")
            return 1

    try:
        ctx = asyncio.run(run_async(
            {"active": active, "db": db, "run_id": args.run_id, "api_keys": api_keys,
             "dry_run": args.dry_run, "mock_fail_rate": args.mock_fail_rate}, todo))
    except KeyboardInterrupt:
        print("\nInterrupted. Finished calls are saved; run the same command again to resume.")
        return 130

    print(f"\nFinished: {ctx.ok} ok, {ctx.failed} failed. "
          f"{'Re-run the same command to retry failures. ' if ctx.failed else ''}"
          f"Next: python spike.py summary --run-id {args.run_id}"
          f"{' --dry-run' if args.dry_run else ''}")
    return 1 if ctx.failed else 0


async def _repro_async(m: ModelCfg, api_key: Optional[str], prompt_text: str,
                       seed: int, repeats: int) -> list[str]:
    outputs: list[str] = []
    async with httpx.AsyncClient() as client:
        for _ in range(repeats):
            result = await call_openai_compatible(client, m, api_key, prompt_text, seed)
            outputs.append(result.text)
    return outputs


def cmd_repro_check(args: argparse.Namespace) -> int:
    """Send the SAME prompt N times with the SAME seed and compare the outputs.
    Identical output = the server honours the seed on this machine (needed for reproducible runs).
    It says nothing about other machines, runtime versions or hardware."""
    models = load_models(Path(args.config))
    m = next((x for x in models if x.key == args.model), None)
    if m is None:
        sys.exit(f"Unknown model key: {args.model}")
    if m.provider != "openai_compatible":
        sys.exit("repro-check only supports openai_compatible (local) models")
    prompts = load_prompts(Path(args.prompts))
    prompt = next((p for p in prompts if p["id"] == args.prompt), None) if args.prompt else prompts[0]
    if prompt is None:
        sys.exit(f"Unknown prompt id: {args.prompt}")
    api_key = None
    if m.api_key_env:
        api_key = os.environ.get(m.api_key_env)
        if not api_key:
            sys.exit(f"env var {m.api_key_env} is not set")
    seed = m.seed_base if m.seed_base is not None else FALLBACK_REPRO_SEED

    print(f"repro-check: model={m.key} prompt={prompt['id']} seed={seed} "
          f"temperature={m.temperature} repeats={args.repeats}")
    try:
        outputs = asyncio.run(_repro_async(m, api_key, prompt["text"], seed, args.repeats))
    except CallError as exc:
        sys.exit(f"Call failed: {exc}")
    for i, out in enumerate(outputs):
        print(f"\n--- call {i + 1} ---\n{out}")
    if len(set(outputs)) == 1:
        print("\nRESULT: IDENTICAL. The seed is honoured on this setup.")
        return 0
    print("\nRESULT: DIFFERENT. The server may ignore `seed`, or generation is not deterministic "
          "(threading, GPU kernels). Do not claim exact reproducibility; record this in the report.")
    return 2


def refusal_hint(text: Optional[str]) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(h in lowered for h in REFUSAL_HINTS)


def cmd_summary(args: argparse.Namespace) -> int:
    db = open_db(resolve_db(args))
    rows = db.execute(
        "SELECT model_key, model_id, status, latency_ms, response_text, error, attempts "
        "FROM generations WHERE run_id=? ORDER BY model_key", (args.run_id,)).fetchall()
    if not rows:
        sys.exit(f"No data for run '{args.run_id}' in {resolve_db(args)}")

    by_model: dict = {}
    for model_key, model_id, status, latency, text, error, attempts in rows:
        d = by_model.setdefault(model_key, {"model_id": model_id, "ok": 0, "err": 0, "lat": [],
                                            "chars": [], "hint": 0, "retried": 0, "errors": []})
        if status == "ok":
            d["ok"] += 1
            d["lat"].append(latency or 0)
            d["chars"].append(len(text or ""))
            d["hint"] += 1 if refusal_hint(text) else 0
        else:
            d["err"] += 1
            if error and error not in d["errors"]:
                d["errors"].append(error)
        if attempts and attempts > 1:
            d["retried"] += 1

    print(f"Run '{args.run_id}'  (small n: read these as smoke-test numbers, not statistics)\n")
    header = f"{'model':<18} {'ok':>4} {'err':>4} {'retried':>7} {'lat p50':>9} {'lat max':>9} " \
             f"{'avg chars':>9} {'refusal?':>8}"
    print(header)
    print("-" * len(header))
    for key, d in by_model.items():
        p50 = int(statistics.median(d["lat"])) if d["lat"] else 0
        pmax = max(d["lat"]) if d["lat"] else 0
        avg = int(statistics.mean(d["chars"])) if d["chars"] else 0
        print(f"{key:<18} {d['ok']:>4} {d['err']:>4} {d['retried']:>7} {p50:>7}ms {pmax:>7}ms "
              f"{avg:>9} {d['hint']:>8}")
    print("\n'refusal?' is a crude keyword hint only. Read the responses (show) and label them "
          "by hand (export-csv).")
    for key, d in by_model.items():
        for e in d["errors"][:3]:
            print(f"  error [{key}]: {e[:200]}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    db = open_db(resolve_db(args))
    query = ("SELECT prompt_id, model_key, sample_idx, status, response_text, error, prompt_json, seed "
             "FROM generations WHERE run_id=?")
    params: list = [args.run_id]
    if args.model:
        query += " AND model_key=?"
        params.append(args.model)
    if args.prompt:
        query += " AND prompt_id=?"
        params.append(args.prompt)
    query += " ORDER BY prompt_id, model_key, sample_idx"
    rows = db.execute(query, params).fetchall()
    if not rows:
        sys.exit("No matching rows.")
    last_prompt = None
    for prompt_id, model_key, sample_idx, status, text, error, prompt_json, seed in rows:
        if prompt_id != last_prompt:
            print("\n" + "=" * 90)
            print(f"PROMPT {prompt_id}: {json.loads(prompt_json)['text']}")
            last_prompt = prompt_id
        seed_txt = f" seed={seed}" if seed is not None else ""
        print(f"\n--- {model_key} #{sample_idx}{seed_txt} [{status}] ---")
        print(text if status == "ok" else f"ERROR: {error}")
    return 0


def cmd_export_csv(args: argparse.Namespace) -> int:
    db = open_db(resolve_db(args))
    rows = db.execute(
        "SELECT run_id, prompt_id, model_key, model_id, sample_idx, status, latency_ms, "
        "response_text, prompt_json, seed FROM generations WHERE run_id=? "
        "ORDER BY prompt_id, model_key, sample_idx", (args.run_id,)).fetchall()
    if not rows:
        sys.exit(f"No data for run '{args.run_id}'")
    # utf-8-sig adds a BOM so Excel / Numbers open Thai text correctly
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "prompt_id", "model_key", "model_id", "sample_idx", "seed", "status",
                    "latency_ms", "prompt_text", "response_text", "refusal_hint",
                    "human_refusal_label(y/n)", "human_notes"])
        for run_id, prompt_id, model_key, model_id, idx, status, lat, text, pj, seed in rows:
            w.writerow([run_id, prompt_id, model_key, model_id, idx, seed, status, lat,
                        json.loads(pj)["text"], text or "",
                        "y" if refusal_hint(text) else "n", "", ""])
    print(f"Wrote {len(rows)} rows to {args.out}")
    return 0


# ----------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase-0 spike for the Thai LLM bias POC (local models)")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--run-id", required=True, help="stable id; reuse it to resume a run")
        sp.add_argument("--db", default=None, help=f"SQLite file (default {DEFAULT_DB}, "
                                                    f"or {DRY_RUN_DB} with --dry-run)")
        sp.add_argument("--dry-run", action="store_true",
                        help="use mock models and the separate dry-run database")

    r = sub.add_parser("run", help="call the models (resumable)")
    common(r)
    r.add_argument("--config", default=DEFAULT_CONFIG)
    r.add_argument("--prompts", default=DEFAULT_PROMPTS)
    r.add_argument("--samples", type=int, default=1, help="samples per prompt per model")
    r.add_argument("--model", action="append", help="only these model keys (repeatable)")
    r.add_argument("--max-calls", type=int, default=200, help="safety cap on planned calls")
    r.add_argument("--yes", action="store_true", help="skip the cost confirmation prompt")
    r.add_argument("--force", action="store_true",
                   help="resume even if models/prompts changed since the run started")
    r.add_argument("--mock-fail-rate", type=float, default=0.0,
                   help="dry-run only: probability that a mock call fails")
    r.set_defaults(func=cmd_run)

    rc = sub.add_parser("repro-check", help="same prompt + same seed twice: is output identical?")
    rc.add_argument("--model", required=True, help="model key from the config")
    rc.add_argument("--config", default=DEFAULT_CONFIG)
    rc.add_argument("--prompts", default=DEFAULT_PROMPTS)
    rc.add_argument("--prompt", help="prompt id (default: the first prompt)")
    rc.add_argument("--repeats", type=int, default=2)
    rc.set_defaults(func=cmd_repro_check)

    s = sub.add_parser("summary", help="latency / error / length per model")
    common(s)
    s.set_defaults(func=cmd_summary)

    sh = sub.add_parser("show", help="print prompts and responses for eyeballing")
    common(sh)
    sh.add_argument("--model")
    sh.add_argument("--prompt")
    sh.set_defaults(func=cmd_show)

    e = sub.add_parser("export-csv", help="export responses for manual labelling")
    common(e)
    e.add_argument("--out", required=True)
    e.set_defaults(func=cmd_export_csv)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
