#!/usr/bin/env python3
"""
webui.py - a small local web UI for spike.py (bias-poc v3).

What it does:
  * Shows the models in config/models.yaml, whether their API key env var is set
    (never the key itself), the prompt files, and the runs already in the databases.
  * Builds a spike.py command from a form, runs it as a subprocess, and streams its
    output to the page. One job at a time; a Stop button terminates it.
  * Lets you download label CSVs produced by `export-csv`.

It does NOT change how anything is measured: every action is an ordinary spike.py call,
so the CLI and the UI always produce identical evidence.

Safety:
  * Binds to 127.0.0.1 only. Requests with any other Host header are refused (blocks
    DNS-rebinding), and every POST needs a per-start random token that only the page
    served by this process knows (blocks other websites from triggering paid runs).
  * Arguments are validated against allow-lists and passed as an argv list (no shell).
  * The API key is read by spike.py from the environment you started this server in.
    Set it before starting:  export OPENROUTER_API_KEY=...

Usage:
  python3 webui.py              # then open http://127.0.0.1:8765
  python3 webui.py --port 9000

Standard library only (plus pyyaml, which spike.py already needs).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sqlite3
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import yaml

# ----------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent          # project directory; all paths are relative to it
SPIKE = ROOT / "spike.py"
CONFIG = ROOT / "config" / "models.yaml"
DBS = {"real": ROOT / "bias_poc.db", "dry": ROOT / "bias_poc_dry.db"}
TOKEN = secrets.token_urlsafe(24)               # new on every start; embedded in the page only
RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
PROMPT_ID_RE = re.compile(r"^[A-Za-z0-9._#-]{1,120}$")
LABEL_CSV_RE = re.compile(r"^(v3|spike)-[A-Za-z0-9._-]{1,80}\.csv$")  # same patterns as .gitignore
MAX_OUTPUT_CHARS = 2_000_000                    # keep the in-memory log bounded


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


# ----------------------------------------------------------------------------------------
# Project state (read-only)
# ----------------------------------------------------------------------------------------
def load_models() -> list[dict]:
    """Model entries from the config, reduced to what the page needs. Keys are never read."""
    try:
        data = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return []
    out = []
    for m in data.get("models", []):
        env_name = m.get("api_key_env")
        out.append({
            "key": m.get("key"),
            "model": m.get("model"),
            "note": m.get("note"),
            "api_key_env": env_name,
            "key_set": bool(env_name and os.environ.get(env_name)),
        })
    return out


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
    return {"models": load_models(), "prompt_files": prompt_files(),
            "label_csvs": label_csvs(), "runs": list_runs(), "job": JOB.snapshot(0)}


# ----------------------------------------------------------------------------------------
# Form -> spike.py argv (validated)
# ----------------------------------------------------------------------------------------
class BadRequest(Exception):
    pass


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
        samples = _int(req, "samples", 1, 20, 1)
        max_calls = _int(req, "max_calls", 1, 100_000, 200)
        # spike.py asks on stdin before paid calls; the page asks instead and sends
        # confirm_cost=true, so --yes is only added after that confirmation.
        if not dry and req.get("confirm_cost") is not True:
            raise BadRequest("ต้องยืนยันค่าใช้จ่ายก่อนรันจริง")
        argv = ["run", *common(rid), "--prompts", prompts,
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
        argv = ["repro-check", "--model", _model_keys({"models": [model]})[0],
                "--prompts", _prompts(req), "--repeats", str(_int(req, "repeats", 2, 5, 2))]
        prompt = str(req.get("prompt") or "").strip()
        if prompt:
            if not PROMPT_ID_RE.match(prompt):
                raise BadRequest("prompt id ไม่ถูกรูปแบบ")
            argv += ["--prompt", prompt]
        return argv

    raise BadRequest(f"ไม่รู้จักคำสั่ง: {cmd}")


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
        if url.path == "/":
            html = PAGE.replace("__TOKEN__", TOKEN).encode("utf-8")
            return self._send(200, html, "text/html; charset=utf-8")
        if url.path == "/api/state":
            return self._json(state())
        if url.path == "/api/job":
            since = int((parse_qs(url.query).get("since") or ["0"])[0] or 0)
            return self._json(JOB.snapshot(max(0, since)))
        if url.path == "/download":
            # GET is fine without the token: it only reads a git-ignored label CSV, and other
            # origins cannot read the response. The name must match the allow-list exactly.
            name = (parse_qs(url.query).get("file") or [""])[0]
            if name not in label_csvs():
                return self._send(404, b"not found", "text/plain")
            data = (ROOT / name).read_bytes()
            return self._send(200, data, "text/csv; charset=utf-8",
                              {"Content-Disposition": f'attachment; filename="{name}"'})
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if not self._host_ok():
            return self._send(HTTPStatus.FORBIDDEN, b"bad host", "text/plain")
        if not secrets.compare_digest(self.headers.get("X-Token", ""), TOKEN):
            return self._json({"error": "token ไม่ถูกต้อง (รีเฟรชหน้าเว็บ)"}, 403)
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 100_000)
            req = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._json({"error": "JSON ไม่ถูกต้อง"}, 400)
        path = urlparse(self.path).path
        if path == "/api/job":
            try:
                argv = build_argv(req)
                JOB.start(argv)
            except BadRequest as exc:
                return self._json({"error": str(exc)}, 400)
            except RuntimeError as exc:
                return self._json({"error": str(exc)}, 409)
            return self._json({"ok": True, "argv": argv})
        if path == "/api/job/stop":
            return self._json({"stopped": JOB.stop()})
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
  --code-bg: #1e1f22; --code-ink: #e6e6e3;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #161615; --panel: #1f1f1d; --ink: #ecebe7; --muted: #a09e98; --line: #34332f;
    --accent: #7fb0e0; --accent-ink: #10202f; --ok: #6cc28f; --warn: #e0a24a; --bad: #f08a80;
    --code-bg: #0f0f0e; --code-ink: #dcdcd8;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Sukhumvit Set", "Thonburi", "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 20px 16px 60px; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 0 0 12px; }
.sub { color: var(--muted); font-size: 13px; margin: 0 0 18px; }
.grid { display: grid; gap: 16px; grid-template-columns: 1fr; }
@media (min-width: 900px) { .grid { grid-template-columns: 380px 1fr; } }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px; }
.stack > * + * { margin-top: 16px; }
label { display: block; font-size: 13px; color: var(--muted); margin: 10px 0 4px; }
input[type=text], input[type=number], select {
  width: 100%; padding: 7px 9px; border: 1px solid var(--line); border-radius: 7px;
  background: var(--bg); color: var(--ink); font: inherit; }
.check { display: flex; align-items: center; gap: 8px; color: var(--ink); margin: 6px 0; font-size: 14px; }
.tabs { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
.tabs button { background: transparent; color: var(--ink); border: 1px solid var(--line); }
.tabs button.on { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
button { padding: 7px 14px; border: 0; border-radius: 7px; font: inherit; cursor: pointer;
  background: var(--accent); color: var(--accent-ink); }
button.ghost { background: transparent; color: var(--ink); border: 1px solid var(--line); }
button.danger { background: var(--bad); color: #fff; }
button:disabled { opacity: .45; cursor: default; }
.row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-top: 14px; }
.pill { display: inline-block; font-size: 12px; padding: 1px 8px; border-radius: 99px; border: 1px solid currentColor; }
.ok { color: var(--ok); } .warn { color: var(--warn); } .bad { color: var(--bad); } .muted { color: var(--muted); }
.hint { font-size: 12.5px; color: var(--muted); margin-top: 8px; }
table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
th, td { text-align: left; padding: 7px 6px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 500; }
td.actions { white-space: nowrap; }
td.actions button { padding: 3px 9px; font-size: 12.5px; }
.scroll { overflow-x: auto; }
pre#out { background: var(--code-bg); color: var(--code-ink); margin: 0; padding: 12px; border-radius: 8px;
  height: 420px; overflow: auto; font: 12.5px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace;
  white-space: pre-wrap; word-break: break-word; }
.cmdline { font: 12px ui-monospace, Menlo, monospace; color: var(--muted); word-break: break-all; }
.models div { margin: 4px 0; font-size: 13.5px; }
[hidden] { display: none !important; }
</style>
</head>
<body>
<main>
  <h1>Bias POC Runner</h1>
  <p class="sub">หน้านี้เรียก <code>spike.py</code> บนเครื่องนี้ ทุกปุ่มเท่ากับรันคำสั่งเดียวกับ CLI ผลที่ได้จึงเหมือนกันทุกประการ</p>

  <div class="grid">
    <div class="stack">
      <section class="card">
        <h2>โมเดล (config/models.yaml)</h2>
        <div id="models" class="models muted">กำลังโหลด…</div>
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
            <label for="labels">ไฟล์ label ที่ติดป้ายแล้ว <span class="muted">(ไม่บังคับ)</span></label>
            <select id="labels"></select>
          </div>
          <div data-for="run score summary show export-csv">
            <div class="check"><input type="checkbox" id="dry_run"><label for="dry_run" style="margin:0;color:inherit">Dry run (โมเดลจำลอง, ไม่เสียเงิน, ใช้ bias_poc_dry.db)</label></div>
          </div>
          <p class="hint" id="cmd_hint"></p>
          <div class="row">
            <button type="submit" id="go">เริ่ม</button>
          </div>
        </form>
      </section>
    </div>

    <div class="stack">
      <section class="card">
        <div class="row" style="margin-top:0;justify-content:space-between">
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
        <div class="row" style="margin-top:0;justify-content:space-between">
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
</main>

<script>
const TOKEN = "__TOKEN__";
let S = null;            // last /api/state
let cmd = "run";         // selected tab
let outLen = 0;          // chars of job output already shown
let polling = null;

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

async function api(path, body) {
  const opt = body === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json", "X-Token": TOKEN},
    body: JSON.stringify(body)};
  const r = await fetch(path, opt);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || ("HTTP " + r.status));
  return j;
}

function renderState() {
  // models + key status
  $("models").innerHTML = S.models.length ? S.models.map(m =>
    `<div><b>${esc(m.key)}</b> <span class="muted">${esc(m.model)}</span><br>
     ${m.key_set ? `<span class="pill ok">${esc(m.api_key_env)} ตั้งแล้ว</span>`
                 : `<span class="pill bad">${esc(m.api_key_env || "ไม่มี api_key_env")} ยังไม่ได้ตั้ง</span>`}</div>`
  ).join("") : `<span class="bad">อ่าน config ไม่ได้</span>`;

  const keep = (id) => $(id).value;
  const pv = keep("prompts"), mv = keep("model_one"), lv = keep("labels");
  $("prompts").innerHTML = S.prompt_files.map(f =>
    `<option value="${esc(f.name)}">${esc(f.name)} (${f.count} ข้อ)</option>`).join("");
  if (pv) $("prompts").value = pv;
  else if (S.prompt_files.some(f => f.name === "controls_only.jsonl")) $("prompts").value = "controls_only.jsonl";
  $("model_one").innerHTML = S.models.map(m => `<option>${esc(m.key)}</option>`).join("");
  if (mv) $("model_one").value = mv;
  if (cmd === "show") $("model_one").insertAdjacentHTML("afterbegin", `<option value="">(ทุกโมเดล)</option>`);
  $("labels").innerHTML = `<option value="">(ไม่ใช้ ใช้ hint อัตโนมัติ)</option>` +
    S.label_csvs.map(f => `<option>${esc(f)}</option>`).join("");
  if (lv) $("labels").value = lv;

  if (!$("model_checks").dataset.done) {
    $("model_checks").innerHTML = S.models.map(m =>
      `<div class="check"><input type="checkbox" name="mk" value="${esc(m.key)}" id="mk_${esc(m.key)}" checked>
       <label for="mk_${esc(m.key)}" style="margin:0;color:inherit">${esc(m.key)}</label></div>`).join("");
    $("model_checks").dataset.done = "1";
  }

  $("run_ids").innerHTML = S.runs.map(r => `<option value="${esc(r.run_id)}">`).join("");
  $("runs").innerHTML = S.runs.length ? S.runs.map(r => {
    const models = Object.entries(r.models).map(([k, st]) =>
      `${esc(k)}: <span class="ok">${st.ok || 0}</span> / <span class="${(st.error||0) ? "bad" : "muted"}">${st.error || 0}</span>`).join("<br>");
    const b = (c, t) => `<button class="ghost" data-quick="${c}" data-run="${esc(r.run_id)}" data-dry="${r.db === "dry"}">${t}</button>`;
    return `<tr><td><b>${esc(r.run_id)}</b><br><span class="muted">${esc((r.created_at||"").slice(0,16).replace("T"," "))}</span></td>
      <td>${r.db === "dry" ? '<span class="pill warn">dry</span>' : '<span class="pill ok">จริง</span>'}</td>
      <td>${models || '<span class="muted">-</span>'}</td>
      <td class="actions">${b("score","Score")} ${b("summary","Summary")} ${b("export-csv","CSV")}</td></tr>`;
  }).join("") : `<tr><td colspan="4" class="muted">ยังไม่มี run</td></tr>`;

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

function plannedCalls() {
  const f = S.prompt_files.find(p => p.name === $("prompts").value);
  const nModels = document.querySelectorAll('input[name=mk]:checked').length;
  return (f ? f.count : 0) * (parseInt($("samples").value) || 1) * nModels;
}

function updateHint() {
  let t = "";
  if (cmd === "run" && S) {
    const n = plannedCalls();
    t = $("dry_run").checked ? `สูงสุด ${n} calls (โมเดลจำลอง ไม่เสียเงิน)`
      : `สูงสุด ${n} API calls ที่เสียเงินจริง (ข้อที่ทำเสร็จแล้วใน run id เดิมจะข้าม) · ถ้า run id นี้มีอยู่แล้วแต่ config ต่างจากเดิม spike.py จะไม่ยอมรัน`;
  } else if (cmd === "repro-check") {
    t = "เรียก API จริงด้วย prompt และ seed เดิมซ้ำ เพื่อดูว่าได้คำตอบเหมือนเดิมไหม (เสียเงินเล็กน้อย)";
  } else if (cmd === "export-csv") {
    t = "ไฟล์จะถูกบันทึกเป็น v3-*.csv ในโฟลเดอร์โปรเจกต์ (อยู่ใน .gitignore) แล้วดาวน์โหลดได้จากด้านขวา";
  } else if (cmd === "score") {
    t = "ถ้ายังไม่ได้เลือกไฟล์ label ผลของข้อสอบควบคุมแบบถามตรงจะเป็นผลชั่วคราว (PROVISIONAL)";
  }
  $("cmd_hint").textContent = t;
}

function collect() {
  return {
    command: cmd,
    run_id: $("run_id").value.trim(),
    prompts: $("prompts").value,
    models: [...document.querySelectorAll('input[name=mk]:checked')].map(x => x.value),
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
      if (!confirm(`จะเรียก API จริงสูงสุด ${n} ครั้ง ซึ่งมีค่าใช้จ่าย\nrun id: ${req.run_id}\nยืนยันไหม?`)) return;
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

$("tabs").addEventListener("click", e => { if (e.target.dataset.cmd) setTab(e.target.dataset.cmd); });
$("form").addEventListener("submit", e => { e.preventDefault(); submit(collect()); });
$("form").addEventListener("input", updateHint);
$("stop").addEventListener("click", () => api("/api/job/stop", {}).catch(e => alert(e.message)));
$("refresh").addEventListener("click", refresh);
$("runs").addEventListener("click", e => {
  const b = e.target.closest("button[data-quick]"); if (!b) return;
  submit({command: b.dataset.quick, run_id: b.dataset.run, dry_run: b.dataset.dry === "true"});
});

setTab("run");
refresh().then(() => { if (S.job.running || S.job.argv.length) startPolling(); });
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Local web UI for spike.py")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        if exc.errno == 48 or "in use" in str(exc).lower():  # 48 = EADDRINUSE on macOS
            sys.exit(f"Port {args.port} is already in use (another webui.py still running?).\n"
                     f"Stop it, or pick another port: python3 webui.py --port {args.port + 1}")
        raise
    print(f"Bias POC web UI: http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
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
