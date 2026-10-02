// UI sound effects for the Thai LLM Bias Probe web UI, synthesized with Web Audio (no audio files).
// window.SFX.<name>() plays a sound; the mute switch is remembered per browser.
// Sounds are dropped, never queued, until audio is actually running: a context that is
// suspended (no click/key yet, or paused by the system) has its clock frozen, so anything
// scheduled on it would all play at once on the next click.
(() => {
  const KEY = "bias_muted";
  let ctx = null, master = null, noiseBuf = null, lastHover = 0;
  let muted = false;
  try { muted = localStorage.getItem(KEY) === "1"; } catch (e) {}

  function ac() {
    const ua = navigator.userActivation;
    if (ua && !ua.hasBeenActive) return null;  // no click/key yet: the browser would not start audio
    if (!ctx) {
      const C = window.AudioContext || window.webkitAudioContext;
      if (!C) return null;
      ctx = new C();
      master = ctx.createGain();
      master.gain.value = 0.32;
      master.connect(ctx.destination);
    }
    if (ctx.state !== "running") { ctx.resume().catch(() => {}); return null; }
    return ctx;
  }

  // One enveloped oscillator note. f2 = glide target (Hz), t = start delay (s).
  function tone({f = 660, f2 = null, t = 0, dur = 0.12, type = "sine", vol = 0.5, attack = 0.004}) {
    const c = ac(); if (!c || muted) return;
    const now = c.currentTime + t;
    const o = c.createOscillator(), g = c.createGain();
    o.type = type;
    o.frequency.setValueAtTime(f, now);
    if (f2) o.frequency.exponentialRampToValueAtTime(f2, now + dur);
    g.gain.setValueAtTime(0.0001, now);
    g.gain.exponentialRampToValueAtTime(vol, now + attack);
    g.gain.exponentialRampToValueAtTime(0.0001, now + dur);
    o.connect(g).connect(master);
    o.start(now); o.stop(now + dur + 0.05);
  }

  // Band-passed noise whose centre frequency sweeps from `from` to `to` (a "whoosh").
  function sweep({t = 0, dur = 0.8, from = 300, to = 2400, vol = 0.2, q = 0.9}) {
    const c = ac(); if (!c || muted) return;
    if (!noiseBuf) {
      noiseBuf = c.createBuffer(1, c.sampleRate, c.sampleRate);
      const d = noiseBuf.getChannelData(0);
      for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    }
    const now = c.currentTime + t;
    const src = c.createBufferSource(), bp = c.createBiquadFilter(), g = c.createGain();
    src.buffer = noiseBuf; src.loop = true;
    bp.type = "bandpass"; bp.Q.value = q;
    bp.frequency.setValueAtTime(from, now);
    bp.frequency.exponentialRampToValueAtTime(to, now + dur * 0.6);
    bp.frequency.exponentialRampToValueAtTime(from * 1.5, now + dur);
    g.gain.setValueAtTime(0.0001, now);
    g.gain.exponentialRampToValueAtTime(vol, now + dur * 0.45);
    g.gain.exponentialRampToValueAtTime(0.0001, now + dur);
    src.connect(bp).connect(g).connect(master);
    src.start(now); src.stop(now + dur + 0.05);
  }

  // Pentatonic notes, one per step object, so each object has its own voice.
  const NOTES = [523.25, 587.33, 659.25, 783.99, 880.0, 440.0];

  window.SFX = {
    click() { tone({f: 1250, f2: 720, dur: 0.07, type: "triangle", vol: 0.22}); },
    hover() {
      const now = performance.now();
      if (now - lastHover < 90) return;
      lastHover = now;
      tone({f: 2100, dur: 0.035, vol: 0.045});
    },
    select(i = 0) {
      const f = NOTES[i % NOTES.length];
      tone({f, dur: 0.5, vol: 0.28});
      tone({f: f * 2, t: 0.03, dur: 0.4, vol: 0.1});
      tone({f: f * 1.5, t: 0.07, dur: 0.55, vol: 0.07, type: "triangle"});
    },
    whoosh() { sweep({dur: 1.1, from: 260, to: 2200, vol: 0.16}); },
    ok() { tone({f: 880, dur: 0.12, vol: 0.22}); tone({f: 1318.5, t: 0.08, dur: 0.22, vol: 0.2}); },
    bad() { tone({f: 311, f2: 233, dur: 0.22, type: "triangle", vol: 0.28}); },
    success() {
      [523.25, 659.25, 783.99, 1046.5].forEach((f, i) => tone({f, t: i * 0.09, dur: 0.5, vol: 0.2}));
      tone({f: 2093, t: 0.36, dur: 0.7, vol: 0.05});
    },
    error() { tone({f: 262, dur: 0.2, type: "triangle", vol: 0.28}); tone({f: 196, t: 0.18, dur: 0.35, type: "triangle", vol: 0.28}); },
    get muted() { return muted; },
    setMuted(m) {
      muted = !!m;
      try { localStorage.setItem(KEY, muted ? "1" : "0"); } catch (e) {}
    },
  };
})();
