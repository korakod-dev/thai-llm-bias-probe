// 3D overview for the Bias POC web UI (three.js r186, vendored in static/vendor/three-0.186.1).
//
// One object per step of the page. Clicking an object (or its label) calls the page's go(step);
// the camera then flies to that object and the page shows the step's normal 2D panel next to it.
// The overview is reached only through the page's own "ภาพรวม" button, never by clicking empty
// space, so a stray click cannot take the user out of a step they are working in.
// The page and this module only talk through window events and two globals:
//   bias:step  {step}                       the page switched step ("hub" = overview)
//   bias:done  {"1": bool, "2": bool, ...}  which steps are complete
//   bias:job   {running, k, n, exit, isRun} spike.py job progress
//   window.biasState() -> {step, done, job} (read once at start-up)
//   window.go(step), window.disable3D(err)  (defined by the page)
import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const V3 = THREE.Vector3;
const UP = new V3(0, 1, 0);
const OK_COLOR = 0x6cc28f;
const BG = 0x04060c;

// Label number and name are read from the page's step buttons (#steps), so they never drift.
const STEPS = [
  {step: "1", color: 0x5aa9ff, build: buildModels},
  {step: "2", color: 0xa78bfa, build: buildQuestions},
  {step: "3", color: 0xffb454, build: buildRun},
  {step: "4", color: 0x2dd4bf, build: buildReview},
  {step: "5", color: 0x4ade80, build: buildSummary},
  {step: "adv", color: 0x94a3b8, build: buildAdvanced},
];
function navLabel(step) {
  const b = document.querySelector(`#steps button[data-step="${step}"]`);
  const n = b && b.querySelector(".n");
  return {n: n ? n.textContent.trim() : step,
          name: b ? b.textContent.replace(n ? n.textContent : "", "").trim() : step};
}

const sfx = (name, ...a) => { try { window.SFX && window.SFX[name] && window.SFX[name](...a); } catch (e) {} };
const wide = () => innerWidth >= 1000;
const portrait = () => innerWidth / innerHeight < 0.8;
// Where object i stands: a shallow arc on wide screens, a 2 x 3 grid on tall (phone) screens.
function slot(i, tall) {
  if (tall) return new V3(i % 2 ? 1.6 : -1.6, 1.35, -3.5 + Math.floor(i / 2) * 3.5);
  const x = -5.6 + i * 2.24;
  return new V3(x, 1.35, 0.08 * x * x - 1.2);
}
const ease = (u) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);
const approach = (cur, target, dt, speed = 7) => cur + (target - cur) * Math.min(1, dt * speed);

/* ----------------------------- materials ----------------------------- */
function solid(color, extra = {}) {
  return new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(color).multiplyScalar(0.3), metalness: 0.6, roughness: 0.22,
    clearcoat: 1, clearcoatRoughness: 0.12,
    emissive: new THREE.Color(color), emissiveIntensity: 0.22, ...extra,
  });
}
function glass(color) {
  return new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(color).lerp(new THREE.Color(0xffffff), 0.55), metalness: 0, roughness: 0.06,
    transmission: 1, thickness: 0.9, ior: 1.45, iridescence: 0.6, iridescenceIOR: 1.3,
    emissive: new THREE.Color(color), emissiveIntensity: 0.08,
  });
}
function glow(color, strength = 2.2) {  // bright unlit colour that the bloom pass picks up
  return new THREE.MeshBasicMaterial({color: new THREE.Color(color).multiplyScalar(strength), toneMapped: false});
}
function edges(geo, color, strength = 2.0) {
  const m = new THREE.LineBasicMaterial({color: new THREE.Color(color).multiplyScalar(strength),
    transparent: true, opacity: 0.85, toneMapped: false});
  return new THREE.LineSegments(new THREE.EdgesGeometry(geo, 1), m);
}

/* ----------------------------- the six objects -----------------------------
   Each builder fills `g` (a Group) and returns tick(t, dt, st), where st = {hover, active,
   job, done} with hover/active eased 0..1. Emissive materials registered in `glows` get their
   intensity scaled by the common code, so hover/active brighten every object the same way. */

// 1 เลือกโมเดล: a glass crystal with a wire shell and three orbiting "models".
function buildModels(g, color, glows) {
  const core = new THREE.Mesh(new THREE.IcosahedronGeometry(0.62, 0), glass(color));
  const heart = new THREE.Mesh(new THREE.IcosahedronGeometry(0.26, 1), glow(color, 1.6));
  const shellGeo = new THREE.IcosahedronGeometry(0.98, 1);
  const shell = edges(shellGeo, color, 1.4); shell.material.opacity = 0.5;
  g.add(core, heart, shell);
  const orbiters = [0, 1, 2].map((i) => {
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.075, 20, 12), glow(color, 2.6));
    g.add(m); return m;
  });
  glows.push(heart.material);
  return (t, dt, st) => {
    core.rotation.y += dt * (0.4 + st.hover); core.rotation.x += dt * 0.15;
    shell.rotation.y -= dt * 0.25; shell.rotation.z += dt * 0.1;
    heart.scale.setScalar(1 + 0.12 * Math.sin(t * 2.4));
    orbiters.forEach((m, i) => {
      const a = t * (0.9 + i * 0.25) + (i * Math.PI * 2) / 3, r = 1.15;
      m.position.set(Math.cos(a) * r, Math.sin(a * 1.3 + i) * 0.35, Math.sin(a) * r);
    });
  };
}

// 2 ชุดคำถาม: a fanned stack of cards with glowing "text lines" on top.
function buildQuestions(g, color, glows) {
  const cards = [];
  for (let i = 0; i < 5; i++) {
    const geo = new THREE.BoxGeometry(1.35, 0.06, 0.95);
    const card = new THREE.Mesh(geo, solid(color, {emissiveIntensity: 0.12 + i * 0.05}));
    card.add(edges(geo, color, 1.2 + i * 0.25));
    card.position.y = -0.45 + i * 0.2;
    g.add(card); cards.push(card); glows.push(card.material);
  }
  const top = cards[cards.length - 1];
  [0.55, 0.8, 0.4].forEach((w, i) => {
    const line = new THREE.Mesh(new THREE.BoxGeometry(w, 0.012, 0.06), glow(color, 2.4));
    line.position.set(-0.52 + w / 2, 0.04, -0.25 + i * 0.2);
    top.add(line);
  });
  return (t, dt, st) => {
    const fan = 0.12 + 0.18 * st.hover + 0.1 * st.active;
    cards.forEach((c, i) => {
      c.rotation.y = (i - 2) * fan + Math.sin(t * 0.7) * 0.08;
      c.position.y = -0.45 + i * (0.2 + 0.05 * st.hover);
    });
    g.rotation.x = 0.35;
  };
}

// 3 รัน: a pulsing core inside a ring that fills up with the job's progress.
function buildRun(g, color, glows) {
  const core = new THREE.Mesh(new THREE.SphereGeometry(0.4, 40, 24), solid(color, {emissiveIntensity: 0.5, metalness: 0.2}));
  const track = new THREE.Mesh(new THREE.TorusGeometry(0.95, 0.035, 12, 160),
    new THREE.MeshBasicMaterial({color: new THREE.Color(color).multiplyScalar(0.35), transparent: true, opacity: 0.6}));
  const arcMat = glow(color, 2.0);
  const arc = new THREE.Mesh(new THREE.BufferGeometry(), arcMat);
  arc.rotation.z = Math.PI / 2; arc.scale.x = -1;  // start at 12 o'clock, fill clockwise
  const outer = new THREE.Mesh(new THREE.TorusGeometry(1.22, 0.012, 8, 160), glow(color, 1.4));
  const outer2 = outer.clone(); outer2.rotation.x = Math.PI / 2.4;
  g.add(core, track, arc, outer, outer2);
  glows.push(core.material);
  let shown = -1;
  return (t, dt, st) => {
    const j = st.job || {};
    const running = !!(j.running && j.isRun);
    const finished = !j.running && j.isRun && j.exit === 0;
    const frac = running && j.n ? j.k / j.n : finished ? 1 : 0;
    if (Math.abs(frac - shown) > 0.002) {
      shown = frac;
      arc.geometry.dispose();
      arc.geometry = frac > 0 ? new THREE.TorusGeometry(0.95, 0.06, 12, 160, Math.max(0.01, frac * Math.PI * 2))
                              : new THREE.BufferGeometry();
    }
    arcMat.color.set(finished ? OK_COLOR : color).multiplyScalar(2.0);
    const speed = running ? 2.6 : 0.35;
    outer.rotation.y += dt * speed; outer2.rotation.z += dt * speed * 0.8;
    // A slow, shallow "breathing" while running. A fast pulse of the glow (the old sin(t * 6),
    // x1.6) made the bloom blow out to white and strobe across half the screen during a run.
    const pulse = running ? 0.5 + 0.5 * Math.sin(t * 2.2) : 0.5 + 0.5 * Math.sin(t * 1.5);
    core.scale.setScalar(1 + 0.05 * pulse);
    core.material.emissiveIntensity *= running ? 1.0 + 0.15 * pulse : 1;
  };
}

// 4 ตรวจโดยคน: a glass diamond held in a gyroscope of rings (a watching "eye").
function buildReview(g, color, glows) {
  const gem = new THREE.Mesh(new THREE.OctahedronGeometry(0.62, 0), glass(color));
  gem.add(edges(gem.geometry, color, 1.6));
  const heart = new THREE.Mesh(new THREE.OctahedronGeometry(0.2, 0), glow(color, 2.2));
  const r1 = new THREE.Mesh(new THREE.TorusGeometry(0.95, 0.025, 10, 120), solid(color, {emissiveIntensity: 0.6}));
  const r2 = new THREE.Mesh(new THREE.TorusGeometry(1.08, 0.02, 10, 120), solid(color, {emissiveIntensity: 0.4}));
  g.add(gem, heart, r1, r2);
  glows.push(r1.material, r2.material);
  return (t, dt, st) => {
    gem.rotation.y += dt * (0.5 + st.hover);
    heart.rotation.y -= dt * 2;
    r1.rotation.x = Math.PI / 2 + Math.sin(t * 0.6) * 0.5; r1.rotation.y += dt * 0.6;
    r2.rotation.y = Math.PI / 2 + Math.cos(t * 0.5) * 0.6; r2.rotation.x += dt * 0.4;
  };
}

// 5 สรุป & เทียบ: a small bar chart on a disc.
function buildSummary(g, color, glows) {
  const base = new THREE.Mesh(new THREE.CylinderGeometry(0.95, 1.0, 0.08, 64), solid(color, {emissiveIntensity: 0.08}));
  base.position.y = -0.6;
  g.add(base);
  const heights = [0.55, 1.05, 0.75, 1.35, 0.9];
  const bars = heights.map((h, i) => {
    const bar = new THREE.Mesh(new THREE.BoxGeometry(0.2, 1, 0.2), solid(color, {emissiveIntensity: 0.3}));
    const cap = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.03, 0.22), glow(color, 2.6));
    bar.position.set(-0.6 + i * 0.3, 0, 0); cap.position.x = bar.position.x;
    g.add(bar, cap); glows.push(bar.material);
    return {bar, cap, h, i};
  });
  return (t, dt, st) => {
    for (const b of bars) {
      const h = b.h * (0.85 + 0.15 * Math.sin(t * 1.6 + b.i * 1.3)) * (1 + 0.15 * st.hover);
      b.bar.scale.y = h; b.bar.position.y = -0.56 + h / 2; b.cap.position.y = -0.56 + h + 0.015;
    }
    g.rotation.y = Math.sin(t * 0.35) * 0.5;
  };
}

// ⚙ ขั้นสูง: two interlocking metal gears.
function gearShape(teeth, rOut, rIn, rHole) {
  const s = new THREE.Shape(), step = (Math.PI * 2) / teeth;
  for (let i = 0; i < teeth; i++) {
    const a = i * step;
    [[rIn, a], [rOut, a + step * 0.18], [rOut, a + step * 0.45], [rIn, a + step * 0.62]].forEach(([r, th], j) => {
      const x = r * Math.cos(th), y = r * Math.sin(th);
      if (i === 0 && j === 0) s.moveTo(x, y); else s.lineTo(x, y);
    });
  }
  s.closePath();
  const hole = new THREE.Path(); hole.absarc(0, 0, rHole, 0, Math.PI * 2, true);
  s.holes.push(hole);
  return s;
}
function buildAdvanced(g, color, glows) {
  const opts = {depth: 0.2, bevelEnabled: true, bevelThickness: 0.03, bevelSize: 0.025, bevelSegments: 3, curveSegments: 20};
  const mk = (teeth, rOut, rIn, rHole) => {
    const geo = new THREE.ExtrudeGeometry(gearShape(teeth, rOut, rIn, rHole), opts); geo.center();
    const m = new THREE.Mesh(geo, solid(color, {metalness: 0.9, roughness: 0.28, emissiveIntensity: 0.12}));
    m.add(edges(geo, color, 0.9)); glows.push(m.material);
    return m;
  };
  const big = mk(14, 0.72, 0.6, 0.2), small = mk(9, 0.46, 0.35, 0.12);
  big.position.set(-0.28, 0.12, 0); small.position.set(0.72, -0.38, 0.05);
  const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.34, 24), glow(color, 2.2));
  hub.rotation.x = Math.PI / 2; big.add(hub);
  g.add(big, small);
  return (t, dt, st) => {
    const s = dt * (0.35 + 1.2 * st.hover);
    big.rotation.z += s; small.rotation.z -= s * (14 / 9);
    g.rotation.y = Math.sin(t * 0.4) * 0.35;
  };
}

/* ----------------------------- scene ----------------------------- */
function start() {
  const canvas = document.getElementById("scene");
  const labelsBox = document.getElementById("labels");
  const renderer = new THREE.WebGLRenderer({canvas, antialias: true, powerPreference: "high-performance"});
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75));
  renderer.setSize(innerWidth, innerHeight, false);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(BG);
  scene.fog = new THREE.FogExp2(BG, 0.045);
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.55;

  const camera = new THREE.PerspectiveCamera(42, innerWidth / innerHeight, 0.1, 250);
  scene.add(new THREE.HemisphereLight(0x8fb4ff, 0x080a10, 0.6));
  const key = new THREE.DirectionalLight(0xffffff, 1.4); key.position.set(4, 8, 6); scene.add(key);

  // floor: dark, slightly glossy, with a fading grid
  const floor = new THREE.Mesh(new THREE.CircleGeometry(40, 96),
    new THREE.MeshStandardMaterial({color: 0x070a12, metalness: 0.7, roughness: 0.42}));
  floor.rotation.x = -Math.PI / 2; scene.add(floor);
  const grid = new THREE.GridHelper(80, 80, 0x1c2942, 0x111a2c);
  grid.material.transparent = true; grid.material.opacity = 0.45; grid.position.y = 0.002; scene.add(grid);

  // stars
  const starGeo = new THREE.BufferGeometry(), sp = [];
  for (let i = 0; i < 1400; i++) {
    const v = new V3().randomDirection().multiplyScalar(45 + Math.random() * 60);
    v.y = Math.abs(v.y) * 0.8 + 2; sp.push(v.x, v.y, v.z);
  }
  starGeo.setAttribute("position", new THREE.Float32BufferAttribute(sp, 3));
  const stars = new THREE.Points(starGeo, new THREE.PointsMaterial({color: 0xa9bcff, size: 0.12, transparent: true,
    opacity: 0.85, depthWrite: false, fog: false}));
  scene.add(stars);

  // the six objects on a shallow arc
  const beamVS = "varying float vY; void main(){ vY = uv.y; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }";
  const beamFS = "uniform vec3 uColor; uniform float uAlpha; varying float vY; void main(){ gl_FragColor = vec4(uColor, uAlpha * pow(1.0 - vY, 2.2) * 0.5); }";
  const objs = STEPS.map((def, i) => {
    const base = slot(i, portrait());
    const g = new THREE.Group(); g.position.copy(base);
    const glows = [];
    const tick = def.build(g, def.color, glows);
    const baseGlow = glows.map((m) => m.emissiveIntensity);
    // invisible hit sphere so the whole object is easy to click
    const hit = new THREE.Mesh(new THREE.SphereGeometry(1.25, 16, 12),
      new THREE.MeshBasicMaterial({transparent: true, opacity: 0, depthWrite: false}));
    hit.userData.idx = i; g.add(hit);
    scene.add(g);

    const ringMat = new THREE.MeshBasicMaterial({color: new THREE.Color(def.color).multiplyScalar(2), transparent: true,
      opacity: 0.15, toneMapped: false, depthWrite: false});
    const ring = new THREE.Mesh(new THREE.RingGeometry(1.0, 1.1, 96), ringMat);
    ring.rotation.x = -Math.PI / 2; ring.position.set(base.x, 0.01, base.z); scene.add(ring);
    const doneMat = new THREE.MeshBasicMaterial({color: new THREE.Color(OK_COLOR).multiplyScalar(1.8), transparent: true,
      opacity: 0, toneMapped: false, depthWrite: false});
    const doneRing = new THREE.Mesh(new THREE.RingGeometry(1.22, 1.27, 96), doneMat);
    doneRing.rotation.x = -Math.PI / 2; doneRing.position.set(base.x, 0.012, base.z); scene.add(doneRing);
    const beamMat = new THREE.ShaderMaterial({vertexShader: beamVS, fragmentShader: beamFS, transparent: true,
      depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
      uniforms: {uColor: {value: new THREE.Color(def.color)}, uAlpha: {value: 0}}});
    const beam = new THREE.Mesh(new THREE.CylinderGeometry(1.0, 1.05, 3.2, 64, 1, true), beamMat);
    beam.position.set(base.x, 1.6, base.z); scene.add(beam);

    const label = document.createElement("button");
    label.type = "button"; label.className = "l3";
    label.innerHTML = '<span class="n"></span><span class="t"></span><span class="s"></span>';
    const nav = navLabel(def.step);
    label.children[0].textContent = nav.n; label.children[1].textContent = nav.name;
    label.addEventListener("click", () => { sfx("select", i); window.go(def.step); });
    label.addEventListener("mouseenter", () => sfx("hover"));
    labelsBox.appendChild(label);

    return {def, i, g, base, tick, glows, baseGlow, hit, ring, doneRing, beam, ringMat, doneMat, beamMat, label, h: 0, a: 0, d: 0};
  });
  const hits = objs.map((o) => o.hit);

  // post-processing: bloom makes the emissive parts glow
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.75, 0.6, 0.62);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  /* ---------- state from the page ---------- */
  const init = (window.biasState && window.biasState()) || {};
  let view = init.step || "hub";
  let done = init.done || {};
  let job = init.job || {};
  addEventListener("bias:done", (e) => { done = e.detail || {}; });
  addEventListener("bias:job", (e) => { job = e.detail || {}; });
  addEventListener("bias:step", (e) => setView(String(e.detail.step)));

  /* ---------- camera ---------- */
  const HUB_TGT = new V3(0, 1.0, 0);
  let tall = portrait();
  function pose(v) {
    const idx = objs.findIndex((o) => o.def.step === v);
    if (idx < 0) {
      if (tall) {  // look down on the grid from above
        const k = Math.max(1, 0.5 / camera.aspect);
        return {pos: new V3(0, 12 * k, 11.5 * k), tgt: new V3(0, 0.8, 0.2)};
      }
      const d = 12.5 * Math.max(1, 1.9 / camera.aspect);  // keep all six in view on narrower windows
      return {pos: new V3(0, 3.4 + (d - 12.5) * 0.25, d), tgt: HUB_TGT.clone()};
    }
    const o = objs[idx].base;
    const dist = wide() ? 6.8 : 7.2;
    const dir = new V3(o.x * 0.3, 0, 10).normalize();
    const pos = o.clone().addScaledVector(dir, dist); pos.y = o.y + 1.0;
    const tgt = o.clone();
    const fwd = tgt.clone().sub(pos).normalize(), right = new V3().crossVectors(fwd, UP).normalize();
    const halfW = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * dist * camera.aspect;
    if (wide()) {  // put the object in the left half; the step's panel fills the right half
      pos.addScaledVector(right, halfW * 0.5); tgt.addScaledVector(right, halfW * 0.5);
    } else {       // narrow screens: object near the top, above the panel
      pos.y -= 1.3; tgt.y -= 1.3;
    }
    return {pos, tgt};
  }
  const cam = {pos: new V3(), tgt: new V3()};
  const fly = {fromPos: new V3(), fromTgt: new V3(), toPos: new V3(), toTgt: new V3(), t0: -10, dur: 1.5};
  const timer = new THREE.Timer();
  const now = () => timer.getElapsed();
  {
    const p = pose(view);
    // open with a short descent from above for a sense of arrival
    cam.pos.copy(p.pos).add(new V3(0, 6, 8)); cam.tgt.copy(p.tgt);
    fly.fromPos.copy(cam.pos); fly.fromTgt.copy(cam.tgt); fly.toPos.copy(p.pos); fly.toTgt.copy(p.tgt);
    fly.t0 = 0; fly.dur = 2.4;
  }
  function setView(v) {
    if (v === view) return;
    view = v;
    const p = pose(v);
    fly.fromPos.copy(cam.pos); fly.fromTgt.copy(cam.tgt); fly.toPos.copy(p.pos); fly.toTgt.copy(p.tgt);
    fly.t0 = now(); fly.dur = 1.5;
    sfx("whoosh");
  }

  /* ---------- pointer ---------- */
  const ndc = new THREE.Vector2(), par = new THREE.Vector2(), mouse = new THREE.Vector2();
  const ray = new THREE.Raycaster();
  let hover = -1, pointerOnCanvas = false;
  addEventListener("pointermove", (e) => {
    mouse.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
    pointerOnCanvas = e.target === canvas;
  });
  canvas.addEventListener("pointerleave", () => { pointerOnCanvas = false; });
  canvas.addEventListener("click", (e) => {
    ndc.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const hitObj = ray.intersectObjects(hits, false)[0];
    if (!hitObj) return;
    const o = objs[hitObj.object.userData.idx];
    sfx("select", o.i);
    window.go(o.def.step);
  });

  /* ---------- resize ---------- */
  function resize() {
    camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
    if (portrait() !== tall) {  // switch between arc and grid
      tall = portrait();
      for (const o of objs) {
        o.base.copy(slot(o.i, tall));
        o.g.position.copy(o.base);
        o.ring.position.set(o.base.x, 0.01, o.base.z);
        o.doneRing.position.set(o.base.x, 0.012, o.base.z);
        o.beam.position.set(o.base.x, 1.6, o.base.z);
      }
    }
    renderer.setSize(innerWidth, innerHeight, false);  // resizing clears the canvas...
    composer.setSize(innerWidth, innerHeight);
    lastFrame = 0;  // ...so draw on the very next frame, not after the 30 fps throttle
    const p = pose(view);
    fly.toPos.copy(p.pos); fly.toTgt.copy(p.tgt);
    if (now() - fly.t0 >= fly.dur) { cam.pos.copy(p.pos); cam.tgt.copy(p.tgt); }
  }
  addEventListener("resize", resize);

  /* ---------- labels ---------- */
  const tmp = new V3();
  function placeLabels() {
    const w = innerWidth, h = innerHeight;
    for (const o of objs) {
      tmp.set(o.base.x, 0.02, o.base.z + 1.15).project(camera);
      const vis = tmp.z < 1 && Math.abs(tmp.x) < 1.2 && Math.abs(tmp.y) < 1.2;
      o.label.style.visibility = vis ? "visible" : "hidden";
      if (!vis) continue;
      o.label.style.transform = `translate(${((tmp.x + 1) / 2) * w}px, ${((1 - tmp.y) / 2) * h}px) translate(-50%, -50%)`;
    }
  }
  function labelStatus(o) {
    const s = o.def.step, j = job || {};
    let txt = "", cls = "";
    if (s === "3" && j.running && j.isRun) { txt = j.n ? `กำลังรัน ${j.k}/${j.n}` : "กำลังเริ่ม…"; cls = "run"; }
    else if (done[s]) { txt = "✓ เสร็จ"; cls = "done"; }
    o.label.classList.toggle("on", view === s);
    o.label.classList.toggle("done", cls === "done");
    o.label.classList.toggle("run", cls === "run");
    const el = o.label.lastChild;
    if (el.textContent !== txt) el.textContent = txt;
  }

  /* ---------- frame loop ---------- */
  let hubFade = 0;
  let lastFrame = 0;
  renderer.setAnimationLoop((ms) => {
    // With a step open the scene is only a backdrop, and the page is often left open for a whole
    // run: 30 fps there halves the GPU work. Full rate on the overview and during camera flights.
    const busy = view === "hub" || now() < fly.t0 + fly.dur;
    if (!busy && ms - lastFrame < 1000 / 30 - 2) return;
    lastFrame = ms;
    timer.update(ms);
    const dt = Math.min(timer.getDelta(), 0.05), t = now();

    // hover (only when the pointer is over the canvas, not over a panel)
    let nh = -1;
    if (pointerOnCanvas) {
      ray.setFromCamera(mouse, camera);
      const hi = ray.intersectObjects(hits, false)[0];
      if (hi) nh = hi.object.userData.idx;
    }
    if (nh !== hover) { hover = nh; if (hover >= 0) sfx("hover"); }
    canvas.style.cursor = hover >= 0 ? "pointer" : "default";

    // camera flight (with a gentle upward arc) + pointer parallax
    const u = Math.min(1, (t - fly.t0) / fly.dur), e = ease(u);
    cam.pos.lerpVectors(fly.fromPos, fly.toPos, e); cam.pos.y += Math.sin(Math.PI * e) * 0.9 * (u < 1 ? 1 : 0);
    cam.tgt.lerpVectors(fly.fromTgt, fly.toTgt, e);
    par.x = approach(par.x, mouse.x, dt, 2.5); par.y = approach(par.y, mouse.y, dt, 2.5);
    hubFade = approach(hubFade, view === "hub" ? 1 : 0, dt, 3);
    const amp = 0.25 + 0.45 * hubFade;
    camera.position.set(cam.pos.x + par.x * amp, cam.pos.y + par.y * amp * 0.5 + Math.sin(t * 0.3) * 0.05 * hubFade, cam.pos.z);
    camera.lookAt(cam.tgt);
    scene.fog.density = 0.045 * Math.min(1, 14 / camera.position.distanceTo(cam.tgt));

    for (const o of objs) {
      const isActive = view === o.def.step;
      o.h = approach(o.h, hover === o.i ? 1 : 0, dt, 9);
      o.a = approach(o.a, isActive ? 1 : 0, dt, 4);
      o.d = approach(o.d, done[o.def.step] ? 1 : 0, dt, 3);
      o.g.position.y = o.base.y + Math.sin(t * 1.1 + o.i * 1.7) * 0.09 + 0.12 * o.a;
      o.g.scale.setScalar(1 + 0.1 * o.h + 0.1 * o.a);
      const k = 1 + 1.6 * o.h + 1.3 * o.a;
      o.glows.forEach((m, n) => { m.emissiveIntensity = o.baseGlow[n] * k; });
      o.tick(t, dt, {hover: o.h, active: o.a, job, done: !!done[o.def.step]});
      o.ringMat.opacity = 0.12 + 0.45 * o.h + 0.75 * o.a;
      o.doneMat.opacity = 0.75 * o.d * (0.75 + 0.25 * Math.sin(t * 2 + o.i));
      o.beamMat.uniforms.uAlpha.value = 0.45 * o.a + 0.2 * o.h;
      labelStatus(o);
    }
    stars.rotation.y += dt * 0.004;

    composer.render(dt);
    placeLabels();
  });
  document.documentElement.classList.add("three-ready");
}

try {
  start();
} catch (err) {
  console.error("3D scene failed, falling back to 2D:", err);
  window.disable3D && window.disable3D(err);
}
