/**
 * FieldScan TZ THT-02 — Three.js twin, spatial CV HUD, VAD voice, SCADA uPlot.
 */
import * as THREE from "three";
import { OrbitControls } from "/static/vendor/OrbitControls.js";
import uPlot from "/static/vendor/uPlot.esm.js";

const $ = (id) => document.getElementById(id);
const DIP_WEIGHTS = [128, 64, 32, 16, 8, 4, 2, 1];
const OUT_W = 1920;
const OUT_H = 1080;
const POLL_MS = 350;
const CV_HUD_MODES = ["Off", "Zones", "Dims", "Spatial"];
const ZONES = [
  { id: "strainer_head", label: "Strainer Head", z0: 0, z1: 32 },
  { id: "upper_seal_ring", label: "Upper Seal Ring", z0: 32, z1: 38 },
  { id: "upper_barrel_dip", label: "Upper Barrel DIP", z0: 38, z1: 68 },
  { id: "mounting_flange", label: "Mounting Flange", z0: 68, z1: 88 },
  { id: "lower_barrel_rs485", label: "Lower Barrel RS-485", z0: 88, z1: 122 },
  { id: "waterproof_gland_leads", label: "Gland & Leads", z0: 122, z1: 140 },
];

const state = {
  status: null,
  cvHudMode: 0,
  rawStream: true,
  handsFree: false,
  autoSpeak: true,
  manualPage: 1,
  manualTotal: 1,
  manualIndex: [],
  wireframe: false,
  exploded: false,
  selectedZone: null,
  dipBits: [1, 0, 0, 0, 0, 0, 0, 0],
  chartHistory: [],
  aiSpeakCooldownUntil: 0,
  lastStatusTs: 0,
  fpsEwma: 0,
  mediaRecorder: null,
  audioCtx: null,
  vadAnalyser: null,
  vadStream: null,
  pttActive: false,
};

/** object-fit: cover — map normalized frame coords to canvas pixels */
function vidPt(nx, ny, cw, ch, fw = OUT_W, fh = OUT_H) {
  const scale = Math.max(cw / fw, ch / fh);
  const dw = fw * scale;
  const dh = fh * scale;
  const ox = (cw - dw) / 2;
  const oy = (ch - dh) / 2;
  return { x: ox + nx * dw, y: oy + ny * dh, scale, ox, oy, dw, dh };
}

function bitsToAddr(bits) {
  return Math.max(1, bits.reduce((a, b, i) => a + (b ? DIP_WEIGHTS[i] : 0), 0));
}

async function apiJson(url, opts = {}) {
  const t0 = performance.now();
  const r = await fetch(url, {
    headers: opts.body && !(opts.body instanceof FormData) ? { "Content-Type": "application/json" } : undefined,
    ...opts,
  });
  const latency = Math.round(performance.now() - t0);
  $("pillLatencyText").textContent = `API: ${latency} ms`;
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json();
}

async function apiPost(url, body) {
  return apiJson(url, {
    method: "POST",
    body: body instanceof FormData ? body : JSON.stringify(body ?? {}),
  });
}

// ——— Spatial CV HUD (style 3 = Spatial) ———
function resizeHudCanvas() {
  const host = $("videoFrame");
  const canvas = $("cvSpatialHudCanvas");
  if (!host || !canvas) return;
  const rect = host.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(rect.width * dpr);
  canvas.height = Math.floor(rect.height * dpr);
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
}

function drawSpatialHud(vision) {
  resizeHudCanvas();
  const canvas = $("cvSpatialHudCanvas");
  const ctx = canvas.getContext("2d");
  const mode = state.cvHudMode;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (mode === 0 || !vision) return;

  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.width;
  const ch = canvas.height;
  ctx.save();
  ctx.scale(dpr, dpr);
  const cssW = cw / dpr;
  const cssH = ch / dpr;
  const fw = vision.frame_size?.[0] || OUT_W;
  const fh = vision.frame_size?.[1] || OUT_H;

  const stroke = (x0, y0, x1, y1, color, lw = 2) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
  };

  if (mode >= 1 && vision.zones_2d?.length) {
    for (const z of vision.zones_2d) {
      const p0 = vidPt(z.p0[0] / fw, z.p0[1] / fh, cssW, cssH, fw, fh);
      const p1 = vidPt(z.p1[0] / fw, z.p1[1] / fh, cssW, cssH, fw, fh);
      const hot = vision.anomaly_zone === z.id;
      stroke(p0.x, p0.y, p1.x, p1.y, hot ? "#ef4444" : "rgba(56,189,248,0.75)", hot ? 3 : 2);
      if (mode >= 2) {
        const c = vidPt(z.center[0] / fw, z.center[1] / fh, cssW, cssH, fw, fh);
        ctx.fillStyle = hot ? "#ef4444" : "#38bdf8";
        ctx.font = "10px system-ui";
        ctx.fillText(z.label.split(" ")[0], c.x + 4, c.y - 4);
      }
    }
  }

  if (mode >= 2 && vision.p_start && vision.p_end) {
    const a = vidPt(vision.p_start[0] / fw, vision.p_start[1] / fh, cssW, cssH, fw, fh);
    const b = vidPt(vision.p_end[0] / fw, vision.p_end[1] / fh, cssW, cssH, fw, fh);
    stroke(a.x, a.y, b.x, b.y, "#10b981", 3);
    const cx = (a.x + b.x) / 2;
    const cy = (a.y + b.y) / 2;
    ctx.strokeStyle = "#f59e0b";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.arc(cx, cy, 28, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    if (vision.estimated_length_mm) {
      ctx.fillStyle = "#e2e8f0";
      ctx.fillText(`${vision.estimated_length_mm.toFixed(1)} mm`, cx + 32, cy);
    }
  }

  if (mode >= 3) {
    if (vision.tool_tip && vision.p_start) {
      const t = vidPt(vision.tool_tip[0] / fw, vision.tool_tip[1] / fh, cssW, cssH, fw, fh);
      ctx.fillStyle = "rgba(245,158,11,0.85)";
      ctx.beginPath();
      ctx.arc(t.x, t.y, 6, 0, Math.PI * 2);
      ctx.fill();
    }
    if (vision.dip_corners?.length === 4) {
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 2;
      ctx.beginPath();
      vision.dip_corners.forEach((pt, i) => {
        const p = vidPt(pt[0] / fw, pt[1] / fh, cssW, cssH, fw, fh);
        if (i === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      });
      ctx.closePath();
      ctx.stroke();
    }
    ctx.fillStyle = "rgba(56,189,248,0.15)";
    ctx.fillRect(8, cssH - 28, 120, 20);
    ctx.fillStyle = "#38bdf8";
    ctx.font = "11px monospace";
    ctx.fillText("HUD · Spatial", 12, cssH - 14);
  }
  ctx.restore();
}

// ——— Three.js parametric twin ———
let renderer, scene, camera, controls, sensorRig, explodeParts = [];
let animId = 0;

function makeMaterial(wire) {
  return new THREE.MeshStandardMaterial({
    color: 0x6b7c8f,
    metalness: 0.55,
    roughness: 0.35,
    wireframe: wire,
  });
}

function buildTwin() {
  const canvas = $("cadCanvas");
  const stage = $("canvasStage");
  const w = stage.clientWidth;
  const h = stage.clientHeight;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x060a10);
  scene.fog = new THREE.Fog(0x060a10, 8, 28);

  camera = new THREE.PerspectiveCamera(42, w / h, 0.1, 100);
  camera.position.set(2.4, 1.6, 3.2);

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(w, h, false);
  renderer.shadowMap.enabled = true;

  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.target.set(0, 0.7, 0);

  const amb = new THREE.AmbientLight(0x334155, 0.9);
  scene.add(amb);
  const key = new THREE.DirectionalLight(0x38bdf8, 1.1);
  key.position.set(4, 6, 3);
  key.castShadow = true;
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x10b981, 0.35);
  rim.position.set(-3, 2, -2);
  scene.add(rim);

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(12, 12),
    new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.95 })
  );
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);

  sensorRig = new THREE.Group();
  sensorRig.rotation.y = Math.PI / 2;
  scene.add(sensorRig);

  const mat = makeMaterial(false);
  const matDip = new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.2, roughness: 0.8 });

  const parts = [];
  const mkCyl = (name, r, len, y, colorMat) => {
    const g = new THREE.Mesh(new THREE.CylinderGeometry(r, r, len, 32), colorMat || mat);
    g.name = name;
    g.position.y = y;
    g.castShadow = true;
    sensorRig.add(g);
    parts.push(g);
    return g;
  };

  mkCyl("strainer_head", 0.08, 0.32, 0.16);
  mkCyl("upper_seal_ring", 0.082, 0.06, 0.35);
  const upper = mkCyl("upper_barrel_dip", 0.08, 0.3, 0.53);
  mkCyl("mounting_flange", 0.11, 0.2, 0.78);
  mkCyl("lower_barrel_rs485", 0.08, 0.34, 1.05);
  mkCyl("waterproof_gland_leads", 0.06, 0.18, 1.31);

  const dipBlock = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.06, 0.08), matDip);
  dipBlock.name = "dip_pcb";
  dipBlock.position.set(0.12, 0.58, 0);
  upper.add(dipBlock);
  parts.push(dipBlock);

  const rockers = new THREE.Group();
  rockers.name = "dip_rockers";
  for (let i = 0; i < 8; i++) {
    const sw = new THREE.Mesh(
      new THREE.BoxGeometry(0.012, 0.018, 0.022),
      new THREE.MeshStandardMaterial({ color: 0x374151 })
    );
    sw.position.set(-0.045 + i * 0.013, 0.04, 0.02);
    sw.userData.pin = 8 - i;
    rockers.add(sw);
  }
  dipBlock.add(rockers);
  explodeParts = [dipBlock, rockers, ...parts.slice(0, 3)];

  buildAssemblyTree();
  syncDipRockers(state.dipBits);
  loopTwin();
}

function syncDipRockers(bits) {
  if (!sensorRig) return;
  const rockers = sensorRig.getObjectByName("dip_rockers");
  if (!rockers) return;
  rockers.children.forEach((sw, i) => {
    const on = bits[7 - i];
    sw.material.color.setHex(on ? 0x10b981 : 0x374151);
    sw.position.z = on ? 0.028 : 0.02;
  });
  const addr = bitsToAddr(bits);
  $("labelDipAddr").textContent = `DIP addr: ${addr}`;
  $("dipCalcAddr").textContent = `Addr: ${addr}`;
}

function applyExplode(on) {
  state.exploded = on;
  if (!sensorRig) return;
  const spread = on ? 0.22 : 0;
  explodeParts.forEach((p, i) => {
    p.userData.baseY = p.userData.baseY ?? p.position.y;
    p.position.y = p.userData.baseY + (on ? i * spread * 0.15 : 0);
    if (p.name === "dip_pcb") p.position.x = on ? 0.28 : 0.12;
  });
}

function setWireframe(on) {
  state.wireframe = on;
  sensorRig?.traverse((o) => {
    if (o.isMesh && o.material && "wireframe" in o.material) o.material.wireframe = on;
  });
  $("btnModeSolid").classList.toggle("active", !on);
  $("btnModeWireframe").classList.toggle("active", on);
}

function highlightZone(zoneId) {
  state.selectedZone = zoneId;
  if (!sensorRig) return;
  sensorRig.traverse((o) => {
    if (!o.isMesh || !o.material) return;
    const hot = o.name === zoneId;
    if ("emissive" in o.material) {
      o.material.emissive = new THREE.Color(hot ? 0x0ea5e9 : 0x000000);
      o.material.emissiveIntensity = hot ? 0.35 : 0;
    }
  });
  document.querySelectorAll(".tree-pill").forEach((el) => {
    el.classList.toggle("active", el.dataset.zone === zoneId);
  });
  const z = ZONES.find((x) => x.id === zoneId);
  $("selectedPartSummary").textContent = z
    ? `${z.label} · z ${z.z0}–${z.z1} mm`
    : "Select an assembly zone on the twin or tree.";
}

function buildAssemblyTree() {
  const wrap = $("assemblyTreePills");
  wrap.innerHTML = "";
  ZONES.forEach((z) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "tree-pill";
    b.dataset.zone = z.id;
    b.textContent = z.label.split(" ")[0];
    b.addEventListener("click", () => highlightZone(z.id));
    wrap.appendChild(b);
  });
}

function loopTwin() {
  cancelAnimationFrame(animId);
  const tick = () => {
    controls?.update();
    renderer?.render(scene, camera);
    animId = requestAnimationFrame(tick);
  };
  tick();
}

function onTwinResize() {
  const stage = $("canvasStage");
  if (!renderer || !camera) return;
  const w = stage.clientWidth;
  const h = stage.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
}

/** Capture four sign-off angles from twin */
function captureMultiAngle3DViews() {
  if (!renderer || !camera) return {};
  const saved = {
    pos: camera.position.clone(),
    target: controls.target.clone(),
  };
  const shots = {};
  const rigs = {
    isometric: { pos: [2.2, 1.8, 2.6], target: [0, 0.65, 0] },
    front_dim: { pos: [0.05, 0.75, 4.2], target: [0, 0.7, 0] },
    exploded_dip: { pos: [2.8, 1.2, 1.4], target: [0.2, 0.55, 0] },
    side_gland: { pos: [4.0, 0.9, 0.2], target: [0, 1.0, 0] },
  };
  const wasExploded = state.exploded;
  for (const [key, cfg] of Object.entries(rigs)) {
    applyExplode(key === "exploded_dip" || wasExploded);
    camera.position.set(...cfg.pos);
    controls.target.set(...cfg.target);
    controls.update();
    renderer.render(scene, camera);
    shots[key] = renderer.domElement.toDataURL("image/png");
  }
  applyExplode(wasExploded);
  camera.position.copy(saved.pos);
  controls.target.copy(saved.target);
  controls.update();
  return shots;
}

// ——— uPlot SCADA chart ———
let scadaPlot = null;

function initScadaChart() {
  const host = $("scadaChart");
  if (!host) return;
  const opts = {
    width: host.clientWidth || 400,
    height: 110,
    series: [
      {},
      { label: "Temp °C", stroke: "#38bdf8", width: 2 },
      { label: "RH %", stroke: "#10b981", width: 2 },
    ],
    axes: [
      { stroke: "#64748b", grid: { show: false } },
      { stroke: "#64748b", grid: { stroke: "#1e2a3a" } },
    ],
    scales: { x: { time: true } },
  };
  scadaPlot = new uPlot(opts, [[], [], []], host);
}

function pushScadaSample(modbus) {
  if (!modbus || modbus.temp_c == null) return;
  const t = Date.now() / 1000;
  state.chartHistory.push({
    t,
    temp: modbus.temp_c,
    hum: modbus.humidity_rh ?? 0,
  });
  if (state.chartHistory.length > 120) state.chartHistory.shift();
  const xs = state.chartHistory.map((s) => s.t);
  const ts = state.chartHistory.map((s) => s.temp);
  const hs = state.chartHistory.map((s) => s.hum);
  if (scadaPlot) {
    scadaPlot.setData([xs, ts, hs]);
  } else {
    drawScadaFallback(xs, ts);
  }
}

function drawScadaFallback(xs, ts) {
  const c = $("scadaFallbackCanvas");
  const ctx = c.getContext("2d");
  c.style.display = scadaPlot ? "none" : "block";
  ctx.fillStyle = "#10161f";
  ctx.fillRect(0, 0, c.width, c.height);
  if (ts.length < 2) return;
  const min = Math.min(...ts);
  const max = Math.max(...ts);
  const span = max - min || 1;
  ctx.strokeStyle = "#38bdf8";
  ctx.beginPath();
  ts.forEach((v, i) => {
    const x = (i / (ts.length - 1)) * (c.width - 8) + 4;
    const y = c.height - 4 - ((v - min) / span) * (c.height - 12);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

// ——— DIP rocker UI ———
function renderDipSwitches(bits) {
  const row = $("dipSwitchesRow");
  row.innerHTML = "";
  bits.forEach((on, i) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `dip-switch${on ? " on" : ""}`;
    btn.textContent = `${8 - i}`;
    btn.title = `Pin ${8 - i} weight ${DIP_WEIGHTS[i]}`;
    btn.addEventListener("click", async () => {
      const next = [...bits];
      next[i] = next[i] ? 0 : 1;
      state.dipBits = next;
      renderDipSwitches(next);
      syncDipRockers(next);
      try {
        await apiPost("/api/video/mode", { mode: $("selectCameraMode").value, dip_bits: next });
        await apiPost("/api/modbus/mode", { mode: $("selectModbusMode").value, slave_id: bitsToAddr(next) });
      } catch (_) {
        /* offline / server starting */
      }
    });
    row.appendChild(btn);
  });
}

// ——— Status poll ———
async function pollStatus() {
  const t0 = performance.now();
  try {
    const data = await apiJson("/api/status");
    state.status = data;
    state.lastStatusTs = Date.now();
    applyStatus(data);
    const dt = performance.now() - t0;
    if (state.fpsEwma === 0) state.fpsEwma = 1000 / dt;
    else state.fpsEwma = state.fpsEwma * 0.85 + (1000 / dt) * 0.15;
  } catch (e) {
    $("liveFeedProof").classList.add("stale");
    $("liveFeedProofText").textContent = "Live proof: API unreachable";
  }
  setTimeout(pollStatus, POLL_MS);
}

function applyStatus(data) {
  const vision = data.vision || {};
  const modbus = data.modbus || {};
  const repair = data.repair || {};
  const system = data.system || {};

  $("pillModelText").textContent = `Model: ${system.sensor_id || "TZ-THT-02"}`;
  $("pillRamText").textContent = `RAM: ${system.ram_mb != null ? `${system.ram_mb} MB` : "—"}`;
  const offline = data.offline_verified !== false;
  $("pillAirplaneText").textContent = offline ? "Offline verified" : "Network guard alert";
  $("pillAirplane").classList.toggle("pill--offline", !offline);
  $("hudIsolationPill").textContent = offline ? "Offline guard" : "Guard tripped";

  const vfps = vision.fps ?? 0;
  $("fpsCounterText").textContent = `FPS: ${vfps.toFixed(1)}`;
  $("camSourceBadge").textContent = (vision.active_source || "—").toUpperCase();
  $("pillCamText").textContent = `Camera: ${vision.camera_mode || "—"}`;

  $("scanDimVal").textContent = vision.estimated_length_mm
    ? `${vision.estimated_length_mm.toFixed(1)} mm`
    : "— mm";
  const cov = vision.detected ? (vision.tracking_locked ? 92 : 64) : 12;
  $("scanCoverageVal").textContent = `${cov}%`;
  $("scanCoverageFill").style.width = `${cov}%`;
  $("scanAngleVal").textContent = vision.angle_deg != null ? `${vision.angle_deg.toFixed(1)}°` : "—°";
  $("scanZoneVal").textContent = vision.anomaly_zone_label || vision.tool_target_zone || "—";

  drawSpatialHud(vision);

  $("liveFeedProof").classList.remove("stale");
  $("liveFeedProofText").textContent = `Live proof: frame ${vision.frame_idx ?? "—"} @ ${vfps.toFixed(0)} fps`;

  const stage = repair.stage || "IDLE_MONITORING";
  $("systemStateText").textContent = stage;
  $("systemStateSub").textContent =
    repair.proactive_event || repair.voice_prompt || "Reactive watchdog @ 350 ms";
  const andon = (repair.andon_state || modbus.andon_state || "GREEN").toUpperCase();
  const banner = $("systemStateBanner");
  banner.className =
    andon === "RED" ? "state-red" : andon === "AMBER" ? "state-amber" : "state-green";

  $("andonRed").classList.toggle("lit", andon === "RED");
  $("andonAmber").classList.toggle("lit", andon === "AMBER");
  $("andonGreen").classList.toggle("lit", andon === "GREEN");

  $("scadaTempVal").textContent =
    modbus.temp_c != null ? `${Number(modbus.temp_c).toFixed(1)}°C` : "—";
  $("scadaHumVal").textContent =
    modbus.humidity_rh != null ? `${Number(modbus.humidity_rh).toFixed(1)}%` : "—";
  $("scadaSlaveVal").textContent = modbus.slave_id != null ? String(modbus.slave_id) : "—";

  const badge = $("scadaStatusBadge");
  badge.textContent = modbus.online
    ? modbus.healthy
      ? "HEALTHY"
      : modbus.sensor_abnormal_7fff
        ? "0x7FFF"
        : "DEGRADED"
    : "OFFLINE";
  badge.className = modbus.healthy ? "healthy" : "fault";

  $("scadaCrcPill").textContent = `CRC: ${modbus.crc_ok === false ? "FAIL" : "OK"}`;
  $("scadaFrameHex").textContent = `TX: ${modbus.last_tx_hex || "—"}\nRX: ${modbus.last_rx_hex || "—"}`;
  pushScadaSample(modbus);

  if (repair.target_dip_bits) {
    state.dipBits = repair.target_dip_bits;
    renderDipSwitches(state.dipBits);
    syncDipRockers(state.dipBits);
  }
  if (repair.explode_pcb != null) {
    applyExplode(repair.explode_pcb);
    $("btnToggleExplode").classList.toggle("active", repair.explode_pcb);
  }

  if (repair.highlighted_zone) {
    highlightZone(repair.highlighted_zone);
    $("faultCallout").classList.add("visible");
    $("faultCalloutZone").textContent = repair.highlighted_zone;
    $("faultCalloutText").textContent = repair.proactive_event || "Zone highlighted for repair";
  } else {
    $("faultCallout").classList.remove("visible");
  }

  $("twinMatchBadge").textContent = vision.detected
    ? `Twin sync: ${(vision.estimated_length_mm || 140).toFixed(0)} mm`
    : "Twin sync: waiting for pose";

  if (repair.manual_page) setManualPage(repair.manual_page);

  if (repair.voice_prompt) {
    $("nextStepBody").textContent = repair.voice_prompt;
  }
  updateRepairStepButtons(repair.stage);

  if (vision.dip_crop_available) {
    $("dipCropImg").src = `/api/video/dip_crop?t=${Date.now()}`;
  }
}

function updateRepairStepButtons(stage) {
  const map = {
    STEP1_OPEN_FAULT_DETECTED: "btnStepInspect",
    STEP2_GUIDING_DIP_MODBUS: "btnStepDip",
    STEP3_SCADA_ONLINE_VERIFIED: "btnStepConnect",
    STEP4_SIGNOFF_PACKAGE_READY: "btnStepSignoff",
  };
  document.querySelectorAll(".repair-steps .btn").forEach((b) => b.classList.remove("active"));
  const id = map[stage];
  if (id) $(id)?.classList.add("active");
}

// ——— Manual ———
async function loadManualIndex() {
  try {
    const idx = await apiJson("/api/manual/index");
    state.manualIndex = idx.pages || idx || [];
    state.manualTotal = state.manualIndex.length || idx.count || 12;
    renderManualQuickPills();
    setManualPage(1);
  } catch {
    state.manualTotal = 12;
    setManualPage(1);
  }
}

function renderManualQuickPills() {
  const wrap = $("manualQuickPills");
  wrap.innerHTML = "";
  const pages = [1, 5, 6, 7, 8, 9];
  pages.forEach((p) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "quick-pill";
    b.textContent = `p${p}`;
    b.addEventListener("click", () => setManualPage(p));
    wrap.appendChild(b);
  });
}

function setManualPage(n) {
  state.manualPage = Math.max(1, Math.min(n, state.manualTotal || 12));
  const url = `/api/manual/page/${state.manualPage}`;
  $("manualPageImg").src = url;
  $("lightboxImg").src = url;
  $("manualPageIndicator").textContent = `Page ${state.manualPage} / ${state.manualTotal}`;
  $("lightboxPageBadge").textContent = `Page ${state.manualPage}`;
  const entry = state.manualIndex.find((p) => p.page === state.manualPage);
  $("manualPageCaption").textContent = entry?.title || entry?.snippet?.slice(0, 80) || `Manual page ${state.manualPage}`;
  document.querySelectorAll(".quick-pill").forEach((el) => {
    el.classList.toggle("active", el.textContent === `p${state.manualPage}`);
  });
}

// ——— Voice / VAD ———
function pushChat(role, text) {
  const feed = $("chatFeed");
  const div = document.createElement("div");
  div.className = `chat-bubble ${role}`;
  div.textContent = text;
  feed.appendChild(div);
  feed.scrollTop = feed.scrollHeight;
}

async function playTts(text) {
  if (!state.autoSpeak || !text) return;
  state.aiSpeakCooldownUntil = Date.now() + 2500 + text.length * 40;
  try {
    const res = await apiPost("/api/tts", { text });
    if (res.audio_wav_b64) {
      const audio = new Audio(`data:audio/wav;base64,${res.audio_wav_b64}`);
      audio.play();
    }
  } catch {
    if ("speechSynthesis" in window) {
      const u = new SpeechSynthesisUtterance(text);
      u.lang = "en-US";
      speechSynthesis.speak(u);
    }
  }
}

async function submitQuery(text) {
  if (!text?.trim()) return;
  pushChat("operator", text.trim());
  try {
    const views = captureMultiAngle3DViews();
    const res = await apiPost("/api/diagnose", {
      query: text.trim(),
      cad_views: views,
    });
    const reply = res.reply || res.diagnosis || res.message || JSON.stringify(res).slice(0, 200);
    pushChat("assistant", reply);
    $("nextStepTitle").textContent = res.mode ? `Mode: ${res.mode}` : "Assistant";
    $("nextStepBody").textContent = reply;
    await playTts(reply);
    if (res.highlighted_zone) highlightZone(res.highlighted_zone);
    if (res.manual_page) setManualPage(res.manual_page);
  } catch (e) {
    pushChat("assistant", `Request failed: ${e.message}`);
  }
}

function setMeterLevel(rms) {
  const pct = Math.min(100, Math.max(0, rms * 280));
  $("voiceMeterFill").style.width = `${pct}%`;
}

async function startHandsFreeVad() {
  if (state.vadStream) return;
  try {
    state.audioCtx = new AudioContext();
    state.vadStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const src = state.audioCtx.createMediaStreamSource(state.vadStream);
    state.vadAnalyser = state.audioCtx.createAnalyser();
    state.vadAnalyser.fftSize = 1024;
    src.connect(state.vadAnalyser);
    $("handsFreeText").textContent = "VAD listening";
    vadLoop();
  } catch (e) {
    $("handsFreeText").textContent = "Mic denied";
    state.handsFree = false;
  }
}

function stopHandsFreeVad() {
  state.vadStream?.getTracks().forEach((t) => t.stop());
  state.vadStream = null;
  state.audioCtx?.close();
  state.audioCtx = null;
  $("handsFreeText").textContent = "VAD off";
}

let vadSpeechMs = 0;
let vadArmed = false;

function vadLoop() {
  if (!state.handsFree || !state.vadAnalyser) return;
  const buf = new Float32Array(state.vadAnalyser.fftSize);
  state.vadAnalyser.getFloatTimeDomainData(buf);
  let sum = 0;
  for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
  const rms = Math.sqrt(sum / buf.length);
  setMeterLevel(rms);

  if (Date.now() < state.aiSpeakCooldownUntil) {
    requestAnimationFrame(vadLoop);
    return;
  }

  const talking = rms > 0.028;
  if (talking) {
    vadSpeechMs += 50;
    if (vadSpeechMs > 400 && !vadArmed) vadArmed = true;
  } else if (vadArmed) {
    vadArmed = false;
    vadSpeechMs = 0;
    finalizeVoiceCapture();
  } else {
    vadSpeechMs = 0;
  }
  requestAnimationFrame(vadLoop);
}

let voiceChunks = [];

async function startPttRecording() {
  if (state.mediaRecorder) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    voiceChunks = [];
    const rec = new MediaRecorder(stream, { mimeType: "audio/webm" });
    state.mediaRecorder = rec;
    rec.ondataavailable = (e) => voiceChunks.push(e.data);
    rec.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      sendVoiceBlob(new Blob(voiceChunks, { type: "audio/webm" }));
      state.mediaRecorder = null;
    };
    rec.start();
    $("btnVoiceRecord").classList.add("recording");
  } catch (e) {
    pushChat("assistant", `Microphone error: ${e.message}`);
  }
}

function stopPttRecording() {
  state.mediaRecorder?.stop();
  $("btnVoiceRecord").classList.remove("recording");
}

async function finalizeVoiceCapture() {
  await startPttRecording();
  setTimeout(stopPttRecording, 120);
}

async function sendVoiceBlob(blob) {
  const fd = new FormData();
  fd.append("audio", blob, "ptt.webm");
  try {
    state.aiSpeakCooldownUntil = Date.now() + 1500;
    const res = await fetch("/api/voice", { method: "POST", body: fd });
    const data = await res.json();
    if (data.empty_speech) {
      pushChat("assistant", "(empty speech ignored)");
      return;
    }
    const transcript = data.transcript || data.query || "";
    if (transcript) await submitQuery(transcript);
    else if (data.reply) {
      pushChat("assistant", data.reply);
      await playTts(data.reply);
    }
  } catch (e) {
    pushChat("assistant", `Voice API error: ${e.message}`);
  }
}

// ——— Escalation modal ———
async function openEscalation() {
  $("escalationModal").classList.add("open");
  const views = captureMultiAngle3DViews();
  try {
    const pkg = await apiPost("/api/escalate", {
      operator_notes: $("queryInput").value,
      cad_views: views,
    });
    fillEscalation(pkg);
  } catch (e) {
    $("escOutboxState").textContent = `Outbox: error — ${e.message}`;
  }
}

function fillEscalation(pkg) {
  $("escOutboxState").textContent = `Outbox: ${pkg.state || "queued"}`;
  $("escPackagePath").textContent = `Package: ${pkg.package_path || pkg.package_id || "—"}`;
  const zip = pkg.zip_url || pkg.download_url;
  if (zip) {
    $("escZipDownloadLink").href = zip;
  }
  $("escEmailTo").textContent = pkg.email_to || "manufacturer-escalation@localhost";
  $("escEmailSubject").textContent = pkg.email_subject || "FieldScan escalation";
  $("escEmailBody").value = pkg.email_body || "";
  const list = $("escAttachmentsList");
  list.innerHTML = "";
  (pkg.attachments || []).forEach((a) => {
    const li = document.createElement("li");
    li.textContent = a;
    list.appendChild(li);
  });
  if (pkg.quad_preview) $("tabPreviewQuad").src = pkg.quad_preview;
  if (pkg.strip_preview) $("tabPreviewStrip").src = pkg.strip_preview;
  if (pkg.snapshot) $("escSnapshotImg").src = pkg.snapshot;
  $("escSummaryZone").textContent = `Zone: ${pkg.fault_zone || "—"}`;
  $("escSummaryDip").textContent = `DIP: ${pkg.dip_address ?? bitsToAddr(state.dipBits)}`;
  $("escQueueTag").textContent = `Esc queue: ${pkg.state || "ready"}`;
}

// ——— Wire UI ———
function bindUi() {
  $("btnCycleCvHud").addEventListener("click", () => {
    state.cvHudMode = (state.cvHudMode + 1) % CV_HUD_MODES.length;
    $("cvHudModeLabel").textContent = CV_HUD_MODES[state.cvHudMode];
    drawSpatialHud(state.status?.vision);
  });

  $("btnToggleDipPip").addEventListener("click", () => $("dipMacroPip").classList.add("open"));
  $("btnCloseDipPip").addEventListener("click", () => $("dipMacroPip").classList.remove("open"));

  $("btnRawLiveToggle").addEventListener("click", () => {
    state.rawStream = !state.rawStream;
    $("videoStream").src = state.rawStream
      ? "/api/video/stream?raw=1"
      : "/api/video/stream";
    $("btnRawLiveToggle").textContent = state.rawStream ? "Raw MJPEG" : "Annotated MJPEG";
  });

  $("btnZoomAuto").addEventListener("click", () => {
    $("videoStream").style.objectFit = "cover";
    pushChat("assistant", "Auto zoom: object-fit cover engaged on live feed.");
  });

  $("selectCameraMode").addEventListener("change", async (e) => {
    try {
      await apiPost("/api/video/mode", { mode: e.target.value, dip_bits: state.dipBits });
    } catch (_) {}
  });

  $("btnUploadPhoto").addEventListener("click", () => $("photoUploadInput").click());
  $("photoUploadInput").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    try {
      await fetch("/api/video/upload", { method: "POST", body: fd });
      pushChat("assistant", `Uploaded ${file.name} to vision pipeline.`);
    } catch (err) {
      pushChat("assistant", `Upload failed: ${err.message}`);
    }
  });

  $("manualPrevBtn").addEventListener("click", () => setManualPage(state.manualPage - 1));
  $("manualNextBtn").addEventListener("click", () => setManualPage(state.manualPage + 1));
  $("btnToggleManualDrawer").addEventListener("click", () =>
    $("manualDrawer").classList.toggle("collapsed")
  );
  $("btnExpandManual").addEventListener("click", () => {
    $("manualModal").classList.add("open");
    setManualPage(state.manualPage);
  });
  $("lightboxCloseBtn").addEventListener("click", () => $("manualModal").classList.remove("open"));
  $("lightboxPrevBtn").addEventListener("click", () => setManualPage(state.manualPage - 1));
  $("lightboxNextBtn").addEventListener("click", () => setManualPage(state.manualPage + 1));

  $("btnModeSolid").addEventListener("click", () => setWireframe(false));
  $("btnModeWireframe").addEventListener("click", () => setWireframe(true));
  $("btnToggleExplode").addEventListener("click", () => {
    applyExplode(!state.exploded);
    $("btnToggleExplode").classList.toggle("active", state.exploded);
    apiPost("/api/repair/step", { action: "explode", value: state.exploded }).catch(() => {});
  });
  $("btnResetView").addEventListener("click", () => {
    camera?.position.set(2.4, 1.6, 3.2);
    controls.target.set(0, 0.7, 0);
    controls.update();
  });

  $("selectModbusMode").addEventListener("change", async (e) => {
    try {
      await apiPost("/api/modbus/mode", {
        mode: e.target.value,
        slave_id: bitsToAddr(state.dipBits),
      });
    } catch (_) {}
  });
  $("btnScanRealModbus").addEventListener("click", async () => {
    try {
      await apiPost("/api/modbus/mode", { mode: "hardware" });
      $("selectModbusMode").value = "hardware";
    } catch (_) {}
  });
  $("btnSimConnectModbus").addEventListener("click", async () => {
    try {
      await apiPost("/api/modbus/mode", { mode: "sim_healthy", slave_id: bitsToAddr(state.dipBits) });
      $("selectModbusMode").value = "sim_healthy";
    } catch (_) {}
  });
  $("btnBreathTest").addEventListener("click", async () => {
    try {
      await apiPost("/api/modbus/mode", { mode: "sim_healthy" });
      pushChat("assistant", "Breath test: sim injection of 24.6 °C / 56 %RH.");
    } catch (_) {}
  });

  const repairStep = async (action) => {
    try {
      const res = await apiPost("/api/repair/step", { action });
      if (res.voice_prompt) $("nextStepBody").textContent = res.voice_prompt;
    } catch (_) {}
  };
  $("btnStepInspect").addEventListener("click", () => repairStep("inspect"));
  $("btnStepDip").addEventListener("click", () => repairStep("dip"));
  $("btnStepConnect").addEventListener("click", () => repairStep("connect"));
  $("btnStepSignoff").addEventListener("click", async () => {
    const views = captureMultiAngle3DViews();
    try {
      const res = await apiPost("/api/repair/signoff", {
        operator_notes: $("queryInput").value,
        cad_views: views,
      });
      pushChat("assistant", `Sign-off package: ${res.package_id || "ready"}`);
      fillEscalation(res);
    } catch (e) {
      pushChat("assistant", `Sign-off failed: ${e.message}`);
    }
  });

  $("btnHandsFreeMic").addEventListener("click", () => {
    state.handsFree = !state.handsFree;
    $("btnHandsFreeMic").classList.toggle("active", state.handsFree);
    if (state.handsFree) startHandsFreeVad();
    else stopHandsFreeVad();
  });

  $("btnAutoSpeak").addEventListener("click", () => {
    state.autoSpeak = !state.autoSpeak;
    $("autoSpeakText").textContent = state.autoSpeak ? "TTS on" : "TTS off";
    $("btnAutoSpeak").classList.toggle("active", state.autoSpeak);
  });

  $("btnVoiceRecord").addEventListener("mousedown", startPttRecording);
  $("btnVoiceRecord").addEventListener("mouseup", stopPttRecording);
  $("btnVoiceRecord").addEventListener("mouseleave", () => state.pttActive && stopPttRecording());
  $("btnVoiceRecord").addEventListener("touchstart", (e) => {
    e.preventDefault();
    startPttRecording();
  });
  $("btnVoiceRecord").addEventListener("touchend", (e) => {
    e.preventDefault();
    stopPttRecording();
  });

  $("textQueryForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = $("queryInput").value;
    $("queryInput").value = "";
    submitQuery(q);
  });

  $("btnOpenEscalateModal").addEventListener("click", openEscalation);
  $("escalationCloseBtn").addEventListener("click", () => $("escalationModal").classList.remove("open"));

  document.querySelectorAll(".esc-tabs .btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      document.querySelectorAll(".esc-tabs .btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      $("tabPreviewQuad").hidden = tab !== "quad";
      $("tabPreviewStrip").hidden = tab !== "strip";
    });
  });

  window.addEventListener("resize", () => {
    onTwinResize();
    resizeHudCanvas();
    if (scadaPlot) {
      scadaPlot.setSize({ width: $("scadaChart").clientWidth, height: 110 });
    }
  });

  new ResizeObserver(() => resizeHudCanvas()).observe($("videoFrame"));
}

function boot() {
  buildTwin();
  renderDipSwitches(state.dipBits);
  initScadaChart();
  bindUi();
  loadManualIndex();
  pollStatus();
  $("cvHudModeLabel").textContent = CV_HUD_MODES[0];
  $("hudSensorId").textContent = "TZ-THT-02";
  $("hudStatusTag").textContent = "SCAN";
  $("btnSensorPill").addEventListener("click", () =>
    pushChat("assistant", "TZ THT-02 · RS-485 Modbus RTU · SHT30 probe.")
  );
}

boot();

export { captureMultiAngle3DViews, vidPt };