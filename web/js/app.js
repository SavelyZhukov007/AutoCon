import { ICONS, mountIcons } from "./icons.js";
import { $, $$, Bus, esc, fmt, ready } from "./bridge.js";

let API;
const state = {
  settings: {},
  project: null,
  detections: [],
  currentDetections: [],
  signSeq: [],
  vehicles: [],
  plates: [],
  comments: [],
  cameraRunning: false,
};

ready().then(async (api) => {
  API = api;
  mountIcons();
  $("#logo").innerHTML = await (await fetch("assets/logo.svg")).text();
  bind();
  bindBus();
  await refreshEnv();
  await refreshCameras();
  await maybeFirstRun();
});

function bind() {
  $("#btnOpen").addEventListener("click", openVideo);
  $("#btnAnalyze").addEventListener("click", analyze);
  $("#btnExport").addEventListener("click", exportReport);
  $("#btnCamera").addEventListener("click", () => showView("camera"));
  $("#btnPlay").addEventListener("click", togglePlay);
  $("#seek").addEventListener("input", seekVideo);
  $("#video").addEventListener("timeupdate", onVideoTime);
  $("#video").addEventListener("loadedmetadata", () => $("#tcDur").textContent = fmt($("#video").duration));
  $("#btnStartCamera").addEventListener("click", startCamera);
  $("#btnStopCamera").addEventListener("click", stopCamera);
  $("#btnSettings").addEventListener("click", openSettings);
  $("#settingsClose").addEventListener("click", () => $("#settingsOverlay").classList.remove("open"));
  $("#settingsSave").addEventListener("click", saveSettings);
  $("#installClose").addEventListener("click", () => $("#installOverlay").classList.remove("open"));
  $("#installSkip").addEventListener("click", finishFirstRun);
  $("#installRun").addEventListener("click", installSelected);
  $$(".rail-btn[data-view]").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));
  $$(".tab").forEach((t) => t.addEventListener("click", () => showPane(t.dataset.pane)));
  window.addEventListener("resize", () => drawBoxes($("#videoOverlay"), state.currentDetections, $("#video")));
}

function bindBus() {
  Bus.on("install:progress", (d) => {
    $("#installProgress").classList.remove("hidden");
    if (typeof d.progress === "number") $("#installFill").style.width = Math.round(d.progress * 100) + "%";
    $("#installLog").classList.remove("hidden");
    $("#installLog").textContent = `${d.text || ""}\nlog: ${d.log_path || ""}`;
  });
  Bus.on("install:done", async (d) => {
    $("#installLog").textContent = d.ok ? "Runtime установлен." : `Ошибка установки: ${JSON.stringify(d.failed || d.error || d)}`;
    await refreshEnv();
  });
  Bus.on("model:progress", (d) => {
    $("#installProgress").classList.remove("hidden");
    if (typeof d.progress === "number") $("#installFill").style.width = Math.round(d.progress * 100) + "%";
    $("#installLog").classList.remove("hidden");
    $("#installLog").textContent = d.text || "";
  });
  Bus.on("model:done", async (d) => {
    $("#installLog").classList.remove("hidden");
    $("#installLog").textContent = d.ok ? `Модель готова: ${d.path || ""}` : `Модель не установлена: ${d.error || ""}`;
    await refreshEnv();
  });
  Bus.on("pull:progress", (d) => {
    $("#installLog").classList.remove("hidden");
    $("#installLog").textContent = `Ollama: ${d.model}\n${d.text || ""}`;
  });
  Bus.on("process:start", () => {
    $("#processBox").classList.remove("hidden");
    $("#processPct").textContent = "0%";
    $("#processFill").style.width = "0";
    $("#processText").textContent = "Анализ видео";
  });
  Bus.on("process:progress", (d) => {
    const pct = Math.round((d.progress || 0) * 100);
    $("#processPct").textContent = pct + "%";
    $("#processFill").style.width = pct + "%";
    $("#processText").textContent = d.label || "Анализ";
  });
  Bus.on("process:done", (p) => {
    $("#processBox").classList.add("hidden");
    state.project = p;
    state.signSeq = p.sign_sequences || [];
    state.vehicles = p.vehicles || [];
    state.plates = p.plates || [];
    state.comments = p.comments || [];
    $("#btnExport").disabled = false;
    renderAll();
  });
  Bus.on("vision:detections", (d) => {
    state.detections.push(...(d.detections || []));
    state.currentDetections = d.detections || [];
    drawBoxes($("#videoOverlay"), state.currentDetections, $("#video"));
  });
  Bus.on("vision:event", (d) => {
    addEvent(d);
    if (d.kind === "sign_sequence") {
      state.signSeq.push(d);
      renderSigns();
    }
  });
  Bus.on("vision:commentary", (d) => {
    state.comments.push(d);
    $("#liveComment").textContent = d.text || "";
    renderComments();
  });
  Bus.on("vision:frame", (d) => {
    $("#emptyCamera").classList.add("hidden");
    $("#cameraFrame").src = d.image || "";
    state.currentDetections = d.detections || [];
    requestAnimationFrame(() => drawBoxes($("#cameraOverlay"), state.currentDetections, $("#cameraFrame")));
  });
  Bus.on("camera:error", (d) => {
    $("#liveComment").textContent = d.message || "Ошибка камеры";
  });
  Bus.on("error", (d) => {
    $("#processBox").classList.add("hidden");
    $("#liveComment").textContent = d.message || "Ошибка";
  });
}

async function refreshEnv() {
  const env = await API.environment();
  state.settings = env.settings || {};
  flag("#envDevice", true, env.device?.selected_device || "CPU");
  flag("#envRuntime", (env.packages || []).some((p) => p.key === "vision" && p.installed), "runtime");
  flag("#envModels", (env.model_packs || []).some((p) => p.installed), "models");
  flag("#envOllama", env.ollama, env.ollama_model || "ollama");
  return env;
}

function flag(sel, ok, text) {
  const el = $(sel);
  el.classList.toggle("ok", !!ok);
  el.classList.toggle("bad", !ok);
  el.querySelector("b") ? el.querySelector("b").textContent = text : el.lastChild.textContent = text;
}

async function maybeFirstRun() {
  const env = await refreshEnv();
  const packages = env.packages || [];
  const models = env.model_packs || [];
  if (env.first_run_done && packages.some((p) => p.key === "vision" && p.installed)) return;
  $("#installList").innerHTML = packages.map((p) => checkRow(p.key, p.title, p.desc, p.installed, p.recommended, p.available)).join("");
  $("#modelList").innerHTML = models.map((p) => checkRow("model:" + p.key, p.title, p.desc, p.installed, p.recommended, true)).join("");
  mountIcons($("#installOverlay"));
  $("#installOverlay").classList.add("open");
}

function checkRow(key, title, desc, installed, recommended, available) {
  const checked = installed || recommended ? "checked" : "";
  const disabled = installed || !available ? "disabled" : "";
  return `<label class="check"><input type="checkbox" data-key="${esc(key)}" ${checked} ${disabled}/><span><h4>${esc(title)}</h4><p>${esc(desc)}</p></span><b class="state">${installed ? "готово" : recommended ? "рекомендовано" : ""}</b></label>`;
}

async function installSelected() {
  const keys = $$("#installList input:checked:not(:disabled)").map((x) => x.dataset.key);
  const models = $$("#modelList input:checked:not(:disabled)").map((x) => x.dataset.key.replace("model:", ""));
  if (keys.length) API.install_packages(keys);
  for (const key of models) API.install_model_pack(key);
  if (!keys.length && !models.length) finishFirstRun();
}

async function finishFirstRun() {
  await API.finish_first_run();
  $("#installOverlay").classList.remove("open");
  await refreshEnv();
}

async function openVideo() {
  const p = await API.pick_video();
  if (!p) return;
  loadProject(p);
}

function loadProject(p) {
  state.project = p;
  state.detections = p.detections || [];
  state.signSeq = p.sign_sequences || [];
  state.vehicles = p.vehicles || [];
  state.plates = p.plates || [];
  state.comments = p.comments || [];
  $("#video").src = p.media_url;
  $("#emptyVideo").classList.add("hidden");
  $("#btnAnalyze").disabled = false;
  $("#btnExport").disabled = !state.signSeq.length && !state.vehicles.length && !state.plates.length;
  showView("video");
  renderAll();
}

async function analyze() {
  state.detections = [];
  await API.process_video();
}

async function exportReport() {
  const res = await API.export_report("md");
  $("#liveComment").textContent = res.ok ? `Отчёт сохранён: ${res.path}` : res.error;
}

function togglePlay() {
  const v = $("#video");
  if (!v.src) return;
  v.paused ? v.play() : v.pause();
  $("#btnPlay").innerHTML = v.paused ? ICONS.play : ICONS.pause;
}

function seekVideo(e) {
  const v = $("#video");
  if (!v.duration) return;
  v.currentTime = (Number(e.target.value) / 1000) * v.duration;
}

function onVideoTime() {
  const v = $("#video");
  $("#tcCur").textContent = fmt(v.currentTime);
  if (v.duration) $("#seek").value = Math.round((v.currentTime / v.duration) * 1000);
  const t = v.currentTime;
  state.currentDetections = state.detections.filter((d) => Math.abs((d.time || 0) - t) < 0.35);
  drawBoxes($("#videoOverlay"), state.currentDetections, v);
}

async function refreshCameras() {
  const cams = await API.list_camera_devices();
  $("#cameraList").innerHTML = cams.length
    ? cams.map((c) => `<option value="${c.index}">${esc(c.name)} ${c.width ? `(${c.width}x${c.height})` : ""}</option>`).join("")
    : `<option value="0">Камера 0</option>`;
}

async function startCamera() {
  showView("camera");
  state.cameraRunning = true;
  await API.start_camera(Number($("#cameraList").value || 0));
}

async function stopCamera() {
  state.cameraRunning = false;
  await API.stop_camera();
}

function drawBoxes(canvas, detections, mediaEl) {
  if (!canvas || !mediaEl) return;
  const rect = mediaEl.getBoundingClientRect();
  const parent = canvas.parentElement.getBoundingClientRect();
  canvas.width = parent.width;
  canvas.height = parent.height;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const naturalW = mediaEl.videoWidth || mediaEl.naturalWidth || rect.width || canvas.width;
  const naturalH = mediaEl.videoHeight || mediaEl.naturalHeight || rect.height || canvas.height;
  const scale = Math.min(canvas.width / naturalW, canvas.height / naturalH);
  const drawW = naturalW * scale;
  const drawH = naturalH * scale;
  const ox = (canvas.width - drawW) / 2;
  const oy = (canvas.height - drawH) / 2;
  for (const d of detections || []) {
    const [x1, y1, x2, y2] = d.bbox || [0, 0, 0, 0];
    const x = ox + x1 * scale, y = oy + y1 * scale, w = (x2 - x1) * scale, h = (y2 - y1) * scale;
    const color = d.kind === "sign" ? "#f0b441" : d.kind === "plate" ? "#ee5d50" : "#18b7a6";
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
    const label = `${d.label || d.kind} ${Math.round((d.confidence || 0) * 100)}%`;
    ctx.font = "12px Segoe UI";
    const tw = ctx.measureText(label).width + 10;
    ctx.fillStyle = color;
    ctx.fillRect(x, Math.max(0, y - 21), tw, 20);
    ctx.fillStyle = "#061412";
    ctx.fillText(label, x + 5, Math.max(14, y - 7));
  }
}

function addEvent(e) {
  const box = $("#eventList");
  box.insertAdjacentHTML("afterbegin", `<div class="item sign"><b>${esc(e.label || e.kind)}</b><div class="meta">${fmt(e.start || e.time || 0)} · ${esc(e.position || "")}</div></div>`);
}

function renderAll() {
  renderSigns();
  renderVehicles();
  renderComments();
}

function renderSigns() {
  $("#signList").innerHTML = state.signSeq.length ? state.signSeq.map((s) =>
    `<div class="item sign"><b>${esc(s.label)}</b><span class="tag">x${s.count || 1}</span><div class="meta">${fmt(s.start)}–${fmt(s.end)} · ${esc(s.position || "")} · ${Math.round((s.confidence || 0) * 100)}%</div></div>`
  ).join("") : `<div class="item">Знаки появятся после анализа.</div>`;
}

function renderVehicles() {
  const rows = [];
  rows.push(...(state.vehicles || []).map((v) => `<div class="item vehicle"><b>${esc(v.label || "vehicle")}</b><div class="meta">${fmt(v.first_t)}–${fmt(v.last_t)} · id ${esc(v.track_id ?? "-")} · ${esc(v.position || "")}</div></div>`));
  rows.push(...(state.plates || []).map((p) => `<div class="item plate"><b>${esc(p.text)}</b><div class="meta">${fmt(p.first_t)}–${fmt(p.last_t)} · ${Math.round((p.confidence || 0) * 100)}%</div></div>`));
  $("#vehicleList").innerHTML = rows.length ? rows.join("") : `<div class="item">Авто и номера появятся после анализа.</div>`;
}

function renderComments() {
  $("#commentsList").innerHTML = (state.comments || []).slice().reverse().map((c) =>
    `<div class="item"><b>${fmt(c.t || c.time || 0)}</b><div>${esc(c.text || "")}</div></div>`
  ).join("");
}

function showView(name) {
  $$(".rail-btn[data-view]").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
  $("#viewVideo").classList.toggle("hidden", name !== "video" && name !== "events");
  $("#viewCamera").classList.toggle("hidden", name !== "camera");
  if (name === "events") showPane("events");
}

function showPane(name) {
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.pane === name));
  $$(".pane").forEach((p) => p.classList.remove("active"));
  $("#pane" + name[0].toUpperCase() + name.slice(1)).classList.add("active");
}

function openSettings() {
  const s = state.settings || {};
  $("#settingsForm").innerHTML = `
    <div class="settings-grid">
      <label>Устройство</label><select data-k="device"><option value="auto">auto</option><option value="cpu">cpu</option><option value="cuda">cuda</option></select>
      <label>GPU index</label><input data-k="gpu_index" type="number" min="0" value="${esc(s.gpu_index ?? 0)}" />
      <label>Target FPS</label><input data-k="target_fps" type="number" min="1" max="60" value="${esc(s.target_fps ?? 8)}" />
      <label>YOLO imgsz</label><input data-k="imgsz" type="number" min="320" step="32" value="${esc(s.imgsz ?? 960)}" />
      <label>Confidence</label><input data-k="conf" type="number" min="0.05" max="0.95" step="0.05" value="${esc(s.conf ?? 0.35)}" />
      <label>Tracker</label><select data-k="tracker"><option>bytetrack.yaml</option><option>botsort.yaml</option></select>
      <label>Ollama host</label><input data-k="ollama_host" type="text" value="${esc(s.ollama_host || "http://127.0.0.1:11434")}" />
      <label>Ollama model</label><input data-k="ollama_model" type="text" value="${esc(s.ollama_model || "")}" placeholder="qwen2.5:3b" />
      <label>Комментарий</label><select data-k="commentary_enabled"><option value="true">on</option><option value="false">off</option></select>
      <label>Голос</label><select data-k="voice_enabled"><option value="false">off</option><option value="true">on</option></select>
    </div>`;
  $(`[data-k=device]`).value = s.device || "auto";
  $(`[data-k=tracker]`).value = s.tracker || "bytetrack.yaml";
  $(`[data-k=commentary_enabled]`).value = String(s.commentary_enabled !== false);
  $(`[data-k=voice_enabled]`).value = String(!!s.voice_enabled);
  $("#settingsOverlay").classList.add("open");
}

async function saveSettings() {
  const patch = {};
  $$("#settingsForm [data-k]").forEach((el) => {
    let v = el.value;
    if (el.type === "number") v = Number(v);
    if (v === "true") v = true;
    if (v === "false") v = false;
    patch[el.dataset.k] = v;
  });
  state.settings = await API.update_settings(patch);
  $("#settingsOverlay").classList.remove("open");
  await refreshEnv();
}
