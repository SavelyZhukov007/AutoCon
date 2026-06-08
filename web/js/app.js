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
  contexts: [],
  currentContext: null,
  cameraRunning: false,
  videoRealtimeRunning: false,
  examImage: null,
  chatAnswerEl: null,
  chatAnswerBuf: "",
  signHotspots: [],
  firstRunInstalling: false,
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
  $("#btnExam").addEventListener("click", () => showView("exam"));
  $("#btnPlay").addEventListener("click", togglePlay);
  $("#seek").addEventListener("input", seekVideo);
  $("#video").addEventListener("timeupdate", onVideoTime);
  $("#videoOverlay").addEventListener("click", explainClickedSign);
  $("#video").addEventListener("loadedmetadata", () => $("#tcDur").textContent = fmt($("#video").duration));
  $("#btnStartCamera").addEventListener("click", startCamera);
  $("#btnStopCamera").addEventListener("click", stopCamera);
  $("#examPick").addEventListener("click", pickExamImage);
  $("#examAnalyze").addEventListener("click", analyzeExamImage);
  $("#examPullModel").addEventListener("click", pullVisionModel);
  $("#chatRefresh").addEventListener("click", refreshChatProjects);
  $("#chatSend").addEventListener("click", sendChat);
  $("#chatQuestion").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendChat();
    }
  });
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
  Bus.on("setup:progress", (d) => {
    state.firstRunInstalling = true;
    $("#installProgress").classList.remove("hidden");
    if (typeof d.progress === "number") $("#installFill").style.width = Math.round(d.progress * 100) + "%";
    $("#installLog").classList.remove("hidden");
    $("#installLog").textContent = `${d.text || ""}\n${d.phase || ""}${d.model ? " · " + d.model : ""}${d.pack ? " · " + d.pack : ""}\nlog: ${d.log_path || ""}`;
    $("#installRun").disabled = true;
    $("#installSkip").disabled = true;
    $("#installClose").disabled = true;
  });
  Bus.on("setup:done", async (d) => {
    state.firstRunInstalling = false;
    $("#installRun").disabled = false;
    $("#installClose").disabled = false;
    $("#installLog").classList.remove("hidden");
    if (d.ok) {
      $("#installLog").textContent = `Готово.\nlog: ${d.log_path || ""}`;
      $("#installOverlay").classList.remove("open");
      await refreshEnv();
      return;
    }
    $("#installSkip").disabled = false;
    $("#installLog").textContent = `Ошибка первого запуска:\n${JSON.stringify(d.failed || d.error || d, null, 2)}\nlog: ${d.log_path || ""}`;
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
    state.detections = p.detections || state.detections || [];
    state.signSeq = p.sign_sequences || [];
    state.vehicles = p.vehicles || [];
    state.plates = p.plates || [];
    state.comments = p.comments || [];
    state.contexts = p.contexts || state.contexts || [];
    $("#btnExport").disabled = false;
    renderAll();
  });
  Bus.on("process:error", (d) => {
    $("#processBox").classList.add("hidden");
    $("#liveComment").textContent = d.message || "Ошибка анализа видео";
  });
  Bus.on("video:analysis_start", () => {
    state.videoRealtimeRunning = true;
    $("#videoRealtimeStatus").textContent = "анализ запущен";
  });
  Bus.on("video:analysis_status", (d) => {
    state.videoRealtimeRunning = !!d.running;
    const pct = Math.round((d.progress || 0) * 100);
    $("#videoRealtimeStatus").textContent = `${d.label || "анализ"} ${pct}% · до ${fmt(d.analyzed_time || 0)}`;
  });
  Bus.on("video:analysis_done", (d) => {
    state.videoRealtimeRunning = false;
    $("#videoRealtimeStatus").textContent = d.error ? `ошибка: ${d.error}` : (d.label || "анализ готов");
  });
  Bus.on("vision:detections", (d) => {
    state.detections.push(...(d.detections || []));
    onVideoTime();
  });
  Bus.on("vision:context", (d) => {
    state.contexts.push(d);
    const v = $("#video");
    if (!v.src || Math.abs((d.time || 0) - (v.currentTime || 0)) < 0.85) {
      renderCurrentContext(d);
    }
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
  Bus.on("camera:started", (d) => {
    state.cameraRunning = true;
    $("#liveComment").textContent = `Камера ${d.index ?? ""} запущена${d.backend ? " · " + d.backend : ""}`;
  });
  Bus.on("vision:frame", (d) => {
    $("#emptyCamera").classList.add("hidden");
    state.cameraRunning = true;
    $("#cameraFrame").src = d.image || "";
    state.currentDetections = d.detections || [];
    requestAnimationFrame(() => drawBoxes($("#cameraOverlay"), state.currentDetections, $("#cameraFrame")));
  });
  Bus.on("camera:error", (d) => {
    state.cameraRunning = false;
    $("#liveComment").textContent = d.message || "Ошибка камеры";
  });
  Bus.on("exam:image_loaded", (d) => {
    state.examImage = d;
    $("#examImage").src = d.image_url || "";
    $("#emptyExam").classList.add("hidden");
  });
  Bus.on("exam:analysis_start", () => {
    $("#examAnswer").textContent = "Анализирую изображение и вопрос...";
    $("#examDetections").innerHTML = "";
  });
  Bus.on("exam:analysis_done", (d) => {
    $("#examAnswer").textContent = d.answer || "Ответ пустой.";
    renderExamFindings(d.findings || {});
  });
  Bus.on("exam:error", (d) => {
    $("#examAnswer").textContent = d.message || "Ошибка анализа фото";
  });
  Bus.on("chat:start", (d) => {
    addChatMessage("user", d.question || "");
    state.chatAnswerBuf = "";
    state.chatAnswerEl = addChatMessage("ai", "", "AutoCon / Ollama");
  });
  Bus.on("chat:token", (d) => {
    state.chatAnswerBuf += d.token || "";
    if (state.chatAnswerEl) state.chatAnswerEl.querySelector(".txt").textContent = state.chatAnswerBuf;
  });
  Bus.on("chat:done", (d) => {
    if (state.chatAnswerEl) state.chatAnswerEl.querySelector(".txt").textContent = d.answer || state.chatAnswerBuf;
  });
  Bus.on("chat:error", (d) => {
    addChatMessage("ai", d.message || "Ошибка чата", "Ошибка");
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
  $("#installRun").disabled = false;
  $("#installSkip").disabled = false;
  $("#installClose").disabled = false;
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
  const ok = await API.finish_first_run();
  if (!ok) {
    $("#installLog").classList.remove("hidden");
    $("#installLog").textContent = "Первый запуск ещё не готов: установите все пакеты и модели.";
    return;
  }
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
  state.contexts = p.contexts || [];
  state.currentContext = null;
  $("#video").src = p.media_url;
  $("#emptyVideo").classList.add("hidden");
  $("#btnAnalyze").disabled = false;
  $("#btnExport").disabled = !state.signSeq.length && !state.vehicles.length && !state.plates.length;
  showView("video");
  renderAll();
  renderCurrentContext(null);
  analyze();
  refreshChatProjects();
}

async function analyze() {
  state.detections = [];
  state.contexts = [];
  state.signSeq = [];
  state.vehicles = [];
  state.plates = [];
  state.comments = [];
  state.signHotspots = [];
  $("#eventList").innerHTML = "";
  $("#signPopover").classList.add("hidden");
  $("#btnExport").disabled = true;
  renderAll();
  renderCurrentContext(null);
  const res = await API.process_video();
  if (res && res.ok === false) {
    $("#liveComment").textContent = res.error || "Не удалось начать анализ";
  }
}

async function startVideoRealtime(force = false) {
  if (!state.project || !API.start_video_realtime) return;
  if (state.videoRealtimeRunning && !force) return;
  const res = await API.start_video_realtime(state.project.id || "");
  if (res && res.ok === false) {
    $("#videoRealtimeStatus").textContent = res.error || "анализ не запущен";
  }
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
  renderCurrentContext(contextAt(t));
}

function contextAt(t) {
  let best = null;
  let bestDelta = Infinity;
  for (const ctx of state.contexts || []) {
    const delta = Math.abs((ctx.time || 0) - t);
    if (delta < bestDelta) {
      best = ctx;
      bestDelta = delta;
    }
  }
  return bestDelta <= 1.2 ? best : null;
}

function renderCurrentContext(ctx) {
  state.currentContext = ctx || null;
  if (!ctx) {
    $("#contextExplanation").textContent = state.videoRealtimeRunning
      ? "Анализ догоняет текущий момент видео."
      : "Загрузите видео: здесь появится объяснение текущих знаков и полосы.";
    $("#contextLane").textContent = "unknown";
    $("#contextApplies").textContent = "unknown";
    $("#contextConfidence").textContent = "0%";
    $("#contextSigns").innerHTML = "";
    return;
  }
  $("#contextExplanation").textContent = ctx.explanation || "Контекст сцены обновлён.";
  $("#contextLane").textContent = ctx.lane?.lane || "unknown";
  $("#contextApplies").textContent = ctx.applies_to_ego_lane || "unknown";
  $("#contextConfidence").textContent = Math.round((ctx.confidence || 0) * 100) + "%";
  const signs = ctx.visible_signs || [];
  $("#contextSigns").innerHTML = signs.length
    ? signs.map((s) => `<span>${esc(s.label)} · ${esc(s.applies_to_ego_lane || "unknown")}</span>`).join("")
    : `<span>знаки не подтверждены</span>`;
}

async function refreshCameras() {
  const cams = await API.list_camera_devices();
  $("#cameraList").innerHTML = cams.length
    ? cams.map((c) => `<option value="${c.index}">${esc(c.name)} ${c.width ? `(${c.width}x${c.height})` : ""}${c.backend ? ` · ${esc(c.backend)}` : ""}</option>`).join("")
    : `<option value="0">Камера 0</option>`;
}

async function startCamera() {
  showView("camera");
  state.cameraRunning = false;
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
  const collectHotspots = canvas.id === "videoOverlay";
  if (collectHotspots) state.signHotspots = [];
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
    const signLike = d.kind === "sign" || ["stop sign", "traffic light"].includes(String(d.label || "").toLowerCase());
    const color = signLike ? "#f0b441" : d.kind === "plate" ? "#ee5d50" : "#18b7a6";
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
    const label = `${d.label || d.kind} ${Math.round((d.confidence || 0) * 100)}%`;
    ctx.font = "12px Segoe UI";
    const tw = Math.min(ctx.measureText(label).width + 10, canvas.width - 4);
    const labelX = clamp(x, 2, Math.max(2, canvas.width - tw - 2));
    let labelY = signLike ? y + h + 3 : y - 21;
    if (signLike) labelY = Math.min(y + h + 4, Math.max(0, canvas.height - 20));
    else if (labelY + 20 > canvas.height) labelY = y - 21;
    labelY = clamp(labelY, 0, Math.max(0, canvas.height - 20));
    ctx.fillStyle = color;
    ctx.fillRect(labelX, labelY, tw, 20);
    ctx.fillStyle = "#061412";
    ctx.fillText(label, labelX + 5, labelY + 14);
    if (collectHotspots && signLike) {
      state.signHotspots.push({ x, y, w, h, labelX, labelY, labelW: tw, labelH: 20, detection: d });
    }
  }
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

async function explainClickedSign(e) {
  if (!state.project || !state.signHotspots.length || !API.explain_sign) return;
  const canvas = $("#videoOverlay");
  const rect = canvas.getBoundingClientRect();
  const sx = canvas.width / Math.max(1, rect.width);
  const sy = canvas.height / Math.max(1, rect.height);
  const x = (e.clientX - rect.left) * sx;
  const y = (e.clientY - rect.top) * sy;
  const hit = state.signHotspots.find((h) =>
    (x >= h.x && x <= h.x + h.w && y >= h.y && y <= h.y + h.h) ||
    (x >= h.labelX && x <= h.labelX + h.labelW && y >= h.labelY && y <= h.labelY + h.labelH)
  );
  if (!hit) {
    $("#signPopover").classList.add("hidden");
    return;
  }
  const pop = $("#signPopover");
  const left = clamp(hit.x + hit.w + 12, 8, Math.max(8, canvas.width - 300));
  const top = clamp(hit.y, 8, Math.max(8, canvas.height - 150));
  pop.style.left = left + "px";
  pop.style.top = top + "px";
  pop.classList.remove("hidden");
  pop.innerHTML = `<b>${esc(hit.detection.label || "sign")}</b><p>Анализирую влияние знака...</p>`;
  const v = $("#video");
  const ctx = state.currentContext || contextAt(v.currentTime || hit.detection.time || 0);
  const res = await API.explain_sign(state.project.id || "", hit.detection, ctx, v.currentTime || hit.detection.time || 0);
  pop.innerHTML = `<b>${esc(hit.detection.label || "sign")}</b><p>${esc(res?.answer || res?.error || "Нет ответа")}</p>`;
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
  $("#viewChat").classList.toggle("hidden", name !== "chat");
  $("#viewExam").classList.toggle("hidden", name !== "exam");
  if (name === "events") showPane("events");
  if (name === "chat") refreshChatProjects();
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
      <label>Знаки .pt/.onnx</label><div class="settings-row"><input data-k="traffic_sign_model" type="text" value="${esc(s.traffic_sign_model || "")}" /><button class="btn" type="button" data-import-model="traffic">Импорт</button></div>
      <label>Номера .pt/.onnx</label><div class="settings-row"><input data-k="plate_model" type="text" value="${esc(s.plate_model || "")}" /><button class="btn" type="button" data-import-model="plate">Импорт</button></div>
      <label>VehicleDINO .onnx</label><div class="settings-row"><input data-k="vehicle_dino_model" type="text" value="${esc(s.vehicle_dino_model || "")}" /><button class="btn" type="button" data-import-model="vehicle_dino">Импорт</button></div>
      <label>Ollama host</label><input data-k="ollama_host" type="text" value="${esc(s.ollama_host || "http://127.0.0.1:11434")}" />
      <label>Ollama model</label><input data-k="ollama_model" type="text" value="${esc(s.ollama_model || "")}" placeholder="qwen2.5:3b" />
      <label>Vision model</label><input data-k="vision_model" type="text" value="${esc(s.vision_model || "qwen2.5vl:3b")}" placeholder="qwen2.5vl:3b" />
      <label>Комментарий</label><select data-k="commentary_enabled"><option value="true">on</option><option value="false">off</option></select>
      <label>Голос</label><select data-k="voice_enabled"><option value="false">off</option><option value="true">on</option></select>
    </div>`;
  $(`[data-k=device]`).value = s.device || "auto";
  $(`[data-k=tracker]`).value = s.tracker || "bytetrack.yaml";
  $(`[data-k=commentary_enabled]`).value = String(s.commentary_enabled !== false);
  $(`[data-k=voice_enabled]`).value = String(!!s.voice_enabled);
  $$("[data-import-model]", $("#settingsForm")).forEach((btn) => btn.addEventListener("click", importModelFromSettings));
  $("#settingsOverlay").classList.add("open");
}

async function importModelFromSettings(e) {
  const kind = e.currentTarget.dataset.importModel;
  const res = await API.import_model_pack(kind);
  if (!res?.ok) return;
  const map = { traffic: "traffic_sign_model", plate: "plate_model", vehicle_dino: "vehicle_dino_model" };
  const key = map[kind];
  if (key) $(`[data-k=${key}]`).value = res.path || "";
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

async function refreshChatProjects() {
  if (!API || !API.list_chat_projects) return;
  const list = await API.list_chat_projects();
  const select = $("#chatProject");
  select.innerHTML = list.length
    ? list.map((p) => `<option value="${esc(p.id)}">${esc(p.title)} · ${fmt(p.duration || 0)}</option>`).join("")
    : `<option value="">Нет обработанных видео</option>`;
  if (!list.length && !$("#chatBody .msg")) {
    $("#chatBody").innerHTML = `<div class="chat-empty">Сначала обработайте видео, потом его можно будет обсуждать с моделью.</div>`;
  }
}

function addChatMessage(kind, text, meta = "") {
  const body = $("#chatBody");
  const empty = body.querySelector(".chat-empty");
  if (empty) empty.remove();
  const div = document.createElement("div");
  div.className = `msg ${kind}`;
  div.innerHTML = `${meta ? `<div class="meta">${esc(meta)}</div>` : ""}<div class="txt"></div>`;
  div.querySelector(".txt").textContent = text || "";
  body.appendChild(div);
  body.scrollTop = body.scrollHeight;
  return div;
}

async function sendChat() {
  const projectId = $("#chatProject").value;
  const question = $("#chatQuestion").value.trim();
  if (!question) return;
  $("#chatQuestion").value = "";
  const res = await API.chat_about_project(projectId, question);
  if (res && res.ok === false) {
    addChatMessage("ai", res.error || "Не удалось отправить вопрос", "Ошибка");
  }
}

async function pickExamImage() {
  const res = await API.pick_exam_image();
  if (!res) return;
  state.examImage = res;
  $("#examImage").src = res.image_url || "";
  $("#emptyExam").classList.add("hidden");
}

async function analyzeExamImage() {
  const question = $("#examQuestion").value.trim();
  if (!question) {
    $("#examAnswer").textContent = "Введите вопрос по изображению.";
    return;
  }
  const res = await API.analyze_exam_image(question);
  if (res && res.ok === false) {
    $("#examAnswer").textContent = res.error || "Не удалось начать анализ.";
  }
}

async function pullVisionModel() {
  const model = state.settings.vision_model || "qwen2.5vl:3b";
  $("#examAnswer").textContent = `Скачиваю vision-модель ${model} через Ollama...`;
  await API.pull_vision_model(model);
}

function renderExamFindings(findings) {
  const rows = [];
  for (const det of findings.detections || []) {
    rows.push(`<div class="item ${esc(det.kind || "")}"><b>${esc(det.label || det.kind)}</b><div class="meta">${esc(det.kind || "")} · ${esc(det.position || "")} · ${Math.round((det.confidence || 0) * 100)}%</div></div>`);
  }
  const ctx = findings.context;
  if (ctx?.explanation) {
    rows.unshift(`<div class="item sign"><b>Контекст</b><div>${esc(ctx.explanation)}</div><div class="meta">полоса: ${esc(ctx.lane?.lane || "unknown")} · применимость: ${esc(ctx.applies_to_ego_lane || "unknown")}</div></div>`);
  }
  if (findings.vision_error) {
    rows.push(`<div class="item"><b>CV предупреждение</b><div class="meta">${esc(findings.vision_error)}</div></div>`);
  }
  $("#examDetections").innerHTML = rows.length ? rows.join("") : `<div class="item">CV-детекции не найдены, ответ построен по vision-модели.</div>`;
}
