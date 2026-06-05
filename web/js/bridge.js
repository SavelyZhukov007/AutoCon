export function ready() {
  return new Promise((resolve) => {
    if (window.pywebview?.api) return resolve(window.pywebview.api);
    window.addEventListener("pywebviewready", () => resolve(window.pywebview.api), { once: true });
    window.setTimeout(() => {
      if (!window.pywebview?.api) resolve(mockApi());
    }, 250);
  });
}

export const Bus = {
  map: new Map(),
  on(event, fn) {
    const arr = this.map.get(event) || [];
    arr.push(fn);
    this.map.set(event, arr);
  },
  emit(event, payload) {
    (this.map.get(event) || []).forEach((fn) => fn(payload || {}));
  },
};

window.AutoConBus = Bus;

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function fmt(seconds) {
  seconds = Math.max(0, Number(seconds || 0));
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return [h, m, s].map((x) => String(x).padStart(2, "0")).join(":");
}

export function esc(text) {
  return String(text ?? "").replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

function mockApi() {
  return {
    async environment() {
      return {
        settings: { device: "auto", target_fps: 8, imgsz: 960, conf: 0.35, tracker: "bytetrack.yaml", commentary_enabled: true },
        first_run_done: true,
        ffmpeg: true,
        device: { selected_device: "preview", gpus: [] },
        packages: [{ key: "vision", title: "YOLO11", desc: "Preview", installed: true }],
        model_packs: [{ key: "yolo11s", title: "YOLO11s", desc: "Preview", installed: true }],
        ollama: false,
        ollama_model: "qwen2.5:3b",
        server_base: location.origin,
      };
    },
    async list_camera_devices() { return [{ index: 0, name: "Preview camera" }]; },
    async pick_video() { return null; },
    async process_video() { return { ok: false, error: "preview" }; },
    async export_report() { return { ok: false, error: "preview" }; },
    async start_camera() { return { ok: true }; },
    async stop_camera() { return { ok: true }; },
    async update_settings(patch) { return patch || {}; },
    async finish_first_run() { return true; },
    async install_packages() { return { ok: true }; },
    async install_model_pack() { return { ok: true }; },
  };
}
