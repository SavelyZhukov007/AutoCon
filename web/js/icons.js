export const ICONS = {
  video: `<svg viewBox="0 0 24 24" fill="none"><path d="M4 6h11a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2Z" stroke="currentColor" stroke-width="2"/><path d="m17 10 5-3v10l-5-3" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>`,
  camera: `<svg viewBox="0 0 24 24" fill="none"><path d="M4 8h3l2-3h6l2 3h3v11H4V8Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><circle cx="12" cy="14" r="4" stroke="currentColor" stroke-width="2"/></svg>`,
  play: `<svg viewBox="0 0 24 24" fill="none"><path d="M8 5v14l11-7L8 5Z" fill="currentColor"/></svg>`,
  pause: `<svg viewBox="0 0 24 24" fill="none"><path d="M7 5h4v14H7V5Zm6 0h4v14h-4V5Z" fill="currentColor"/></svg>`,
  open: `<svg viewBox="0 0 24 24" fill="none"><path d="M4 20V5h6l2 3h8v12H4Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M8 14h8m-4-4v8" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>`,
  scan: `<svg viewBox="0 0 24 24" fill="none"><path d="M4 8V4h4M16 4h4v4M20 16v4h-4M8 20H4v-4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M5 12h14" stroke="currentColor" stroke-width="2"/></svg>`,
  sign: `<svg viewBox="0 0 24 24" fill="none"><path d="M12 3 21 12l-9 9-9-9 9-9Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M12 8v5m0 3h.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>`,
  car: `<svg viewBox="0 0 24 24" fill="none"><path d="m5 13 2-5h10l2 5v5H5v-5Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M7 18v2m10-2v2M6 13h12" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>`,
  plate: `<svg viewBox="0 0 24 24" fill="none"><rect x="3" y="7" width="18" height="10" rx="2" stroke="currentColor" stroke-width="2"/><path d="M7 12h3m4 0h3" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>`,
  bot: `<svg viewBox="0 0 24 24" fill="none"><rect x="5" y="8" width="14" height="10" rx="3" stroke="currentColor" stroke-width="2"/><path d="M12 8V4m-4 9h.01M16 13h.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>`,
  chat: `<svg viewBox="0 0 24 24" fill="none"><path d="M4 5h16v11H8l-4 4V5Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M8 9h8M8 13h5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>`,
  chip: `<svg viewBox="0 0 24 24" fill="none"><rect x="7" y="7" width="10" height="10" rx="2" stroke="currentColor" stroke-width="2"/><path d="M4 9h3M4 15h3m10-6h3m-3 6h3M9 4v3m6-3v3M9 17v3m6-3v3" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>`,
  download: `<svg viewBox="0 0 24 24" fill="none"><path d="M12 4v10m0 0 4-4m-4 4-4-4M5 20h14" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
  settings: `<svg viewBox="0 0 24 24" fill="none"><path d="M12 8a4 4 0 1 1 0 8 4 4 0 0 1 0-8Z" stroke="currentColor" stroke-width="2"/><path d="M4 12h2m12 0h2M12 4v2m0 12v2M6.3 6.3l1.4 1.4m8.6 8.6 1.4 1.4m0-11.4-1.4 1.4m-8.6 8.6-1.4 1.4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>`,
  export: `<svg viewBox="0 0 24 24" fill="none"><path d="M12 3v12m0-12 4 4m-4-4-4 4" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M5 13v7h14v-7" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>`,
  close: `<svg viewBox="0 0 24 24" fill="none"><path d="m6 6 12 12M18 6 6 18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>`,
  check: `<svg viewBox="0 0 24 24" fill="none"><path d="m5 12 5 5L20 7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
};

export function mountIcons(root = document) {
  root.querySelectorAll("[data-icon]").forEach((el) => {
    el.innerHTML = ICONS[el.dataset.icon] || "";
  });
}
