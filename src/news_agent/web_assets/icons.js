/* Local SVG vocabulary. No external fonts, emoji rendering, or network dependency. */
(() => {
  const paths = {
    flame: '<path fill="#d56b3c" stroke="none" d="M12.3 1.7c.6 4.4 5.9 5.3 5.1 10.1 1.1-.5 1.9-1.7 2.2-2.8 1.2 1.8 1.9 3.8 1.4 6.5-.7 4.2-4 7-8.6 7-4.9 0-8.5-3.1-8.5-7.5 0-3.4 1.8-5.7 4.3-8.2-.3 2.1.2 3.4 1.1 4.1 1.9-2.4 3.6-5.4 3-9.2Z"/><path fill="#ffe2a4" stroke="none" d="M12.5 12.1c.1 2.4 3.5 2.7 3.5 5.4 0 2-1.5 3.5-3.7 3.5-2.3 0-3.9-1.5-3.9-3.6 0-1.6.8-2.9 2.2-4.2-.1 1.2.3 1.8.9 2.2.6-.9 1-2 1-3.3Z"/>',
    search: '<circle cx="10.7" cy="10.7" r="6.7"/><path d="m16 16 4.4 4.4"/>',
    read: '<path d="M4 5.5c3-1 5.7-.7 8 1.1 2.3-1.8 5-2.1 8-1.1v14c-3-1-5.7-.7-8 1.1-2.3-1.8-5-2.1-8-1.1Z M12 6.6v14"/>',
    plan: '<rect x="4" y="4" width="16" height="17" rx="3"/><path d="M9 3h6v4H9z M8 12h8 M8 16h5"/>',
    timeline: '<path d="M6 3v18 M10 6h10 M10 12h7 M10 18h10"/><circle cx="6" cy="6" r="1.5" fill="currentColor"/><circle cx="6" cy="12" r="1.5" fill="currentColor"/><circle cx="6" cy="18" r="1.5" fill="currentColor"/>',
    verify: '<path d="m12 3 8 3v6c0 4.7-3.7 7.7-8 10-4.3-2.3-8-5.3-8-10V6Z m-4 9 3 3 5-6"/>',
    source: '<path d="m9 15 6-6 M8 16l-1 1a4.2 4.2 0 0 1-6-6l4-4a4.2 4.2 0 0 1 6 0 M16 8l1-1a4.2 4.2 0 0 1 6 6l-4 4a4.2 4.2 0 0 1-6 0" transform="translate(1 0) scale(.92)"/>',
    branch: '<path d="M6 3v12a4 4 0 0 0 4 4h8 M6 9h7a4 4 0 0 0 4-4V3 M15 16l3 3-3 3"/><circle cx="6" cy="3" r="1.6" fill="currentColor"/>',
    report: '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9Z M14 3v6h6 M8 13h8 M8 17h5"/>',
    arrow: '<path d="M5 12h14 m-5-5 5 5-5 5"/>',
    external: '<path d="M14 4h6v6 M20 4 10 14 M10 5H5a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-5"/>',
    play: '<path d="m8 4 12 8-12 8Z" fill="currentColor" stroke="none"/>',
    pause: '<path d="M7 5v14 M17 5v14" stroke-width="3.2"/>',
    restart: '<path d="M4 10a8.2 8.2 0 1 1 1 8 M4 4v6h6"/>',
    plus: '<path d="M12 5v14 M5 12h14"/>',
    close: '<path d="m6 6 12 12 M18 6 6 18"/>',
    help: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.6 2.6 0 1 1 4.6 1.6c-1.2 1-2.1 1-2.1 3 M12 17h.01"/>',
    download: '<path d="M12 3v12 m-4-4 4 4 4-4 M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/>',
    point: '<circle cx="12" cy="12" r="4" fill="currentColor" stroke="none"/>',
    unknown: '<path d="m12 2 10 10-10 10L2 12Z M10 9a2 2 0 1 1 3.4 1.5c-.9.7-1.4.9-1.4 2.5 M12 16h.01"/>',
    stop: '<rect x="5" y="5" width="14" height="14" rx="3"/>',
    fit: '<path d="M4 9V4h5 M15 4h5v5 M20 15v5h-5 M9 20H4v-5"/>',
  };
  const svg = name => '<svg class="atlas-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">' + (paths[name] || paths.point) + '</svg>';
  const hydrate = (root=document) => root.querySelectorAll('[data-icon]').forEach(el => { el.innerHTML=svg(el.dataset.icon); });
  window.AtlasIcons = {svg, hydrate};
  document.addEventListener('DOMContentLoaded', () => hydrate());
})();
