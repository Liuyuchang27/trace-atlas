export function migrateBrowserStorage() {
  // The workbench shares its existing v2 storage keys across compatible releases.
  try {
    if (!localStorage.getItem('trace_atlas_lang')) {
      const old = localStorage.getItem('veritrace_lang');
      if (old) localStorage.setItem('trace_atlas_lang',old);
    }
  } catch { /* A storage-restricted browser can use server history. */ }
}
