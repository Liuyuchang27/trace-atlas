/** Migrate browser saves across the package namespace change. */
export function migrateBrowserStorage() {
  for (const [storage, names] of [
    [localStorage, ['lang', 'current_conversation', 'create_new']],
    [sessionStorage, ['session_v2', 'formatted', 'qaList', 'adjList', 'stashedNodes']],
  ] as const) {
    for (const name of names) {
      try {
        const current = `trace_atlas_${name}`;
        if (storage.getItem(current) === null) {
          const saved = storage.getItem(`veritrace_${name}`);
          if (saved !== null) storage.setItem(current, saved);
        }
      } catch {
        // Restricted browser storage does not prevent opening the workbench.
      }
    }
  }
}
