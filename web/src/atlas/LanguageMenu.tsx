import { KeyboardEvent, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { Locale, locales } from './i18n';
import { useLocale } from './locale';

const languages = Object.keys(locales) as Locale[];

export function LanguageMenu({ locale, onChange }: { locale: Locale; onChange: (value: Locale) => void }) {
  const t = useLocale();
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const items = useRef<(HTMLButtonElement | null)[]>([]);

  useLayoutEffect(() => {
    if (open) items.current[languages.indexOf(locale)]?.focus({ preventScroll: true });
  }, [open, locale]);

  useEffect(() => {
    if (!open) return;
    function onOutside(event: PointerEvent) {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    }
    document.addEventListener('pointerdown', onOutside);
    return () => document.removeEventListener('pointerdown', onOutside);
  }, [open]);

  function dismiss() {
    setOpen(false);
    trigger.current?.focus({ preventScroll: true });
  }

  function navigate(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    const last = languages.length - 1;
    const next = { ArrowDown: (index + 1) % languages.length, ArrowUp: (index + last) % languages.length,
      Home: 0, End: last }[event.key];
    if (next !== undefined) {
      event.preventDefault();
      items.current[next]?.focus();
    } else if (event.key === 'Escape') {
      event.preventDefault();
      dismiss();
    } else if (event.key === 'Tab') {
      // Continue native tab navigation from the trigger in either direction.
      dismiss();
    }
  }

  return <div className="locale-control" ref={root} onBlur={event => {
    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setOpen(false);
  }}>
    <button type="button" className="locale-trigger" ref={trigger} aria-haspopup="menu" aria-expanded={open}
      aria-controls="language-options" aria-label={t('語言') + ' · ' + locales[locale]} onClick={() => setOpen(!open)}
      onKeyDown={event => {
        if (event.key === 'ArrowDown' || event.key === 'ArrowUp') { event.preventDefault(); setOpen(true); }
      }}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8.7"/><ellipse cx="12" cy="12" rx="3.9" ry="8.7"/><path d="M3.5 12h17"/></svg>
      <span lang={locale}>{locales[locale]}</span>
      <svg className="locale-caret" viewBox="0 0 16 16" aria-hidden="true"><path d="m4.5 6.3 3.5 3.5 3.5-3.5"/></svg>
    </button>
    {open && <div id="language-options" className="locale-menu" role="menu" aria-label={t('語言')}>
      {languages.map((code, index) => <button key={code} ref={item => { items.current[index] = item; }}
        lang={code} type="button" role="menuitemradio" aria-checked={locale === code} tabIndex={-1}
        onKeyDown={event => navigate(event, index)} onClick={() => { onChange(code); dismiss(); }}>
        {locales[code]}
        {locale === code && <svg viewBox="0 0 20 20" aria-hidden="true"><path d="m4.5 10 3.6 3.6 7.4-7.4"/></svg>}
      </button>)}
    </div>}
  </div>;
}
