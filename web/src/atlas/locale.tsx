import { createContext, useContext } from 'react';
import { translator } from './i18n';

export const LocaleContext = createContext(translator('zh-Hans'));
export const useLocale = () => useContext(LocaleContext);
