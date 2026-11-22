import {blank,sessionOf} from './state';
import type {ResearchSession,ResearchState} from './state';

export function readSession(value:any):ResearchSession {
  if (value?.schema_version === 2 && value.current)
    return sessionOf({...blank(),...value.current},(value.rounds || []).map((r:ResearchState)=>({...blank(),...r})));
  return sessionOf();
}
