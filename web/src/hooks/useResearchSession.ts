import {useEffect, useRef, useState} from 'react';
import {advance, blank, restoreSavedResearch, sessionOf} from '../research/state';
import type {ResearchSession} from '../research/state';
import {readSession} from '../research/history';

type Item = {id:string; title:string; updatedAt:number};
type Action = 'continue'|'write_report';
const chats = '/api/conversations';
const runs = '/api/research/';
const currentKey = 'trace_atlas_current_conversation';

async function jsonRequest(path:string, method='GET', body?:unknown) {
  const response = await fetch(path,{method,headers:body ? {'Content-Type':'application/json'} : undefined,
    body:body ? JSON.stringify(body) : undefined});
  if (!response.ok) throw new Error('HTTP '+response.status+' · '+path);
  return response.json();
}

async function consume(response:Response, accept:(frame:unknown)=>void) {
  if (!response.ok || !response.body || !response.headers.get('content-type')?.includes('text/event-stream'))
    throw new Error('Research stream unavailable (HTTP '+response.status+')');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const {done,value} = await reader.read();
      buffer += decoder.decode(value,{stream:!done});
      let boundary;
      while ((boundary = /\r?\n\r?\n/.exec(buffer))) {
        const packet = buffer.slice(0,boundary.index);
        buffer = buffer.slice(boundary.index + boundary[0].length);
        const payload = packet.split(/\r?\n/).filter(l=>l.startsWith('data:')).map(l=>l.slice(5).trimStart()).join('\n');
        if (payload) accept(JSON.parse(payload));
      }
      if (done) return;
    }
  } finally { reader.releaseLock(); }
}

export default function useResearchSession() {
  const [session,setSession] = useState(()=>sessionOf());
  const [conversations,setItems] = useState<Item[]>([]);
  const [currentId,setCurrentId] = useState<string|null>(null);
  const [saveError,setError] = useState('');
  const [isLoading,setLoading] = useState(true);
  const live = useRef(session);
  const selected = useRef<string|null>(null);
  const serial = useRef(0);
  const controller = useRef<AbortController>();
  const pendingWrites = useRef(new Map<string,Promise<void>>());

  function display(next:ResearchSession) {
    live.current = next;
    setSession(next);
  }
  function selectId(id:string|null) {
    selected.current = id;
    setCurrentId(id);
    if (id) localStorage.setItem(currentKey,id);
    else localStorage.removeItem(currentKey);
  }
  function persist(id:string, data:ResearchSession) {
    const title = (data.rounds[0]?.question || data.current.question).slice(0,50);
    try { sessionStorage.setItem('trace_atlas_session_v2',JSON.stringify(data)); } catch { /* Server retains long reports. */ }
    const promise = (pendingWrites.current.get(id) || Promise.resolve()).then(async()=>{
      await jsonRequest(chats+'/'+encodeURIComponent(id),'PUT',{title,data});
      setItems(items=>[{id,title,updatedAt:Date.now()},...items.filter(i=>i.id!==id)]);
      setError('');
    }).catch(e=>setError(String(e)));
    pendingWrites.current.set(id,promise);
    void promise.then(()=>{if (pendingWrites.current.get(id)===promise) pendingWrites.current.delete(id);});
    return promise;
  }

  async function refreshRun(state:ResearchSession) {
    const id = state.current.recovery?.run_id;
    if (!id) return state;
    const res = await fetch(runs+encodeURIComponent(id));
    // Previous workspaces can have archived results whose execution record lives elsewhere.
    if (res.status===404 && state.current.status!=='running') return state;
    if (!res.ok) throw new Error('Cannot read saved investigation ('+res.status+')');
    return sessionOf(restoreSavedResearch(state.current,await res.json()),state.rounds);
  }
  async function watch(token:number,id:string) {
    while (token === serial.current && live.current.current.status==='running') {
      await new Promise(resolve=>setTimeout(resolve,1000));
      if (token!==serial.current) return;
      const refreshed = await refreshRun(live.current);
      if (token!==serial.current) return;
      display(refreshed);
      if (refreshed.current.status!=='running') await persist(id,refreshed);
    }
  }
  async function selectConversation(id:string) {
    const token = ++serial.current;
    try {
      await pendingWrites.current.get(id);
      const stored = await jsonRequest(chats+'/'+encodeURIComponent(id));
      const updated = await refreshRun(readSession(stored.data));
      if (token!==serial.current) return;
      selectId(id); display(updated);
      if (updated.current.status==='running') await watch(token,id);
    } catch(e) { if (token===serial.current) setError(String(e)); }
  }
  useEffect(()=>{
    const token = serial.current;
    void (async()=>{
      try {
        const rows = await jsonRequest(chats+'?include_empty=false');
        if (token!==serial.current) return;
        const items:Item[] = rows.map((r:any)=>({id:r.id,title:r.title,updatedAt:r.updated_at}));
        setItems(items);
        const desired = localStorage.getItem(currentKey);
        const existing = items.find(i=>i.id===desired) || items[0];
        if (existing) void selectConversation(existing.id);
      } catch(e) { setError(String(e)); }
      finally { setLoading(false); }
    })();
    return ()=>{serial.current++;controller.current?.abort();};
  },[]);

  function cancel() {
    const run = live.current.current.recovery?.run_id;
    if (run && live.current.current.status==='running')
      void jsonRequest(runs+run+'/stop','POST').catch(e=>setError(String(e)));
    else if (live.current.current.status==='running') {
      // The request can already be executing. Its header handler will stop it once the id arrives.
      serial.current++;
      display(sessionOf({...live.current.current,status:'cancelled'},live.current.rounds));
    }
  }
  function newConversation() {
    cancel();
    serial.current++;
    if (live.current.current.recovery?.run_id) controller.current?.abort();
    controller.current=undefined;
    selectId(null);
    display(sessionOf());
    sessionStorage.removeItem('trace_atlas_session_v2');
  }
  async function start(text:string,action?:Action,timeline=false,lang?:string) {
    const old = live.current;
    if (!text.trim() || old.current.status==='running') return;
    const token = ++serial.current;
    const abort = new AbortController();
    controller.current=abort;
    const rounds = !action && old.current.question ? [...old.rounds,old.current] : old.rounds;
    const current = action ? {...old.current,status:'running' as const,error:undefined} : blank(text);
    display(sessionOf(current,rounds));
    let id=selected.current;
    let runId:string|null = null;
    try {
      if (!id) {
        id = crypto.randomUUID();
        await jsonRequest(chats,'POST',{id,title:text.slice(0,50),data:live.current});
        if (token!==serial.current) return;
        selectId(id);
      }
      const previous=old.current.recovery?.run_id;
      const response=await fetch('/solve',{method:'POST',signal:abort.signal,headers:{'Content-Type':'application/json'},
        body:JSON.stringify({inputs:text,lang:lang || old.current.recovery?.language || localStorage.getItem('trace_atlas_lang') || 'zh-Hans',
          timeline, ...(action ? {resume_run_id:previous,recovery_action:action} : previous ? {previous_run_id:previous} :
            old.current.article ? {previous_context:{question:old.current.question,report:old.current.article}} : {})})});
      runId=response.headers.get('X-Research-Run-ID');
      if (token!==serial.current) {
        if (runId) await jsonRequest(runs+runId+'/stop','POST');
        return;
      }
      if (runId) {
        current.recovery={run_id:runId,status:'running',can_continue:false,can_write_report:false,language:lang};
        display(sessionOf({...current},rounds));
        void persist(id,live.current);
      }
      await consume(response,frame=>{
        if (token===serial.current) display(sessionOf(advance(live.current.current,frame),rounds));
      });
      if (token===serial.current && live.current.current.status==='running') throw new Error('Stream interrupted');
    } catch(e) {
      if (token!==serial.current) return;
      try {
        if (!runId || !id) throw e;
        const refreshed = await refreshRun(live.current);
        if (token!==serial.current) return;
        display(refreshed);
        if (refreshed.current.status==='running') await watch(token,id);
      } catch(problem) {
        if (token===serial.current) display(sessionOf({...live.current.current,status:'failed',error:String(problem)},rounds));
      }
    } finally {
      if (token===serial.current) {
        controller.current=undefined;
        if (id) await persist(id,live.current);
      }
    }
  }
  function recover(action:Action) { return start(live.current.current.question,action); }
  function retry() { return start(live.current.current.question,live.current.current.recovery?.run_id ? 'continue':undefined); }
  return {session,conversations,currentId,isLoading,saveError,start,cancel,newConversation,selectConversation,recover,retry};
}
