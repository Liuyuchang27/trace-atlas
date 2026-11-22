import json
import sqlite3
import time
from pathlib import Path


class Archive:
    def __init__(self, filename):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(filename, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS research(id TEXT PRIMARY KEY, snapshot TEXT NOT NULL)")
        self.db.execute("""CREATE TABLE IF NOT EXISTS conversations(
            id TEXT PRIMARY KEY,title TEXT NOT NULL,created INTEGER NOT NULL,updated INTEGER NOT NULL,
            visible INTEGER NOT NULL,session TEXT NOT NULL)""")

    def save(self, run):
        self.db.execute("INSERT INTO research VALUES(?,?) ON CONFLICT(id) DO UPDATE SET snapshot=excluded.snapshot",
                        (run["id"], json.dumps(run, ensure_ascii=False)))

    def get(self, run_id):
        row = self.db.execute("SELECT snapshot FROM research WHERE id=?", (run_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def interrupt(self):
        for (raw,) in self.db.execute("SELECT snapshot FROM research WHERE json_extract(snapshot, '$.status') IN ('running','stopping')").fetchall():
            run = json.loads(raw)
            run.update(status="cancelled", error="Server restarted during this investigation.")
            settle(run)
            self.save(run)

    def store_chat(self, identifier, title, session, *, create, keep_time=False):
        now = time.time_ns() // 1_000_000
        visible = bool(session.get("current", {}).get("question") or session.get("rounds"))
        raw = json.dumps(session, ensure_ascii=False)
        if create:
            self.db.execute("INSERT INTO conversations VALUES(?,?,?,?,?,?)", (identifier, title, now, now, visible, raw))
            return True
        return bool(self.db.execute("""UPDATE conversations SET title=?, session=?, visible=?,
            updated=CASE WHEN ? THEN updated ELSE ? END WHERE id=?""",
            (title, raw, visible, keep_time, now, identifier)).rowcount)

    def chats(self, include_empty):
        return [dict(id=i, title=t, created_at=c, updated_at=u) for i,t,c,u in self.db.execute(
            "SELECT id,title,created,updated FROM conversations WHERE visible OR ? ORDER BY updated DESC,rowid DESC", (include_empty,))]

    def import_history(self, source):
        # Use a quiescent, checkpointed snapshot; immutable mode cannot create sidecars in its directory.
        original = sqlite3.connect(source.resolve().as_uri()+"?mode=ro&immutable=1", uri=True)
        imported = skipped = 0
        try:
            columns = {row[1] for row in original.execute("PRAGMA table_info(conversations)")}
            if not {"id","title","created_at","updated_at","data"} <= columns:
                raise ValueError("Expected a previous workbench conversations database")
            where = " WHERE deleted_at IS NULL" if "deleted_at" in columns else ""
            self.db.execute("BEGIN")
            for key,title,created,updated,raw in original.execute("SELECT id,title,created_at,updated_at,data FROM conversations"+where):
                session = json.loads(raw)
                if session.get("schema_version") != 2 or not isinstance(session.get("current"),dict):
                    skipped += 1
                    continue
                for entry in [session["current"],*session.get("rounds",[])]:
                    entry.pop("recovery",None)
                    if entry.get("status") == "running":
                        entry["status"] = "cancelled"
                    for task in entry.get("tasks",{}).values():
                        if task.get("status") in {"running","pending"}:
                            task["status"] = "cancelled"
                visible = bool(session["current"].get("question") or session.get("rounds"))
                imported += self.db.execute("INSERT OR IGNORE INTO conversations VALUES(?,?,?,?,?,?)",
                    (key,title,created,updated,visible,json.dumps(session,ensure_ascii=False))).rowcount
            self.db.execute("COMMIT")
        except BaseException:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise
        finally:
            original.close()
        return imported,skipped

    def chat(self, identifier):
        row = self.db.execute("SELECT title,created,updated,session FROM conversations WHERE id=?", (identifier,)).fetchone()
        if row:
            return dict(id=identifier, title=row[0], created_at=row[1], updated_at=row[2], data=json.loads(row[3]))


def settle(run):
    for task in run["view"]["tasks"].values():
        if task["status"] in {"pending", "running"}:
            task["status"] = "cancelled"


def recovery(run):
    return {"run_id": run["id"], "attempt": run["attempt"], "status": run["status"],
            "can_continue": run["status"] in {"failed", "cancelled"},
            "can_write_report": run["status"] in {"failed", "cancelled"} and bool(run["work"]["sources"]),
            "checkpoint_saved": bool(run["work"]["sources"]), "saved_sections": len(run["work"]["draft"])}


def public_snapshot(run):
    return {"recovery": recovery(run), "view": run["view"], "error": run.get("error"), "language": run["request"]["lang"]}
