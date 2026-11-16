"""Local SQLite scan history (stored on this machine only)."""
import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from .config import DB_PATH

MAX_STORED_CHARS = 500   # never store whole emails

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    scanned_at   TEXT NOT NULL,
    scan_type    TEXT NOT NULL,      -- url | email | sms
    input_text   TEXT NOT NULL,
    label        TEXT NOT NULL,
    probability  REAL NOT NULL,
    risk_level   TEXT NOT NULL,
    indicators   TEXT NOT NULL       -- JSON list
);
"""


def _connect(db_path=None):
    db_path = Path(db_path or DB_PATH)   # resolved at call time so tests can patch DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def init_db(db_path=None) -> None:
    with closing(_connect(db_path)) as con, con:
        con.execute(_SCHEMA)


def log_scan(scan_type: str, result: dict, db_path=None) -> int:
    init_db(db_path)
    text = result["input"][:MAX_STORED_CHARS]
    with closing(_connect(db_path)) as con, con:
        cur = con.execute(
            "INSERT INTO scans (scanned_at, scan_type, input_text, label, probability, risk_level, indicators) "
            "VALUES (?,?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), scan_type, text, result["label"],
             float(result["probability"]), result["risk_level"],
             json.dumps(result.get("indicators", []), ensure_ascii=False)))
        return cur.lastrowid


def get_history(limit: int = 200, db_path=None) -> list:
    init_db(db_path)
    with closing(_connect(db_path)) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM scans ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["indicators"] = json.loads(d["indicators"])
        out.append(d)
    return out


def get_stats(db_path=None) -> dict:
    init_db(db_path)
    with closing(_connect(db_path)) as con:
        total = con.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
        flagged = con.execute("SELECT COUNT(*) FROM scans WHERE probability >= 0.5").fetchone()[0]
        by_type = dict(con.execute("SELECT scan_type, COUNT(*) FROM scans GROUP BY scan_type").fetchall())
        by_risk = dict(con.execute("SELECT risk_level, COUNT(*) FROM scans GROUP BY risk_level").fetchall())
    return {"total": total, "flagged": flagged, "by_type": by_type, "by_risk": by_risk}


def clear_history(db_path=None) -> None:
    init_db(db_path)
    with closing(_connect(db_path)) as con, con:
        con.execute("DELETE FROM scans")
