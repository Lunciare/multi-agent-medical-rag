import sqlite3
import random
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "validation.db"

ARMS = ("rag", "vanilla")

PREFERENCE_VALUES = ("opt1_strong", "opt1_weak", "tie", "opt2_weak", "opt2_strong")
ROUTING_VALUES = ("correct", "acceptable_incomplete", "incorrect")


def connect(db_path=DEFAULT_DB_PATH):
    parent = Path(db_path).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    init_schema(conn)
    return conn


def init_schema(conn):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS config (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS item_order (
            rater_id    TEXT NOT NULL,
            position    INTEGER NOT NULL,
            item_id     TEXT NOT NULL,
            assigned_ts REAL NOT NULL,
            PRIMARY KEY (rater_id, item_id)
        );

        CREATE TABLE IF NOT EXISTS position_map (
            rater_id    TEXT NOT NULL,
            item_id     TEXT NOT NULL,
            option_1_arm TEXT NOT NULL CHECK (option_1_arm IN ('rag','vanilla')),
            option_2_arm TEXT NOT NULL CHECK (option_2_arm IN ('rag','vanilla')),
            assigned_ts  REAL NOT NULL,
            PRIMARY KEY (rater_id, item_id)
        );

        CREATE TABLE IF NOT EXISTS submission (
            rater_id         TEXT NOT NULL,
            item_id          TEXT NOT NULL,
            preference       TEXT NOT NULL,
            safety_flag_opt1 INTEGER NOT NULL CHECK (safety_flag_opt1 IN (0,1)),
            safety_flag_opt2 INTEGER NOT NULL CHECK (safety_flag_opt2 IN (0,1)),
            routing_judgment TEXT NOT NULL,
            client_ts        TEXT,
            server_ts        REAL NOT NULL,
            PRIMARY KEY (rater_id, item_id)
        );
        """
    )
    conn.commit()


def get_config(conn, key, default=None):
    row = conn.execute("SELECT value FROM config WHERE key = ?", (key,)).fetchone()
    return row["value"] if row is not None else default


def set_config(conn, key, value):
    conn.execute(
        "INSERT INTO config (key, value) VALUES (?, ?) "
        "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )
    conn.commit()


def get_or_assign_order(conn, rater_id, all_item_ids, now_ts, rng=random):
    rows = conn.execute(
        "SELECT item_id FROM item_order WHERE rater_id = ? ORDER BY position",
        (rater_id,),
    ).fetchall()
    valid = set(all_item_ids)

    if rows:
        ordered = [r["item_id"] for r in rows if r["item_id"] in valid]
        known = set(ordered)
        extras = [iid for iid in all_item_ids if iid not in known]
        if extras:
            start = len(rows)
            for off, iid in enumerate(extras):
                conn.execute(
                    "INSERT INTO item_order (rater_id, position, item_id, assigned_ts) "
                    "VALUES (?, ?, ?, ?)",
                    (rater_id, start + off, iid, now_ts),
                )
            conn.commit()
            ordered += extras
        return ordered

    order = list(all_item_ids)
    rng.shuffle(order)
    for pos, iid in enumerate(order):
        conn.execute(
            "INSERT INTO item_order (rater_id, position, item_id, assigned_ts) "
            "VALUES (?, ?, ?, ?)",
            (rater_id, pos, iid, now_ts),
        )
    conn.commit()
    return order


def get_or_assign_position(conn, rater_id, item_id, now_ts, rng=random):
    row = conn.execute(
        "SELECT option_1_arm, option_2_arm FROM position_map "
        "WHERE rater_id = ? AND item_id = ?",
        (rater_id, item_id),
    ).fetchone()
    if row is not None:
        return {"option_1_arm": row["option_1_arm"], "option_2_arm": row["option_2_arm"]}

    # First time: flip a coin for which arm lands in option_1.
    opt1, opt2 = ("rag", "vanilla") if rng.random() < 0.5 else ("vanilla", "rag")
    conn.execute(
        "INSERT INTO position_map (rater_id, item_id, option_1_arm, option_2_arm, assigned_ts) "
        "VALUES (?, ?, ?, ?, ?)",
        (rater_id, item_id, opt1, opt2, now_ts),
    )
    conn.commit()
    return {"option_1_arm": opt1, "option_2_arm": opt2}


def get_position(conn, rater_id, item_id):
    """Return the stored map for (rater, item), or None if not yet assigned."""
    row = conn.execute(
        "SELECT option_1_arm, option_2_arm FROM position_map "
        "WHERE rater_id = ? AND item_id = ?",
        (rater_id, item_id),
    ).fetchone()
    if row is None:
        return None
    return {"option_1_arm": row["option_1_arm"], "option_2_arm": row["option_2_arm"]}


def is_done(conn, rater_id, item_id):
    row = conn.execute(
        "SELECT 1 FROM submission WHERE rater_id = ? AND item_id = ?",
        (rater_id, item_id),
    ).fetchone()
    return row is not None


def upsert_submission(conn, *, rater_id, item_id, preference, safety_flag_opt1,
                      safety_flag_opt2, routing_judgment, client_ts, server_ts):
    """Insert or overwrite the raw judgment for (rater, item). No duplicates."""
    conn.execute(
        """
        INSERT INTO submission (
            rater_id, item_id, preference, safety_flag_opt1, safety_flag_opt2,
            routing_judgment, client_ts, server_ts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (rater_id, item_id) DO UPDATE SET
            preference       = excluded.preference,
            safety_flag_opt1 = excluded.safety_flag_opt1,
            safety_flag_opt2 = excluded.safety_flag_opt2,
            routing_judgment = excluded.routing_judgment,
            client_ts        = excluded.client_ts,
            server_ts        = excluded.server_ts
        """,
        (rater_id, item_id, preference, int(bool(safety_flag_opt1)),
         int(bool(safety_flag_opt2)), routing_judgment, client_ts, server_ts),
    )
    conn.commit()


def completed_count_by_item(conn):
    """Map item_id -> number of distinct raters who submitted it."""
    rows = conn.execute(
        "SELECT item_id, COUNT(DISTINCT rater_id) AS n FROM submission GROUP BY item_id"
    ).fetchall()
    return {r["item_id"]: r["n"] for r in rows}


def all_submissions(conn):
    """All raw submission rows, ordered, for export."""
    return conn.execute(
        "SELECT * FROM submission ORDER BY rater_id, item_id"
    ).fetchall()
