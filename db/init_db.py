"""
db/init_db.py
SQLite persistence layer for BKK Flow Predictor inference runs.

Schema
------
inference_runs   — one row per prediction (metadata + summary stats)
zone_results     — per-zone delta_net values for each run (FK → inference_runs)

Usage
-----
    from db.init_db import get_db, save_run, load_runs, load_zone_results

    db = get_db()                       # opens / creates the DB
    run_id = save_run(db, meta, delta_od, zone_ids)
    df     = load_runs(db)              # all runs as DataFrame
    zones  = load_zone_results(db, run_id)
"""

import sqlite3
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Default DB path — sits next to this file (db/inference.db)
DEFAULT_DB_PATH = Path(__file__).resolve().parent / "inference.db"


# ─────────────────────────────────────────────
# Connection
# ─────────────────────────────────────────────

def get_db(path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Open (or create) the SQLite database and ensure the schema exists.
    Returns an open connection. Caller is responsible for closing it,
    or use it as a context manager.
    """
    db_path = Path(path) if path else DEFAULT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row  # dict-like rows
    conn.execute("PRAGMA journal_mode=WAL")  # safe concurrent reads
    conn.execute("PRAGMA foreign_keys=ON")

    _create_schema(conn)
    return conn


# ─────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────

def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS inference_runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       REAL    NOT NULL,           -- Unix epoch (float)
            scenario_name   TEXT    NOT NULL DEFAULT '',
            model_type      TEXT    NOT NULL,           -- 'GAT+LSTM' | 'Hypergraph+LSTM'
            n_zones         INTEGER NOT NULL,
            inference_ms    REAL,                       -- wall-clock inference time
            total_gain      REAL,                       -- sum of positive delta_od entries
            total_loss      REAL,                       -- sum of negative delta_od entries
            n_gaining_zones INTEGER,                    -- zones with delta_net > 0.5
            n_losing_zones  INTEGER,                    -- zones with delta_net < -0.5
            max_gain        REAL,
            max_loss        REAL,
            checkpoint_path TEXT    DEFAULT '',
            notes           TEXT    DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS zone_results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      INTEGER NOT NULL
                            REFERENCES inference_runs(id) ON DELETE CASCADE,
            zone_id     INTEGER NOT NULL,
            delta_net   REAL    NOT NULL,
            delta_inflow  REAL  NOT NULL DEFAULT 0,
            delta_outflow REAL  NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_zone_results_run
            ON zone_results(run_id);
    """)
    conn.commit()


# ─────────────────────────────────────────────
# Write
# ─────────────────────────────────────────────

def save_run(
    conn: sqlite3.Connection,
    model_type: str,
    delta_od: np.ndarray,
    zone_ids: np.ndarray,
    scenario_name: str = "",
    inference_ms: Optional[float] = None,
    checkpoint_path: str = "",
    notes: str = "",
) -> int:
    """
    Persist one inference run.

    Parameters
    ----------
    conn            : open DB connection (from get_db())
    model_type      : 'GAT+LSTM' or 'Hypergraph+LSTM'
    delta_od        : (N, N) predicted ΔOD matrix
    zone_ids        : (N,) zone ID array
    scenario_name   : free-text label (e.g. 'M2 extension')
    inference_ms    : wall-clock time in milliseconds
    checkpoint_path : path to the model checkpoint used
    notes           : any extra free-text notes

    Returns
    -------
    int : the new run's primary key (id)
    """
    delta_inflow  = delta_od.sum(axis=0)   # col sums — inflow change per zone
    delta_outflow = delta_od.sum(axis=1)   # row sums — outflow change per zone
    delta_net     = delta_inflow - delta_outflow

    summary = {
        "total_gain":      float(delta_od[delta_od > 0].sum()),
        "total_loss":      float(delta_od[delta_od < 0].sum()),
        "n_gaining_zones": int((delta_net > 0.5).sum()),
        "n_losing_zones":  int((delta_net < -0.5).sum()),
        "max_gain":        float(delta_net.max()),
        "max_loss":        float(delta_net.min()),
    }

    cur = conn.execute(
        """
        INSERT INTO inference_runs
            (timestamp, scenario_name, model_type, n_zones, inference_ms,
             total_gain, total_loss, n_gaining_zones, n_losing_zones,
             max_gain, max_loss, checkpoint_path, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            time.time(),
            scenario_name,
            model_type,
            int(delta_od.shape[0]),
            inference_ms,
            summary["total_gain"],
            summary["total_loss"],
            summary["n_gaining_zones"],
            summary["n_losing_zones"],
            summary["max_gain"],
            summary["max_loss"],
            checkpoint_path,
            notes,
        ),
    )
    run_id = cur.lastrowid

    # Bulk-insert per-zone results
    rows = [
        (run_id, int(z), float(delta_net[i]),
         float(delta_inflow[i]), float(delta_outflow[i]))
        for i, z in enumerate(zone_ids)
    ]
    conn.executemany(
        "INSERT INTO zone_results (run_id, zone_id, delta_net, delta_inflow, delta_outflow) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return run_id


# ─────────────────────────────────────────────
# Read
# ─────────────────────────────────────────────

def load_runs(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return all inference runs as a DataFrame, newest first."""
    df = pd.read_sql_query(
        "SELECT * FROM inference_runs ORDER BY timestamp DESC",
        conn,
    )
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s") \
                           .dt.tz_localize("UTC") \
                           .dt.tz_convert("Europe/Budapest") \
                           .dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def load_zone_results(conn: sqlite3.Connection, run_id: int) -> pd.DataFrame:
    """Return per-zone results for a single run."""
    return pd.read_sql_query(
        "SELECT zone_id, delta_net, delta_inflow, delta_outflow "
        "FROM zone_results WHERE run_id = ? ORDER BY zone_id",
        conn,
        params=(run_id,),
    )


def delete_run(conn: sqlite3.Connection, run_id: int) -> None:
    """Delete a run and its zone results (CASCADE handles zone_results)."""
    conn.execute("DELETE FROM inference_runs WHERE id = ?", (run_id,))
    conn.commit()
