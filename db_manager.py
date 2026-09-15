#!/usr/bin/env python3
"""
Database Manager per l'Agente di Trading.
Gestisce:
- Previsioni emesse (con sub-score per ogni indicatore)
- Verifica a posteriori (dopo l'orizzonte temporale)
- Pesi dinamici per categoria (Portafoglio / Osservati / ETF)
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple


# ============================================================================
# CONFIGURAZIONE
# ============================================================================
DB_PATH = "trading_history.db"
ORIZZONTE_VERIFICA_GIORNI = 3
SOGLIA_MOVIMENTO_PCT = 0.5   # per classificare BULLISH/BEARISH/NEUTRAL


# ============================================================================
# CONNESSIONE
# ============================================================================
def _get_connection() -> sqlite3.Connection:
    """Ritorna una connessione SQLite con row_factory impostato."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Crea le tabelle se non esistono. Rimuove file corrotti."""
    # Verifica se il file esiste ma non è un DB valido
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
            conn.close()
        except sqlite3.DatabaseError:
            print(f"⚠️ {DB_PATH} corrotto o non valido. Lo rimuovo.")
            try:
                os.remove(DB_PATH)
            except Exception as e:
                print(f"❌ Impossibile rimuovere {DB_PATH}: {e}")
                raise
    conn = _get_connection()
    try:
        cur = conn.cursor()

        # Tabella previsioni emesse
        cur.execute("""
            CREATE TABLE IF NOT EXISTS previsioni (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                categoria TEXT NOT NULL,
                timestamp_emissione TEXT NOT NULL,
                prezzo_emissione REAL NOT NULL,
                score REAL NOT NULL,
                direzione TEXT NOT NULL,
                sub_scores_json TEXT NOT NULL,
                orizzonte_giorni INTEGER NOT NULL,
                timestamp_scadenza TEXT NOT NULL,
                verificato INTEGER NOT NULL DEFAULT 0,
                prezzo_verifica REAL,
                rendimento_pct REAL,
                esito TEXT,
                esito_corretto INTEGER
            )
        """)

        # Tabella pesi per categoria
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pesi (
                categoria TEXT NOT NULL,
                indicatore TEXT NOT NULL,
                peso REAL NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (categoria, indicatore)
            )
        """)

        # Indici per performance
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_prev_verificato
            ON previsioni(verificato, timestamp_scadenza)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_prev_ticker
            ON previsioni(ticker, timestamp_emissione)
        """)

        conn.commit()
        print(f"✅ DB inizializzato: {DB_PATH}")
    finally:
        conn.close()


# ============================================================================
# PREVISIONI
# ============================================================================
def save_previsione(
    ticker: str,
    categoria: str,
    prezzo_emissione: float,
    score: float,
    direzione: str,
    sub_scores: Dict[str, float],
    orizzonte_giorni: int = ORIZZONTE_VERIFICA_GIORNI,
) -> int:
    """
    Salva una previsione emessa.
    Ritorna l'id della riga inserita.
    """
    now = datetime.now()
    scadenza = now + timedelta(days=orizzonte_giorni)

    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO previsioni (
                ticker, categoria, timestamp_emissione, prezzo_emissione,
                score, direzione, sub_scores_json, orizzonte_giorni,
                timestamp_scadenza, verificato
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, (
            ticker,
            categoria,
            now.isoformat(),
            prezzo_emissione,
            score,
            direzione,
            json.dumps(sub_scores),
            orizzonte_giorni,
            scadenza.isoformat(),
        ))
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_previsioni_da_verificare(categoria: Optional[str] = None) -> List[Dict]:
    """
    Ritorna le previsioni non verificate la cui scadenza è passata.
    Se `categoria` è passata, filtra per quella categoria.
    """
    now_iso = datetime.now().isoformat()

    conn = _get_connection()
    try:
        cur = conn.cursor()
        if categoria:
            cur.execute("""
                SELECT * FROM previsioni
                WHERE verificato = 0
                  AND timestamp_scadenza <= ?
                  AND categoria = ?
                ORDER BY timestamp_scadenza ASC
            """, (now_iso, categoria))
        else:
            cur.execute("""
                SELECT * FROM previsioni
                WHERE verificato = 0
                  AND timestamp_scadenza <= ?
                ORDER BY timestamp_scadenza ASC
            """, (now_iso,))

        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def registra_verifica(
    previsione_id: int,
    prezzo_verifica: float,
    rendimento_pct: float,
    esito: str,
    esito_corretto: bool,
) -> None:
    """Aggiorna una previsione con i dati della verifica."""
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            UPDATE previsioni
            SET verificato = 1,
                prezzo_verifica = ?,
                rendimento_pct = ?,
                esito = ?,
                esito_corretto = ?
            WHERE id = ?
        """, (
            prezzo_verifica,
            rendimento_pct,
            esito,
            1 if esito_corretto else 0,
            previsione_id,
        ))
        conn.commit()
    finally:
        conn.close()


def get_storico_ticker(ticker: str, limit: int = 100) -> List[Dict]:
    """Ritorna le ultime N previsioni per un ticker (anche non verificate)."""
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM previsioni
            WHERE ticker = ?
            ORDER BY timestamp_emissione DESC
            LIMIT ?
        """, (ticker, limit))
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_statistiche_categoria(categoria: str) -> Dict:
    """
    Ritorna statistiche aggregate per una categoria:
    - numero previsioni verificate
    - numero corrette
    - accuratezza (%)
    - rendimento medio dei segnali corretti
    """
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                COUNT(*) as totale,
                SUM(esito_corretto) as corrette,
                AVG(rendimento_pct) as rendimento_medio
            FROM previsioni
            WHERE categoria = ? AND verificato = 1
        """, (categoria,))
        row = cur.fetchone()

        totale = row["totale"] or 0
        corrette = row["corrette"] or 0
        rendimento = row["rendimento_medio"] or 0.0

        return {
            "totale_verificate": totale,
            "corrette": corrette,
            "accuratezza_pct": round((corrette / totale * 100.0), 2) if totale > 0 else 0.0,
            "rendimento_medio_pct": round(rendimento, 3),
        }
    finally:
        conn.close()


# ============================================================================
# PESI DINAMICI
# ============================================================================
def _init_pesi_categoria(categoria: str, default_weights: Dict[str, float]) -> None:
    """Inizializza i pesi di una categoria con i default."""
    now_iso = datetime.now().isoformat()
    conn = _get_connection()
    try:
        cur = conn.cursor()
        for ind, peso in default_weights.items():
            cur.execute("""
                INSERT OR IGNORE INTO pesi (categoria, indicatore, peso, updated_at)
                VALUES (?, ?, ?, ?)
            """, (categoria, ind, peso, now_iso))
        conn.commit()
    finally:
        conn.close()


def get_pesi_correnti(categoria: str, default_weights: Dict[str, float]) -> Dict[str, float]:
    """
    Ritorna i pesi correnti per una categoria.
    Se la categoria non esiste, la inizializza con i default.
    """
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT indicatore, peso FROM pesi
            WHERE categoria = ?
        """, (categoria,))
        rows = cur.fetchall()

        if not rows:
            conn.close()
            _init_pesi_categoria(categoria, default_weights)
            return dict(default_weights)

        return {r["indicatore"]: r["peso"] for r in rows}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def salva_pesi_aggiornati(categoria: str, nuovi_pesi: Dict[str, float]) -> None:
    """Salva i pesi aggiornati per una categoria (replace completo)."""
    now_iso = datetime.now().isoformat()
    conn = _get_connection()
    try:
        cur = conn.cursor()
        for ind, peso in nuovi_pesi.items():
            cur.execute("""
                INSERT INTO pesi (categoria, indicatore, peso, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(categoria, indicatore)
                DO UPDATE SET peso = excluded.peso,
                              updated_at = excluded.updated_at
            """, (categoria, ind, peso, now_iso))
        conn.commit()
    finally:
        conn.close()


# ============================================================================
# UTILITY
# ============================================================================
def classifica_esito(rendimento_pct: float) -> str:
    """
    Classifica un rendimento reale in BULLISH / BEARISH / NEUTRAL
    in base alla soglia SOGLIA_MOVIMENTO_PCT.
    """
    if rendimento_pct > SOGLIA_MOVIMENTO_PCT:
        return "BULLISH"
    elif rendimento_pct < -SOGLIA_MOVIMENTO_PCT:
        return "BEARISH"
    else:
        return "NEUTRAL"


def direzione_da_score(score: float) -> str:
    """
    Converte uno score continuo (0..1) in direzione operativa.
    - score >= 0.60 → BULLISH
    - score <  0.40 → BEARISH
    - altrimenti    → NEUTRAL
    """
    if score >= 0.60:
        return "BULLISH"
    elif score < 0.40:
        return "BEARISH"
    else:
        return "NEUTRAL"


# ============================================================================
# TEST MANUALE
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TEST db_manager.py")
    print("=" * 60)

    # 1. Init
    init_db()

    # 2. Salva una previsione di test
    sub_scores_test = {
        "ema_ma": 1.0, "trend": 0.75, "delta_ema_ma": 0.8,
        "ha_force": 1.0, "ha_state": 0.75, "zigzag": 1.0,
        "vol": 0.75, "rsi": 0.65, "macd": 0.75,
    }
    pid = save_previsione(
        ticker="TEST.MI",
        categoria="PORTFOLIO",
        prezzo_emissione=10.0,
        score=0.82,
        direzione=direzione_da_score(0.82),
        sub_scores=sub_scores_test,
        orizzonte_giorni=3,
    )
    print(f"✅ Previsione salvata, id={pid}")

    # 3. Leggi pesi correnti (con init ai default)
    default_w = {
        "ema_ma": 0.15, "trend": 0.13, "analyst": 0.05, "delta_ema_ma": 0.12,
        "ha_force": 0.15, "ha_state": 0.10, "zigzag": 0.10,
        "vol": 0.05, "close_change": 0.05, "rsi": 0.05, "macd": 0.05,
    }
    pesi = get_pesi_correnti("PORTFOLIO", default_w)
    print(f"✅ Pesi PORTFOLIO: {pesi}")

    # 4. Statistiche categoria
    stats = get_statistiche_categoria("PORTFOLIO")
    print(f"✅ Statistiche PORTFOLIO: {stats}")

    print("\n🏁 Test completato.")
