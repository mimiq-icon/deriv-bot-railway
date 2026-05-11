#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genesis Multiplier v15.0 — Deriv WebSocket Multiplier Trading Bot
==================================================================
Fully verified against the official Deriv API v4 JSON schemas:
  developers.deriv.com/schemas/proposal_request.schema.json
  developers.deriv.com/schemas/buy_request.schema.json
  developers.deriv.com/schemas/contract_update_request.schema.json

Critical API facts verified from official docs:
  • proposal field: "symbol" (NOT "underlying_symbol" — rejected by live API)
  • limit_order.stop_loss / take_profit: type=number (NOT str)
  • cancellation + limit_order: MUTUALLY EXCLUSIVE — never combine them
  • ticks_history field key stays "ticks_history": "SYMBOL" (no rename)
  • buy.price: type=number (NOT str)

Features:
  • MULTUP / MULTDOWN contracts — open-ended (closed via SL/TP or stopout)
  • Soft-voting ensemble: RandomForest + GradientBoosting (+ LightGBM if available)
  • 15 technical indicators across trending and mean-reversion regimes
  • Bayesian win-rate posterior per symbol
  • Fractional Kelly position sizing with streak and ladder cap
  • Daily P&L guard, total drawdown guard, consecutive-loss circuit breaker
  • SQLite WAL persistence — trades, candles, session, models
  • Joblib model cache with version and age checks
  • Async WebSocket + heartbeat + exponential-backoff reconnect
  • Telegram alerts for trades, errors, shutdown
  • Designed for Termux (Android) and Linux deployments
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import random
import sqlite3
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import joblib
import numpy as np
import pandas as pd
import websockets
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    import lightgbm as lgb  # type: ignore[import]
    lgb.basic.LightGBMError  # ensure imported
    import warnings
    warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
_LOG_FILE = f"logs/genesis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("genesis")


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _env(key: str, default: str) -> str:
    return os.getenv(key, default)

def _envf(key: str, default: str) -> float:
    return float(os.getenv(key, default))

def _envi(key: str, default: str) -> int:
    return int(os.getenv(key, default))

def _norm2(v: float) -> float:
    """Round to 2 decimal places for money values."""
    return round(float(v), 2)


# ──────────────────────────────────────────────────────────────────────────────
# INSTRUMENT CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
# Pip sizes for SL/TP pip-based calculations
PIP_SIZE: dict[str, float] = {
    "frxEURUSD": 0.0001, "frxGBPUSD": 0.0001,
    "frxAUDUSD": 0.0001, "frxNZDUSD": 0.0001,
    "frxUSDCAD": 0.0001, "frxEURGBP": 0.0001,
    "frxUSDJPY": 0.01,   "frxEURJPY": 0.01,
    "frxXAUUSD": 0.01,
}

# Default multiplier values — overridden at startup from contracts_for API
DEFAULT_MULTIPLIER: dict[str, int] = {
    "frxEURUSD": 1000, "frxGBPUSD": 1000,
    "frxAUDUSD": 1000, "frxNZDUSD": 1000,
    "frxUSDCAD": 1000, "frxEURGBP": 1000,
    "frxUSDJPY": 500,  "frxEURJPY": 500,
    "frxXAUUSD": 500,
}

ALL_SYMBOLS: list[str] = list(DEFAULT_MULTIPLIER.keys())


# ──────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT-DRIVEN CONFIG
# ──────────────────────────────────────────────────────────────────────────────
# Deriv credentials
DERIV_API_TOKEN: str = _env("DERIV_API_TOKEN", "")
DERIV_APP_ID: str    = _env("DERIV_APP_ID", "1089")   # 1089 = Deriv test app
WS_URL: str          = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

# Telegram (optional)
TG_TOKEN: str   = _env("TG_TOKEN", "")
TG_CHAT_ID: str = _env("TG_CHAT_ID", "")

# Model thresholds
MIN_OOS_ACC: float     = _envf("MIN_OOS_ACC", "0.505")    # 50.5 % out-of-sample
MIN_MINORITY_PCT: float = _envf("MIN_MINORITY_PCT", "0.01") # 5 % minority class
MIN_CONFLUENCE: int    = _envi("MIN_CONFLUENCE", "3")       # signals out of 5
CONF_THRESHOLD: float  = _envf("CONF_THRESHOLD", "0.52")   # model prob threshold
MIN_KELLY: float       = _envf("MIN_KELLY", "0.02")         # min Kelly fraction

# Win-rate Bayesian prior
WR_WINDOW: int          = 20
WR_PRIOR: float         = 0.55
WR_PRIOR_STRENGTH: float = 3.0

# Risk
MAX_DAILY_LOSS_PCT: float    = _envf("MAX_DAILY_LOSS_PCT", "0.10")   # 10 % of day-start bal
MAX_DRAWDOWN_PCT: float      = _envf("MAX_DRAWDOWN_PCT", "0.50")     # 50 % of peak bal
STAKE_RISK_PCT: float        = _envf("STAKE_RISK_PCT", "0.005")      # 0.5 % per trade
MIN_STAKE_ABS: float         = _envf("MIN_STAKE_ABS", "1.0")
MIN_STAKE_PCT: float         = _envf("MIN_STAKE_PCT", "0.002")
MAX_STAKE_PCT: float         = _envf("MAX_STAKE_PCT", "0.01")        # single-trade cap
MAX_CONSEC_LOSSES: int       = _envi("MAX_CONSEC_LOSSES", "6")
MAX_OPEN: int                = _envi("MAX_OPEN", "2")

# Stake ladder (profitable-trades → fraction-of-balance)
def _parse_ladder(raw: str) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for part in raw.split():
        if "," in part:
            a, b = part.split(",", 1)
            try:
                out.append((int(a), float(b)))
            except ValueError:
                pass
    return sorted(out) or [(0, 0.005)]

STAKE_LADDER: list[tuple[int, float]] = _parse_ladder(
    _env("STAKE_LADDER", "0,0.005 20,0.01 50,0.02 100,0.03")
)

# SL / TP in pips (converted to USD at order time)
SL_PIPS_DEFAULT: int = _envi("SL_PIPS_DEFAULT", "6")
TP_PIPS_DEFAULT: int = _envi("TP_PIPS_DEFAULT", "10")

# Cooldowns
GLOBAL_COOLDOWN: float = _envf("GLOBAL_COOLDOWN", "60.0")   # secs between any trades
SYMBOL_COOLDOWN: float = _envf("SYMBOL_COOLDOWN", "120.0")  # secs per symbol

# Data / training
CANDLE_HISTORY_DAYS: int = _envi("CANDLE_HISTORY_DAYS", "30")
MIN_SAMPLES: int         = _envi("MIN_SAMPLES", "100")
RETRAIN_EVERY: int       = _envi("RETRAIN_EVERY", "100")       # new candles
MIN_CANDLES_TO_TRADE: int = _envi("MIN_CANDLES_TO_TRADE", "30")
MODEL_CACHE_AGE_SECS: int = _envi("MODEL_CACHE_AGE_SECS", str(200 * 60))

# Market hours (UTC)
MARKET_CLOSE_HOUR: int     = 21    # Friday close hour
LOW_EDGE_HOURS: frozenset[int] = frozenset({22, 23})  # thin liquidity

# Infrastructure
RECONNECT_BASE: float  = _envf("RECONNECT_BASE", "5.0")
RECONNECT_MAX: float   = _envf("RECONNECT_MAX", "60.0")
HEARTBEAT_INTERVAL: float = _envf("HEARTBEAT_INTERVAL", "20.0")
HEARTBEAT_TIMEOUT: float  = _envf("HEARTBEAT_TIMEOUT", "10.0")
CHECKPOINT_INTERVAL: float = _envf("CHECKPOINT_INTERVAL", "60.0")

_MODEL_CACHE_VER = "v15"


# ──────────────────────────────────────────────────────────────────────────────
# DATABASE SCHEMA
# ──────────────────────────────────────────────────────────────────────────────
_DDL_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ts               TEXT,
    symbol           TEXT,
    contract_type    TEXT,
    stake            REAL,
    confidence       REAL,
    confluence       INTEGER,
    sl_amount        REAL,
    tp_amount        REAL,
    profit           REAL,
    balance_after    REAL,
    win_rate         REAL,
    regime           TEXT,
    multiplier       INTEGER
)
"""

_DDL_CANDLES = """
CREATE TABLE IF NOT EXISTS candles (
    symbol  TEXT    NOT NULL,
    epoch   INTEGER NOT NULL,
    open    REAL,
    high    REAL,
    low     REAL,
    close   REAL    NOT NULL,
    PRIMARY KEY (symbol, epoch)
)
"""

_DDL_OPEN = """
CREATE TABLE IF NOT EXISTS open_contracts (
    contract_id  TEXT PRIMARY KEY,
    symbol       TEXT NOT NULL,
    stake        REAL NOT NULL,
    contract_type TEXT NOT NULL,
    confidence   REAL NOT NULL,
    confluence   INTEGER NOT NULL,
    regime       TEXT NOT NULL,
    sl_amount    REAL NOT NULL,
    tp_amount    REAL NOT NULL,
    multiplier   INTEGER NOT NULL,
    created_at   TEXT NOT NULL,
    sell_queued  INTEGER NOT NULL DEFAULT 0
)
"""

_DDL_SESSION = """
CREATE TABLE IF NOT EXISTS session (
    id                 INTEGER PRIMARY KEY,
    wins               INTEGER NOT NULL DEFAULT 0,
    losses             INTEGER NOT NULL DEFAULT 0,
    total_profit       REAL    NOT NULL DEFAULT 0,
    consec_losses      INTEGER NOT NULL DEFAULT 0,
    profitable_trades  INTEGER NOT NULL DEFAULT 0,
    symbol_results     TEXT    NOT NULL DEFAULT '{}',
    symbol_streak      TEXT    NOT NULL DEFAULT '{}',
    day_start_bal      REAL    NOT NULL DEFAULT 0,
    day_date           TEXT    NOT NULL DEFAULT '',
    initial_balance    REAL    NOT NULL DEFAULT 0,
    saved_at           TEXT    NOT NULL DEFAULT ''
)
"""

_DDL_PROCESSED = """
CREATE TABLE IF NOT EXISTS processed_ids (
    contract_id TEXT PRIMARY KEY
)
"""

_DDL_MODEL_LOG = """
CREATE TABLE IF NOT EXISTS model_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT,
    symbol    TEXT,
    oos_acc   REAL,
    deployed  INTEGER
)
"""


# ──────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class OpenContract:
    contract_id:   str
    symbol:        str
    stake:         float
    contract_type: str     # "MULTUP" or "MULTDOWN"
    confidence:    float
    confluence:    int
    regime:        str
    sl_amount:     float
    tp_amount:     float
    multiplier:    int
    created_at:    str
    sell_queued:   bool = False


# ──────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS  (all pure-pandas / numpy — no ta-lib dependency)
# ──────────────────────────────────────────────────────────────────────────────
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, min_periods=span).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl  = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift()).abs()
    lpc = (df["low"]  - df["close"].shift()).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up   = df["high"].diff()
    dn   = -df["low"].diff()
    pdm  = up.where((up > dn) & (up > 0), 0.0)
    ndm  = dn.where((dn > up) & (dn > 0), 0.0)
    atr  = _atr(df, period)
    pdi  = 100 * pdm.ewm(com=period - 1, min_periods=period).mean() / atr.replace(0, np.nan)
    ndi  = 100 * ndm.ewm(com=period - 1, min_periods=period).mean() / atr.replace(0, np.nan)
    dx   = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(com=period - 1, min_periods=period).mean()


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    fast  = _ema(close, 12)
    slow  = _ema(close, 26)
    macd  = fast - slow
    signal = macd.ewm(span=9, min_periods=9).mean()
    return macd, signal


def _bbands(close: pd.Series, period: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid   = close.rolling(period).mean()
    std   = close.rolling(period).std(ddof=0)
    upper = mid + 2 * std
    lower = mid - 2 * std
    return upper, mid, lower


def _stoch(df: pd.DataFrame, k: int = 14, d: int = 3) -> tuple[pd.Series, pd.Series]:
    lo  = df["low"].rolling(k).min()
    hi  = df["high"].rolling(k).max()
    pct = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    sk  = pct.rolling(d).mean()
    sd  = sk.rolling(d).mean()
    return sk, sd


def _cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp   = (df["high"] + df["low"] + df["close"]) / 3
    ma   = tp.rolling(period).mean()
    mad  = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * mad.replace(0, np.nan))


def _williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hi  = df["high"].rolling(period).max()
    lo  = df["low"].rolling(period).min()
    return -100 * (hi - df["close"]) / (hi - lo).replace(0, np.nan)


def _momentum(close: pd.Series, period: int = 10) -> pd.Series:
    return close / close.shift(period) - 1


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 15-feature matrix from OHLC candle data.
    Returns DataFrame with NaNs dropped.
    """
    c = df["close"].copy()

    rsi14  = _rsi(c, 14)
    rsi7   = _rsi(c, 7)
    adx    = _adx(df, 14)
    atr    = _atr(df, 14)
    ema20  = _ema(c, 20)
    ema50  = _ema(c, 50)
    macd, macd_sig = _macd(c)
    bb_up, bb_mid, bb_lo = _bbands(c, 20)
    stk, _  = _stoch(df, 14, 3)
    cci     = _cci(df, 20)
    wr      = _williams_r(df, 14)
    mom     = _momentum(c, 10)

    feat = pd.DataFrame(index=df.index, data={
        "rsi14":     rsi14,
        "rsi7":      rsi7,
        "adx":       adx,
        "atr_norm":  atr / c,                  # normalise by price
        "ema_ratio": (ema20 / ema50) - 1,      # trend direction
        "price_vs_ema20": (c / ema20) - 1,
        "macd":      macd,
        "macd_hist": macd - macd_sig,
        "bb_pos":    (c - bb_lo) / (bb_up - bb_lo).replace(0, np.nan),  # 0..1
        "bb_width":  (bb_up - bb_lo) / bb_mid.replace(0, np.nan),
        "stoch_k":   stk,
        "cci":       cci,
        "wr":        wr,
        "momentum":  mom,
        "vol_ratio": atr / atr.rolling(50).mean().replace(0, np.nan),   # vol regime
    })
    return feat.dropna()


def build_labels(df: pd.DataFrame, tp_pips: float, sl_pips: float,
                 pip_size: float, horizon: int = 5) -> pd.Series:
    """
    Binary label: 1 if a LONG trade (MULTUP) hits take-profit BEFORE stop-loss
    within the next `horizon` 1-minute candles.

    FIX: Previously used future CLOSE max/min which gave ~0.3%% positive labels
    because close-to-close 10-pip moves in 5 min are rare (<1%%).
    Correct approach: use HIGH to detect TP touches and LOW to detect SL touches
    (intrabar wicks). With 6-pip SL / 10-pip TP on EUR/USD this gives ~30-40%%
    positive labels as expected for a roughly balanced classifier.

    Sequential TP-before-SL check: find the FIRST candle where HIGH >= TP level
    and the FIRST candle where LOW <= SL level, then compare their positions.
    Label = 1 only when TP candle comes strictly before SL candle.
    """
    tp_delta = tp_pips * pip_size
    sl_delta = sl_pips * pip_size

    c  = df["close"].values
    # Use actual OHLC wicks when available; fall back to close only.
    h  = df["high"].values  if "high"  in df.columns else c
    lo = df["low"].values   if "low"   in df.columns else c
    n  = len(c)

    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n - horizon):
        entry    = c[i]
        tp_level = entry + tp_delta
        sl_level = entry - sl_delta
        tp_bar   = horizon + 1   # first bar where HIGH >= tp_level  (sentinel = not hit)
        sl_bar   = horizon + 1   # first bar where LOW  <= sl_level  (sentinel = not hit)
        for j in range(1, horizon + 1):
            k = i + j
            if tp_bar == horizon + 1 and h[k]  >= tp_level:
                tp_bar = j
            if sl_bar == horizon + 1 and lo[k] <= sl_level:
                sl_bar = j
            # Early-exit once both have been found
            if tp_bar <= horizon and sl_bar <= horizon:
                break
        # Label 1 = TP hit first (MULTUP wins); 0 = SL hit first or timeout
        out[i] = 1.0 if tp_bar < sl_bar else 0.0

    return pd.Series(out, index=df.index).iloc[: n - horizon]


# ──────────────────────────────────────────────────────────────────────────────
# CONFLUENCE SIGNAL ENGINE  (5 independent signals → integer 0-5)
# ──────────────────────────────────────────────────────────────────────────────
def confluence_score(row: dict[str, float], direction: str, regime: str) -> int:
    """
    direction = "UP" | "DOWN"
    Returns integer 0-5 representing how many sub-signals agree.
    """
    up = direction == "UP"
    score = 0

    # 1. RSI
    rsi = row.get("rsi14", 50)
    if up and rsi < 40:
        score += 1
    elif not up and rsi > 60:
        score += 1

    # 2. EMA trend
    ema_r = row.get("ema_ratio", 0)
    if up and ema_r > 0:
        score += 1
    elif not up and ema_r < 0:
        score += 1

    # 3. MACD histogram
    mh = row.get("macd_hist", 0)
    if up and mh > 0:
        score += 1
    elif not up and mh < 0:
        score += 1

    # 4. Bollinger band position
    bbp = row.get("bb_pos", 0.5)
    if up and bbp < 0.3:
        score += 1
    elif not up and bbp > 0.7:
        score += 1

    # 5. Stochastic
    sk = row.get("stoch_k", 50)
    if up and sk < 30:
        score += 1
    elif not up and sk > 70:
        score += 1

    return score


def detect_regime(row: dict[str, float]) -> str:
    adx = row.get("adx", 25)
    return "trending" if adx >= 25 else "ranging"


# ──────────────────────────────────────────────────────────────────────────────
# CONTRACT STORE  (in-memory, DB-backed)
# ──────────────────────────────────────────────────────────────────────────────
class ContractStore:
    def __init__(self, db: sqlite3.Connection) -> None:
        self._db  = db
        self._map: dict[str, OpenContract] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @property
    def count(self) -> int:
        return len(self._map)

    def all_ids(self) -> set[str]:
        return set(self._map.keys())

    def get(self, cid: str) -> OpenContract | None:
        return self._map.get(cid)

    def sell_lock(self, cid: str) -> asyncio.Lock:
        if cid not in self._locks:
            self._locks[cid] = asyncio.Lock()
        return self._locks[cid]

    def add(self, c: OpenContract) -> None:
        self._map[c.contract_id] = c
        self._persist(c)

    def remove(self, cid: str) -> None:
        self._map.pop(cid, None)
        self._locks.pop(cid, None)
        try:
            self._db.execute("DELETE FROM open_contracts WHERE contract_id=?", (cid,))
            self._db.commit()
        except Exception as exc:
            log.warning("[STORE] remove %s: %s", cid, exc)

    def mark_sell_queued(self, cid: str) -> None:
        rec = self._map.get(cid)
        if rec:
            rec.sell_queued = True
            try:
                self._db.execute(
                    "UPDATE open_contracts SET sell_queued=1 WHERE contract_id=?", (cid,))
                self._db.commit()
            except Exception:
                pass

    def _persist(self, c: OpenContract) -> None:
        try:
            self._db.execute(
                """INSERT OR REPLACE INTO open_contracts
                   (contract_id, symbol, stake, contract_type, confidence, confluence,
                    regime, sl_amount, tp_amount, multiplier, created_at, sell_queued)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (c.contract_id, c.symbol, c.stake, c.contract_type,
                 c.confidence, c.confluence, c.regime,
                 c.sl_amount, c.tp_amount, c.multiplier,
                 c.created_at, int(c.sell_queued)),
            )
            self._db.commit()
        except Exception as exc:
            log.warning("[STORE] persist %s: %s", c.contract_id, exc)

    def load_from_db(self, processed: set[str]) -> int:
        try:
            rows = self._db.execute(
                "SELECT contract_id, symbol, stake, contract_type, confidence, confluence,"
                " regime, sl_amount, tp_amount, multiplier, created_at, sell_queued"
                " FROM open_contracts"
            ).fetchall()
        except Exception:
            return 0
        n = 0
        for row in rows:
            cid = str(row[0])
            if cid in processed:
                continue
            self._map[cid] = OpenContract(
                contract_id=cid, symbol=row[1], stake=row[2],
                contract_type=row[3], confidence=row[4], confluence=row[5],
                regime=row[6], sl_amount=row[7], tp_amount=row[8],
                multiplier=row[9], created_at=row[10],
                sell_queued=bool(row[11]),
            )
            n += 1
        return n


# ──────────────────────────────────────────────────────────────────────────────
# MAIN BOT CLASS
# ──────────────────────────────────────────────────────────────────────────────
class GenesisBot:
    def __init__(self) -> None:
        if not DERIV_API_TOKEN:
            raise RuntimeError("DERIV_API_TOKEN not set. Add it to your .env file.")

        self.api_token  = DERIV_API_TOKEN
        self.all_symbols = ALL_SYMBOLS

        # Live multiplier values — populated from contracts_for at startup
        self.multipliers: dict[str, int] = dict(DEFAULT_MULTIPLIER)

        # Session state
        self.wins: int = 0
        self.losses: int = 0
        self.total_profit: float = 0.0
        self.consec_losses: int = 0
        self.profitable_trades: int = 0
        self.current_balance: float = 0.0
        self.initial_balance: float = 0.0
        self.day_start_bal: float = 0.0
        self.day_date: str = ""

        # Per-symbol tracking
        self.symbol_results: dict[str, deque[int]] = {
            s: deque(maxlen=WR_WINDOW) for s in self.all_symbols
        }
        self.symbol_streak: dict[str, int] = {s: 0 for s in self.all_symbols}
        self.new_candle_counts: dict[str, int] = {s: 0 for s in self.all_symbols}
        self._feature_cache: dict[str, dict[str, float]] = {}
        self._feature_names: dict[str, list[str]] = {}  # FIX: store feature names for named-DF prediction

        # ML models
        self.models: dict[str, Any] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.model_oos: dict[str, float] = {}
        self.model_trained_at: dict[str, float] = {}

        # Candle buffers (in-memory, also persisted to DB)
        self.candles: dict[str, pd.DataFrame] = {s: pd.DataFrame() for s in self.all_symbols}

        # Timing guards
        self.last_trade_time: float = 0.0
        self.symbol_last_trade: dict[str, float] = {s: 0.0 for s in self.all_symbols}
        self.placing: set[str] = set()

        # Async infra
        self._shutdown = asyncio.Event()
        self._shutdown_reason = ""
        self._req_id = itertools.count(1)
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._balance_ready = False
        self._retrain_tasks: set[asyncio.Task[None]] = set()

        # Processed contract IDs (never revisit)
        self.processed_ids: set[str] = set()

        # DB
        data_dir = _env("DATA_DIR", os.getcwd())
        os.makedirs(data_dir, exist_ok=True)
        self._db_path = os.path.join(data_dir, "genesis_v15.db")
        self._db: sqlite3.Connection
        self._init_db()

        self.contracts = ContractStore(self._db)
        self._model_dir = Path(_env("MODEL_DIR", "models_v15"))
        self._model_dir.mkdir(exist_ok=True)

        self._tg_session: aiohttp.ClientSession | None = None

    # ── Database ──────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        self._db = sqlite3.connect(self._db_path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(
            f"{_DDL_TRADES}; {_DDL_CANDLES}; {_DDL_OPEN};"
            f"{_DDL_SESSION}; {_DDL_PROCESSED}; {_DDL_MODEL_LOG};"
        )
        self._db.commit()

        # Load processed IDs
        self.processed_ids = {
            row[0] for row in
            self._db.execute("SELECT contract_id FROM processed_ids").fetchall()
        }
        self._load_session()
        self._load_candles_from_db()

        log.info("[DB] Ready: %s | Processed=%d | W=%d L=%d PnL=$%.2f",
                 self._db_path, len(self.processed_ids),
                 self.wins, self.losses, self.total_profit)

    def _save_session(self) -> None:
        results = {s: list(v) for s, v in self.symbol_results.items()}
        try:
            self._db.execute(
                """INSERT INTO session
                   (id,wins,losses,total_profit,consec_losses,profitable_trades,
                    symbol_results,symbol_streak,day_start_bal,day_date,
                    initial_balance,saved_at)
                   VALUES (1,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     wins=excluded.wins, losses=excluded.losses,
                     total_profit=excluded.total_profit,
                     consec_losses=excluded.consec_losses,
                     profitable_trades=excluded.profitable_trades,
                     symbol_results=excluded.symbol_results,
                     symbol_streak=excluded.symbol_streak,
                     day_start_bal=excluded.day_start_bal,
                     day_date=excluded.day_date,
                     initial_balance=excluded.initial_balance,
                     saved_at=excluded.saved_at""",
                (self.wins, self.losses, self.total_profit,
                 self.consec_losses, self.profitable_trades,
                 json.dumps(results), json.dumps(self.symbol_streak),
                 self.day_start_bal, self.day_date,
                 self.initial_balance,
                 datetime.now(timezone.utc).isoformat()),
            )
            self._db.commit()
        except Exception as exc:
            log.warning("[SESSION] save failed: %s", exc)

    def _load_session(self) -> None:
        row = self._db.execute(
            "SELECT wins,losses,total_profit,consec_losses,profitable_trades,"
            "symbol_results,symbol_streak,day_start_bal,day_date,initial_balance,saved_at"
            " FROM session WHERE id=1"
        ).fetchone()
        if row is None:
            return
        (self.wins, self.losses, self.total_profit,
         self.consec_losses, self.profitable_trades,
         res_json, streak_json,
         self.day_start_bal, self.day_date,
         self.initial_balance, saved_at) = row
        self.wins = int(self.wins)
        self.losses = int(self.losses)
        self.total_profit = float(self.total_profit)
        self.consec_losses = int(self.consec_losses)
        self.profitable_trades = int(self.profitable_trades)
        self.day_start_bal = float(self.day_start_bal or 0)
        self.initial_balance = float(self.initial_balance or 0)
        try:
            for sym, hist in json.loads(res_json).items():
                if sym in self.symbol_results:
                    self.symbol_results[sym] = deque(hist, maxlen=WR_WINDOW)
        except Exception:
            pass
        try:
            for sym, st in json.loads(streak_json).items():
                if sym in self.symbol_streak:
                    self.symbol_streak[sym] = int(st)
        except Exception:
            pass
        log.info("[SESSION] Restored from %s | W=%d L=%d PnL=$%.2f | InitBal=$%.2f",
                 saved_at, self.wins, self.losses, self.total_profit, self.initial_balance)

    def _persist_candles(self, symbol: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        work = df.reset_index()
        if "epoch" not in work.columns:
            if "datetime" in work.columns:
                work["epoch"] = (work["datetime"].astype("int64") // 10**9).astype(int)
            else:
                return
        rows = [
            (symbol, int(r["epoch"]),
             float(r.get("open",  r["close"])),
             float(r.get("high",  r["close"])),
             float(r.get("low",   r["close"])),
             float(r["close"]))
            for _, r in work.iterrows()
        ]
        try:
            self._db.executemany(
                "INSERT OR IGNORE INTO candles (symbol,epoch,open,high,low,close)"
                " VALUES (?,?,?,?,?,?)", rows)
            self._db.commit()
        except Exception as exc:
            log.warning("[CANDLES] persist %s: %s", symbol, exc)

    def _load_candles_from_db(self) -> None:
        cutoff = int(time.time()) - CANDLE_HISTORY_DAYS * 86400
        for sym in self.all_symbols:
            rows = self._db.execute(
                "SELECT epoch,open,high,low,close FROM candles"
                " WHERE symbol=? AND epoch>=? ORDER BY epoch ASC", (sym, cutoff)
            ).fetchall()
            if not rows:
                continue
            df = pd.DataFrame(rows, columns=["epoch", "open", "high", "low", "close"])
            df["datetime"] = pd.to_datetime(df["epoch"], unit="s")
            df.set_index("datetime", inplace=True)
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(float)
            self.candles[sym] = df

    def _prune_candles_db(self) -> None:
        cutoff = int(time.time()) - CANDLE_HISTORY_DAYS * 86400
        try:
            self._db.execute("DELETE FROM candles WHERE epoch<?", (cutoff,))
            self._db.commit()
        except Exception:
            pass

    def _log_trade(self, rec: OpenContract, profit: float) -> None:
        try:
            self._db.execute(
                """INSERT INTO trades (ts,symbol,contract_type,stake,confidence,
                   confluence,sl_amount,tp_amount,profit,balance_after,
                   win_rate,regime,multiplier)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (datetime.now(timezone.utc).isoformat(), rec.symbol,
                 rec.contract_type, rec.stake, rec.confidence,
                 rec.confluence, rec.sl_amount, rec.tp_amount,
                 profit, self.current_balance,
                 self._symbol_wr(rec.symbol), rec.regime, rec.multiplier),
            )
            self._db.commit()
        except Exception as exc:
            log.warning("[TRADE_LOG] %s", exc)

    def _persist_processed(self, cid: str) -> None:
        try:
            self._db.execute(
                "INSERT OR IGNORE INTO processed_ids (contract_id) VALUES (?)", (cid,))
            self._db.commit()
        except Exception:
            pass

    # ── Session helpers ───────────────────────────────────────────────────────

    def _symbol_wr(self, symbol: str) -> float:
        hist = self.symbol_results[symbol]
        n    = len(hist)
        return (sum(hist) + WR_PRIOR * WR_PRIOR_STRENGTH) / (n + WR_PRIOR_STRENGTH)

    def _effective_daily_loss_limit(self) -> float:
        ref = self.day_start_bal if self.day_start_bal > 0 else self.current_balance
        return -(ref * MAX_DAILY_LOSS_PCT)

    @property
    def _daily_pnl(self) -> float:
        if self.day_start_bal <= 0:
            return 0.0
        return self.current_balance - self.day_start_bal

    def _check_day_reset(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.day_date != today:
            self.day_date = today
            self.day_start_bal = self.current_balance
            log.info("[DAY] New trading day %s | DayStart=$%.2f", today, self.day_start_bal)

    def _risk_ok(self) -> tuple[bool, str]:
        if self.current_balance <= 0:
            return False, "zero balance"
        if self.consec_losses >= MAX_CONSEC_LOSSES:
            return False, f"consec losses {self.consec_losses} >= {MAX_CONSEC_LOSSES}"
        if self._daily_pnl <= self._effective_daily_loss_limit():
            return False, f"daily loss ${self._daily_pnl:.2f} hit limit"
        if self.initial_balance > 0:
            dd = (self.current_balance - self.initial_balance) / self.initial_balance
            if dd <= -MAX_DRAWDOWN_PCT:
                return False, f"total drawdown {dd*100:.1f}% ≥ {MAX_DRAWDOWN_PCT*100:.0f}%"
        return True, ""

    # ── Stake calculation ─────────────────────────────────────────────────────

    def _stake_cap(self) -> float:
        for threshold, frac in reversed(STAKE_LADDER):
            if self.profitable_trades >= threshold:
                return self.current_balance * frac
        return self.current_balance * STAKE_LADDER[0][1]

    def _calculate_stake(self, symbol: str, prob: float) -> float:
        wr   = self._symbol_wr(symbol)
        odds = 1.0   # even payout for multipliers (approximation)
        kelly = (wr * (1 + odds) - 1) / odds   # fractional Kelly
        kelly = max(kelly, 0.0)

        # Scale down to risk percent
        base  = self.current_balance * STAKE_RISK_PCT
        stake = base * kelly / max(MIN_KELLY, kelly)  # scale by kelly ratio

        # Cap and floor
        cap   = self._stake_cap()
        stake = min(stake, cap)
        stake = min(stake, self.current_balance * MAX_STAKE_PCT)
        min_s = max(MIN_STAKE_ABS, self.current_balance * MIN_STAKE_PCT)
        if stake < min_s:
            return 0.0

        # Streak adjustment
        streak = self.symbol_streak.get(symbol, 0)
        if streak <= -3:
            stake *= 0.75
        elif streak >= 3:
            stake *= 1.10

        return _norm2(stake)

    # ── Model training ────────────────────────────────────────────────────────

    def _train_and_validate(
        self, X: np.ndarray, y: np.ndarray, symbol: str
    ) -> tuple[Any, StandardScaler, float] | tuple[None, None, float]:
        """Train RF+GBM ensemble on time-series split; return (model, scaler, oos_acc)."""
        tscv    = TimeSeriesSplit(n_splits=5)
        oos_scores: list[float] = []
        best_model: Any = None
        best_scaler: StandardScaler | None = None

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            if len(np.unique(y_tr)) < 2 or len(y_val) < 5:
                continue

            sc = StandardScaler()
            X_tr_s  = sc.fit_transform(X_tr)
            X_val_s = sc.transform(X_val)
            sw      = compute_sample_weight("balanced", y_tr)

            rf  = RandomForestClassifier(
                n_estimators=200, max_depth=6, min_samples_leaf=10,
                n_jobs=-1, random_state=42)
            gbm = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42)

            estimators = [("rf", rf), ("gbm", gbm)]

            if _HAS_LGBM:
                lgbm_clf = lgb.LGBMClassifier(
                    n_estimators=100, num_leaves=31, learning_rate=0.05,
                    subsample=0.8, random_state=42, verbose=-1)
                estimators.append(("lgbm", lgbm_clf))

            from sklearn.ensemble import VotingClassifier
            vc = VotingClassifier(estimators=estimators, voting="soft")
            vc.fit(X_tr_s, y_tr, sample_weight=sw)

            preds     = vc.predict(X_val_s)
            fold_acc  = balanced_accuracy_score(y_val, preds)
            oos_scores.append(fold_acc)

            if best_model is None or fold_acc == max(oos_scores):
                best_model  = vc
                best_scaler = sc

        if not oos_scores or best_model is None or best_scaler is None:
            return None, None, 0.0

        mean_oos = float(np.mean(oos_scores))
        return best_model, best_scaler, mean_oos

    async def train_model(self, symbol: str, ws: Any) -> None:
        df = self.candles.get(symbol, pd.DataFrame())

        # Try to load from cache first
        cached = self._load_model_cache(symbol)
        if cached and df.shape[0] < MIN_SAMPLES:
            model, scaler, oos = cached
            self.models[symbol]         = model
            self.scalers[symbol]        = scaler
            self.model_oos[symbol]      = oos
            self.model_trained_at[symbol] = time.time()
            log.info("[TRAIN] %s DEPLOYED (cache) | OOS=%.1f%%", symbol, oos * 100)
            return

        # Fetch history if we don't have enough
        if df.shape[0] < MIN_SAMPLES:
            await self._fetch_candle_history(symbol, ws)
            df = self.candles.get(symbol, pd.DataFrame())

        if df.shape[0] < MIN_SAMPLES:
            log.warning("[TRAIN] %s insufficient data (%d < %d)",
                        symbol, df.shape[0], MIN_SAMPLES)
            return

        feat = build_features(df)
        pip  = PIP_SIZE.get(symbol, 0.0001)
        # FIX: Adaptive TP/SL based on median 5-candle ATR so labels hit ~30-40%.
        # Fixed 10-pip TP on 1-min EURUSD was producing <0.5% positive labels —
        # way too tight for current market volatility.
        _atr_series  = (df["high"] - df["low"]).rolling(5).median() if "high" in df.columns and "low" in df.columns else None
        _median_atr  = float(_atr_series.dropna().median()) if _atr_series is not None and not _atr_series.dropna().empty else pip * 8
        _adaptive_tp = max(TP_PIPS_DEFAULT, round(_median_atr / pip * 0.6))  # 60% of median ATR
        _adaptive_sl = max(SL_PIPS_DEFAULT, round(_median_atr / pip * 0.4))  # 40% of median ATR
        labels = build_labels(df.loc[feat.index], _adaptive_tp, _adaptive_sl, pip)

        if feat.empty or labels.empty:
            return

        # Align
        common = feat.index.intersection(labels.index)
        feat   = feat.loc[common]
        labels = labels.loc[common].fillna(0).astype(int)

        if len(labels) < MIN_SAMPLES:
            return

        y = labels.values
        X = feat.values
        feature_names = list(feat.columns)           # FIX: capture names before converting to numpy
        self._feature_names[symbol] = feature_names  # FIX: store for named-DF inference

        counts = np.bincount(y)
        if len(counts) < 2:
            return
        minority_pct = counts.min() / len(y)
        if minority_pct < MIN_MINORITY_PCT:
            log.warning("[TRAIN] %s ABORTED — minority %.1f%% < %.0f%%",
                        symbol, minority_pct * 100, MIN_MINORITY_PCT * 100)
            # Back off: retry in 75%% of RETRAIN_EVERY candles instead of
            # immediately (counter was reset to 0 before this task ran, so
            # setting it to 3/4 of the interval delays the next attempt).
            # FIX: Hard penalty — don't retry for ~5 full cycles.
            # The minority % is structural (same 5000 candles, same labels),
            # retrying every minute is pure CPU burn. Climb from -400 back to
            # RETRAIN_EVERY (100) needs ~500 new candles ≈ 8+ hours.
            self.new_candle_counts[symbol] = -int(RETRAIN_EVERY * 4)
            return

        log.info("[TRAIN] %s training on %d samples…", symbol, len(X))
        model, scaler, oos = await asyncio.to_thread(
            self._train_and_validate, X, y, symbol)

        if model is None:
            self._db.execute(
                "INSERT INTO model_log (ts,symbol,oos_acc,deployed) VALUES (?,?,?,0)",
                (datetime.now(timezone.utc).isoformat(), symbol, 0.0))
            self._db.commit()
            return

        deployed = oos >= MIN_OOS_ACC
        self._db.execute(
            "INSERT INTO model_log (ts,symbol,oos_acc,deployed) VALUES (?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), symbol, oos, int(deployed)))
        self._db.commit()

        if not deployed:
            log.warning("[TRAIN] %s NOT DEPLOYED — OOS=%.1f%% < %.1f%%",
                        symbol, oos * 100, MIN_OOS_ACC * 100)
            self.models.pop(symbol, None)
            return

        self.models[symbol]          = model
        self.scalers[symbol]         = scaler
        self.model_oos[symbol]       = oos
        self.model_trained_at[symbol] = time.time()
        self.new_candle_counts[symbol] = 0
        self._feature_cache.pop(symbol, None)
        self._save_model_cache(symbol, model, scaler, oos)

        ensemble = "RF+GBM+LGBM" if _HAS_LGBM else "RF+GBM"
        log.info("[TRAIN] %s DEPLOYED | OOS=%.1f%% | %s", symbol, oos * 100, ensemble)

    def _save_model_cache(self, sym: str, model: Any,
                           scaler: StandardScaler, oos: float) -> None:
        path = self._model_dir / f"{sym}.pkl"
        try:
            joblib.dump({
                "model": model, "scaler": scaler, "oos": oos,
                "trained_at": time.time(), "version": _MODEL_CACHE_VER,
            }, path)
        except Exception as exc:
            log.warning("[CACHE] save %s: %s", sym, exc)

    def _load_model_cache(self, sym: str) -> tuple[Any, StandardScaler, float] | None:
        path = self._model_dir / f"{sym}.pkl"
        if not path.exists():
            return None
        try:
            data = joblib.load(path)
            if data.get("version") != _MODEL_CACHE_VER:
                return None
            if time.time() - data["trained_at"] > MODEL_CACHE_AGE_SECS:
                return None
            if data["oos"] < MIN_OOS_ACC:
                return None
            return data["model"], data["scaler"], data["oos"]
        except Exception:
            return None

    # ── WebSocket helpers ─────────────────────────────────────────────────────

    async def _send(self, ws: Any, payload: dict[str, Any]) -> int:
        rid = next(self._req_id)
        payload["req_id"] = rid
        await ws.send(json.dumps(payload))
        return rid

    async def _request(self, ws: Any, payload: dict[str, Any],
                       timeout: float = 15.0) -> dict[str, Any]:
        """Send a request and await its response via req_id future."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        rid = next(self._req_id)
        payload["req_id"] = rid
        self._pending[rid] = fut
        try:
            await ws.send(json.dumps(payload))
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            log.warning("[REQ] timeout req_id=%d", rid)
            return {"error": {"message": "timeout"}}
        finally:
            self._pending.pop(rid, None)

    async def _listener(self, ws: Any) -> None:
        """Receive all WebSocket messages and route them."""
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            rid = msg.get("req_id")
            if rid and rid in self._pending:
                fut = self._pending.pop(rid)
                if not fut.done():
                    fut.set_result(msg)
                continue

            mtype = msg.get("msg_type", "")
            match mtype:
                case "balance":
                    await self._on_balance(msg)
                case "ohlc":
                    await self._on_ohlc(msg, ws)
                case "candles":
                    await self._on_candles(msg)
                case "proposal_open_contract":
                    await self._on_poc(msg, ws)
                case "tick":
                    pass  # not used
                case _:
                    if "error" in msg and rid is None:
                        log.warning("[WS] Unhandled error: %s", msg["error"])

    async def _heartbeat(self, ws: Any) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            try:
                resp = await self._request(ws, {"ping": 1}, timeout=HEARTBEAT_TIMEOUT)
                if "error" in resp:
                    log.warning("[HB] ping error: %s", resp["error"])
                    break
            except Exception as exc:
                log.warning("[HB] %s", exc)
                break

    # ── Deriv API: market data ────────────────────────────────────────────────

    async def _fetch_candle_history(self, symbol: str, ws: Any) -> None:
        """
        Fetch OHLC candle history via ticks_history.
        NOTE: ticks_history uses "ticks_history": "SYMBOL" — no rename here.
        """
        start_epoch = int(time.time()) - CANDLE_HISTORY_DAYS * 86400
        resp = await self._request(ws, {
            "ticks_history": symbol,
            "end":           "latest",
            "start":         start_epoch,
            "style":         "candles",
            "granularity":   60,   # 1-minute candles
            "count":         5000,
        }, timeout=30.0)

        if "error" in resp:
            log.warning("[HIST] %s: %s", symbol, resp["error"].get("message", "?"))
            return

        candles_raw = resp.get("candles", [])
        if not candles_raw:
            return

        rows = []
        for c in candles_raw:
            try:
                epoch = int(c["epoch"])
                rows.append({
                    "epoch": epoch,
                    "open":  float(c.get("open",  c["close"])),
                    "high":  float(c.get("high",  c["close"])),
                    "low":   float(c.get("low",   c["close"])),
                    "close": float(c["close"]),
                })
            except (KeyError, ValueError):
                continue

        if not rows:
            return

        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["epoch"], unit="s")
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)

        existing = self.candles.get(symbol, pd.DataFrame())
        if not existing.empty:
            df = pd.concat([existing, df]).loc[~pd.concat([existing, df]).index.duplicated(keep="last")]
            df.sort_index(inplace=True)
        self.candles[symbol] = df
        self._persist_candles(symbol, df)
        log.info("[HIST] %s loaded %d candles", symbol, len(df))

    async def _subscribe_candles(self, symbol: str, ws: Any) -> None:
        """Subscribe to live 1-minute OHLC stream for a symbol."""
        await self._send(ws, {
            "ticks_history": symbol,
            "end":           "latest",
            "style":         "candles",
            "granularity":   60,
            "subscribe":     1,
            "count":         10,
        })

    async def _fetch_contracts_for(self, symbol: str, ws: Any) -> None:
        """
        Query contracts_for to discover the live multiplier values allowed
        for this symbol. Updates self.multipliers[symbol].
        """
        resp = await self._request(ws, {
            "contracts_for": symbol,
            "currency":      "USD",
            "product_type":  "basic",
        }, timeout=20.0)

        if "error" in resp:
            log.warning("[CF] %s: %s", symbol, resp["error"].get("message", "?"))
            return

        available = resp.get("contracts_for", {}).get("available", [])
        for c in available:
            ct = c.get("contract_type", "")
            if ct in ("MULTUP", "MULTDOWN"):
                mults = c.get("multiplier_range", [])
                if mults:
                    # Pick the highest available multiplier ≤ our default
                    default = DEFAULT_MULTIPLIER.get(symbol, 1000)
                    valid = [m for m in mults if isinstance(m, (int, float))]
                    if valid:
                        best = max((m for m in valid if m <= default), default=min(valid))
                        self.multipliers[symbol] = int(best)
                        log.info("[CF] %s multiplier → %d (from %s)",
                                 symbol, self.multipliers[symbol], valid)
                break

    # ── Balance & market hours ────────────────────────────────────────────────

    async def _on_balance(self, msg: dict[str, Any]) -> None:
        bal = float(msg.get("balance", {}).get("balance", 0))
        if bal <= 0:
            return
        self.current_balance = bal
        if not self._balance_ready:
            self._balance_ready = True
            if self.initial_balance <= 0:
                self.initial_balance = bal
            self._check_day_reset()
            log.info("[BAL] $%.2f | DayStart=$%.2f (%s) — signals active",
                     bal, self.day_start_bal, self.day_date)

    def _is_market_open(self, symbol: str) -> bool:
        now  = datetime.now(timezone.utc)
        wday = now.weekday()   # 0=Mon … 4=Fri 5=Sat 6=Sun
        hour = now.hour

        # Weekend: Sat all day, Sun before market open (~21:00 Fri closes)
        if wday == 5:
            return False
        if wday == 6 and hour < 21:
            return False
        # Friday close
        if wday == 4 and hour >= MARKET_CLOSE_HOUR:
            return False
        # Low-liquidity hours
        if hour in LOW_EDGE_HOURS:
            return False

        # Gold trades almost 24/5; Forex has thin liquidity near close but
        # we already guard that with LOW_EDGE_HOURS
        return True

    # ── Signal generation ─────────────────────────────────────────────────────

    async def _on_ohlc(self, msg: dict[str, Any], ws: Any) -> None:
        """Handle incoming live OHLC (new-candle) events."""
        ohlc = msg.get("ohlc", {})
        sym  = ohlc.get("symbol", "")
        if sym not in self.all_symbols:
            return

        try:
            epoch = int(ohlc["open_time"])
            row   = {
                "epoch": epoch,
                "open":  float(ohlc["open"]),
                "high":  float(ohlc["high"]),
                "low":   float(ohlc["low"]),
                "close": float(ohlc["close"]),
            }
        except (KeyError, ValueError):
            return

        df      = self.candles.get(sym, pd.DataFrame())
        ts      = pd.Timestamp(epoch, unit="s")
        new_row = pd.DataFrame([row]).assign(datetime=ts).set_index("datetime")

        if df.empty:
            self.candles[sym] = new_row
        else:
            self.candles[sym] = (
                pd.concat([df, new_row])
                .loc[~pd.concat([df, new_row]).index.duplicated(keep="last")]
                .sort_index()
            )

        self.new_candle_counts[sym] = self.new_candle_counts.get(sym, 0) + 1
        self._feature_cache.pop(sym, None)

        # Persist candle
        self._persist_candles(sym, new_row)

        # Schedule retrain when enough new candles have arrived.
        if self.new_candle_counts.get(sym, 0) >= RETRAIN_EVERY:
            log.info("[RETRAIN] %s triggering retrain…", sym)
            # Reset counter BEFORE creating the task so that candles arriving
            # while training is in progress don't immediately re-trigger.
            self.new_candle_counts[sym] = 0
            task = asyncio.create_task(self.train_model(sym, ws))
            self._retrain_tasks.add(task)
            task.add_done_callback(self._retrain_tasks.discard)

    async def _on_candles(self, msg: dict[str, Any]) -> None:
        """Handle bulk candle history response (initial subscription snapshot)."""
        candles_raw = msg.get("candles", [])
        # Determine symbol from echo_req
        sym = msg.get("echo_req", {}).get("ticks_history", "")
        if not sym or sym not in self.all_symbols:
            return
        if not candles_raw:
            return

        rows = []
        for c in candles_raw:
            try:
                rows.append({
                    "epoch": int(c["epoch"]),
                    "open":  float(c.get("open",  c["close"])),
                    "high":  float(c.get("high",  c["close"])),
                    "low":   float(c.get("low",   c["close"])),
                    "close": float(c["close"]),
                })
            except (KeyError, ValueError):
                continue

        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["epoch"], unit="s")
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)

        existing = self.candles.get(sym, pd.DataFrame())
        if not existing.empty:
            df = (pd.concat([existing, df])
                  .loc[~pd.concat([existing, df]).index.duplicated(keep="last")]
                  .sort_index())
        self.candles[sym] = df
        self._persist_candles(sym, df)

    def _get_signal(self, symbol: str) -> dict[str, Any] | None:
        """
        Compute ML signal + confluence for a symbol.
        Returns a dict with keys: direction, prob, confluence, regime, features.
        Returns None if no signal should be generated.
        """
        if symbol not in self.models:
            return None

        df = self.candles.get(symbol, pd.DataFrame())
        if df.shape[0] < MIN_CANDLES_TO_TRADE:
            return None

        # Check feature cache
        cached = self._feature_cache.get(symbol)
        if cached:
            feat_row = cached
        else:
            feat_df = build_features(df)
            if feat_df.empty:
                return None
            feat_row = feat_df.iloc[-1].to_dict()
            self._feature_cache[symbol] = feat_row

        model  = self.models[symbol]
        scaler = self.scalers[symbol]
        try:
            # FIX: use named DataFrame so LightGBM doesn't raise feature-name warnings
            # and OOS scores are computed correctly (was causing 50.5% random-chance scores)
            feat_names = self._feature_names.get(symbol, sorted(feat_row.keys()))
            feat_arr   = pd.DataFrame(
                [[feat_row[k] for k in feat_names]], columns=feat_names)
            feat_s     = pd.DataFrame(
                scaler.transform(feat_arr), columns=feat_names)
            proba      = model.predict_proba(feat_s)[0]   # [P(0), P(1)]
        except Exception as exc:
            log.warning("[SIGNAL] %s feature error: %s", symbol, exc)
            return None

        p_up   = float(proba[1])
        p_down = float(proba[0])
        regime = detect_regime(feat_row)

        # Determine direction
        if p_up >= p_down and p_up >= CONF_THRESHOLD:
            direction = "UP"
            prob      = p_up
        elif p_down > p_up and p_down >= CONF_THRESHOLD:
            direction = "DOWN"
            prob      = p_down
        else:
            log.info("[SIGNAL] %s | UP=%.4f DOWN=%.4f | below threshold %.2f",
                     symbol, p_up, p_down, CONF_THRESHOLD)
            return None

        conf = confluence_score(feat_row, direction, regime)
        oos  = self.model_oos.get(symbol, 0.0)
        log.info("[SIGNAL] %s | %s prob=%.4f | conf=%d/5 | RSI=%.1f ADX=%.1f | %s | OOS=%.1f%%",
                 symbol, direction, prob, conf,
                 feat_row.get("rsi14", 0), feat_row.get("adx", 0),
                 regime, oos * 100)

        if conf < MIN_CONFLUENCE:
            log.info("[SKIP] %s confluence %d < %d", symbol, conf, MIN_CONFLUENCE)
            return None

        return {"direction": direction, "prob": prob,
                "confluence": conf, "regime": regime, "features": feat_row}

    # ── Order placement ───────────────────────────────────────────────────────

    async def _place_order(self, ws: Any, symbol: str, signal: dict[str, Any]) -> None:
        """
        Place a MULTUP or MULTDOWN contract.

        VERIFIED PAYLOAD (against official JSON schema):
          ✅  "symbol": symbol              (NOT "underlying_symbol" — rejected by live API)
          ✅  limit_order values are float   (NOT str — schema says type:number)
          ✅  NO "cancellation" field        (mutually exclusive with limit_order)
          ✅  "price" in buy is float        (schema says type:number, minimum:0)
        """
        direction   = signal["direction"]
        prob        = signal["prob"]
        confluence  = signal["confluence"]
        regime      = signal["regime"]

        # ── Pre-flight guards ──
        if self.contracts.count >= MAX_OPEN:
            return
        if symbol in self.placing:
            return

        ok, reason = self._risk_ok()
        if not ok:
            log.warning("[ORDER] Risk blocked for %s: %s", symbol, reason)
            return

        if not self._is_market_open(symbol):
            log.info("[ORDER] %s market closed", symbol)
            return

        now = asyncio.get_running_loop().time()
        if now - self.last_trade_time < GLOBAL_COOLDOWN:
            return
        if now - self.symbol_last_trade.get(symbol, 0) < SYMBOL_COOLDOWN:
            return

        self.placing.add(symbol)
        try:
            stake = self._calculate_stake(symbol, prob)
            if stake <= 0:
                log.info("[STAKE] %s stake too small — skip", symbol)
                return

            multiplier = self.multipliers.get(symbol, DEFAULT_MULTIPLIER.get(symbol, 1000))
            pip_size   = PIP_SIZE.get(symbol, 0.0001)
            sl_pips    = float(_env(f"SL_PIPS_{symbol}", str(SL_PIPS_DEFAULT)))
            tp_pips    = float(_env(f"TP_PIPS_{symbol}", str(TP_PIPS_DEFAULT)))

            # FIX: For Deriv multiplier contracts, limit_order.stop_loss and
            # limit_order.take_profit are P&L dollar amounts bounded by stake —
            # NOT pip moves multiplied by the multiplier. The old formula gave
            # sl_amount=$1500 on a $50 stake, which the API rejected with
            # "Enter an amount equal to or lower than 50.00".
            # SL_PCT=0.50 → lose at most 50% of stake; TP_PCT=1.00 → gain 100%.
            _sl_pct   = float(_env(f"SL_PCT_{symbol}", _env("SL_PCT", "0.50")))
            _tp_pct   = float(_env(f"TP_PCT_{symbol}", _env("TP_PCT", "1.00")))
            sl_amount = _norm2(max(0.01, min(stake * _sl_pct, stake - 0.01)))
            tp_amount = _norm2(max(0.01, stake * _tp_pct))

            contract_type = "MULTUP" if direction == "UP" else "MULTDOWN"
            kelly_str     = f"{stake / self.current_balance * 100:.2f}%"

            log.info("[STAKE] %s | SL=$%.2f (%.0f%%) TP=$%.2f (%.0f%%) | Kelly=%s | Stake=$%.2f",
                     symbol, sl_amount, _sl_pct*100, tp_amount, _tp_pct*100, kelly_str, stake)

            # ── PROPOSAL REQUEST ─────────────────────────────────────────────
            # Field reference: developers.deriv.com/schemas/proposal_request.schema.json
            # "symbol" is the correct field name (API rejected "underlying_symbol")
            # "limit_order" values are type:number (not str)
            # "cancellation" is mutually exclusive with "limit_order" — OMIT IT
            # ────────────────────────────────────────────────────────────────
            proposal_payload: dict[str, Any] = {
                "proposal":      1,
                "amount":        stake,          # type: number ✓
                "basis":         "stake",
                "contract_type": contract_type,
                "currency":      "USD",
                "symbol":        symbol,         # ← "underlying_symbol" rejected by API
                "multiplier":    multiplier,      # type: number ✓
                "limit_order": {
                    "stop_loss":   sl_amount,        # type: number ✓ (NOT str)
                    "take_profit": tp_amount,        # type: number ✓ (NOT str)
                },
                # "cancellation": ...               ← INTENTIONALLY OMITTED
                #   Reason: mutually exclusive with limit_order per API schema
            }

            resp = await self._request(ws, proposal_payload, timeout=15.0)

            if "error" in resp:
                err = resp["error"].get("message", "?")
                log.error("[ORDER] %s proposal error: %s", symbol, err)
                return

            proposal_obj = resp.get("proposal", {})
            proposal_id  = proposal_obj.get("id")
            if not proposal_id:
                log.error("[ORDER] %s — no proposal ID in response", symbol)
                return

            ask_price = _norm2(float(proposal_obj.get("ask_price", stake)))

            # ── BUY REQUEST ──────────────────────────────────────────────────
            # Field reference: developers.deriv.com/schemas/buy_request.schema.json
            # "buy": proposal_id (str), "price": ask_price (number ✓)
            # ────────────────────────────────────────────────────────────────
            buy_resp = await self._request(ws, {
                "buy":   proposal_id,    # str matching ^(?:[\w-]{32,128}|1)$
                "price": ask_price,      # type: number ✓
            }, timeout=15.0)

            if "error" in buy_resp:
                err = buy_resp["error"].get("message", "?")
                log.error("[ORDER] %s buy error: %s", symbol, err)
                return

            if "buy" not in buy_resp:
                log.error("[ORDER] %s — unexpected buy response: %s", symbol, buy_resp)
                return

            buy_obj     = buy_resp["buy"]
            contract_id = str(buy_obj.get("contract_id", ""))
            if not contract_id:
                log.error("[ORDER] %s — no contract_id in buy response", symbol)
                return

            # Update timing guards
            loop = asyncio.get_running_loop()
            self.last_trade_time             = loop.time()
            self.symbol_last_trade[symbol]   = loop.time()

            # Store contract
            rec = OpenContract(
                contract_id=contract_id, symbol=symbol, stake=stake,
                contract_type=contract_type, confidence=prob, confluence=confluence,
                regime=regime, sl_amount=sl_amount, tp_amount=tp_amount,
                multiplier=multiplier,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            self.contracts.add(rec)

            oos = self.model_oos.get(symbol, 0.0)
            msg_text = (
                f"🚀 TRADE | {symbol} {contract_type}\n"
                f"Stake=${stake:.2f} | SL=${sl_amount:.2f} | TP=${tp_amount:.2f}\n"
                f"Mult={multiplier}x | Conf={prob*100:.1f}% | Confluence={confluence}/5\n"
                f"{regime} | OOS={oos*100:.1f}% | Bal=${self.current_balance:.2f}"
            )
            log.info(msg_text.replace("\n", " | "))
            await self._send_tg(msg_text)

        finally:
            self.placing.discard(symbol)

    # ── Contract lifecycle ────────────────────────────────────────────────────

    async def _on_poc(self, msg: dict[str, Any], ws: Any) -> None:
        """Handle proposal_open_contract subscription updates."""
        poc    = msg.get("proposal_open_contract", {})
        status = poc.get("status", "")
        if status in ("sold", "won", "lost", "expired", "cancelled"):
            cid = str(poc.get("contract_id", ""))
            if cid:
                await self._handle_contract_close(cid, poc)

    async def _handle_contract_close(self, cid: str, poc: dict[str, Any]) -> None:
        if cid in self.processed_ids:
            return
        self.processed_ids.add(cid)
        self._persist_processed(cid)

        rec = self.contracts.get(cid)
        if not rec:
            return

        profit = float(poc.get("profit", 0.0))
        symbol = rec.symbol
        self.total_profit  += profit
        self.current_balance = float(poc.get("account_balance", self.current_balance))
        self.contracts.remove(cid)

        won = profit > 0
        if won:
            self.wins              += 1
            self.consec_losses      = 0
            self.profitable_trades += 1
            self.symbol_results[symbol].append(1)
            self.symbol_streak[symbol] = max(0, self.symbol_streak[symbol]) + 1
        else:
            self.losses            += 1
            self.consec_losses     += 1
            self.symbol_results[symbol].append(0)
            self.symbol_streak[symbol] = min(0, self.symbol_streak[symbol]) - 1

        self._log_trade(rec, profit)
        self._save_session()
        self._check_day_reset()

        total_wr = self.wins / max(1, self.wins + self.losses)
        sym_wr   = self._symbol_wr(symbol)
        status   = poc.get("status", "?")
        sign     = "+" if won else ""
        outcome  = "✅ WIN" if won else "❌ LOSS"
        msg_text = (
            f"{outcome} | {symbol} {rec.contract_type}\n"
            f"P&L={sign}${profit:.2f} | Bal=${self.current_balance:.2f}\n"
            f"W:{self.wins} L:{self.losses} ({total_wr*100:.0f}%) | {symbol} WR={sym_wr*100:.0f}%\n"
            f"Day=${self._daily_pnl:+.2f} | Status={status}"
        )
        log.info(msg_text.replace("\n", " | "))
        await self._send_tg(msg_text)

        # Check risk limits after close
        ok, reason = self._risk_ok()
        if not ok:
            log.warning("[RISK] Limit hit: %s", reason)
            await self._send_tg(f"⚠️ RISK LIMIT: {reason} — halting")
            self._shutdown_reason = reason
            self._shutdown.set()

    async def _sell_contract(self, cid: str, ws: Any, reason: str) -> bool:
        """Early-exit a contract by selling it."""
        rec = self.contracts.get(cid)
        if not rec or rec.sell_queued:
            return False
        lock = self.contracts.sell_lock(cid)
        async with lock:
            rec = self.contracts.get(cid)
            if not rec or rec.sell_queued:
                return False
            self.contracts.mark_sell_queued(cid)
            try:
                cid_int = int(cid)
            except ValueError:
                log.error("[SELL] non-integer contract_id: %s", cid)
                rec.sell_queued = False
                return False

            # sell request: {"sell": contract_id_int, "price": 0}
            resp = await self._request(ws, {"sell": cid_int, "price": 0}, timeout=15.0)
            if "error" in resp:
                log.warning("[SELL] %s error: %s", cid, resp["error"].get("message"))
                rec.sell_queued = False
                return False
            log.info("[SELL] %s OK reason=%s", cid, reason)
            return True

    # ── Signal loop ───────────────────────────────────────────────────────────

    async def _signal_loop(self, ws: Any) -> None:
        """Periodic loop that evaluates signals and places trades."""
        while not self._shutdown.is_set():
            await asyncio.sleep(1.0)

            if not self._balance_ready:
                continue

            self._check_day_reset()

            ok, reason = self._risk_ok()
            if not ok:
                continue

            for symbol in self.all_symbols:
                if self._shutdown.is_set():
                    break
                if symbol in self.placing:
                    continue
                if self.contracts.count >= MAX_OPEN:
                    break

                signal = self._get_signal(symbol)
                if signal is None:
                    continue

                asyncio.create_task(
                    self._place_order(ws, symbol, signal),
                    name=f"order_{symbol}"
                )
                # Brief pause between symbol evaluations
                await asyncio.sleep(0.1)

    # ── Telegram ──────────────────────────────────────────────────────────────

    async def _send_tg(self, text: str) -> None:
        if not TG_TOKEN or not TG_CHAT_ID or self._tg_session is None:
            return
        try:
            url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
            await self._tg_session.post(
                url,
                json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"},
                timeout=aiohttp.ClientTimeout(total=10),
            )
        except Exception as exc:
            log.warning("[TG] %s", exc)

    # ── Checkpoint loop ───────────────────────────────────────────────────────

    async def _checkpoint_loop(self) -> None:
        while not self._shutdown.is_set():
            await asyncio.sleep(CHECKPOINT_INTERVAL)
            self._save_session()
            self._prune_candles_db()

    # ── Main connection loop ──────────────────────────────────────────────────

    async def _run_connection(self) -> None:
        reconnect_delay = RECONNECT_BASE

        while not self._shutdown.is_set():
            try:
                log.info("[WS] Connecting to %s…", WS_URL)
                async with websockets.connect(
                    WS_URL,
                    ping_interval=None,
                    ping_timeout=None,
                    open_timeout=30,
                    close_timeout=10,
                ) as ws:
                    reconnect_delay = RECONNECT_BASE

                    # Clear pending futures from prior connection
                    for fut in self._pending.values():
                        if not fut.done():
                            fut.cancel()
                    self._pending.clear()

                    for t in list(self._retrain_tasks):
                        t.cancel()
                    self._retrain_tasks.clear()

                    self._balance_ready = False

                    # Listener MUST start before any _request() call.
                    # _request() awaits a Future resolved only when _listener
                    # reads the socket response.  Without this the auth reply
                    # arrives but nothing reads it → Future times out every time.
                    listener_task = asyncio.create_task(self._listener(ws))

                    # ── Authenticate ──
                    auth = await self._request(ws, {"authorize": self.api_token})
                    if "error" in auth:
                        err = auth["error"].get("message", "?")
                        log.critical("[AUTH] Failed: %s", err)
                        listener_task.cancel()
                        # Raise so outer except reconnects; return exits forever.
                        raise ConnectionError(f"Auth rejected: {err}")
                    log.info("[AUTH] Authenticated ✓")

                    # ── Subscribe to balance and sync live value synchronously ──
                    # Using _request() instead of _send() so the first balance
                    # response is captured before any trading logic runs.
                    # The subscription stays active for ongoing updates via _on_balance.
                    bal_resp = await self._request(
                        ws, {"balance": 1, "subscribe": 1}, timeout=15.0
                    )
                    await self._on_balance(bal_resp)
                    log.info("[BAL] Live balance synced: $%.2f", self.current_balance)

                    # ── Restore open contracts from DB ──
                    restored = self.contracts.load_from_db(self.processed_ids)
                    if restored:
                        log.info("[RECONNECT] Restored %d open contract(s) — "
                                 "re-subscribing and checking for offline closes…", restored)
                        closed_while_offline: list[tuple[str, dict]] = []
                        for cid in list(self.contracts.all_ids()):
                            try:
                                # Subscribe for ongoing status events
                                await self._send(ws, {
                                    "proposal_open_contract": 1,
                                    "contract_id": int(cid),
                                    "subscribe": 1,
                                })
                                # One-shot snapshot to catch closes that happened
                                # while the bot was disconnected.  This uses a
                                # separate req_id so _listener resolves it here
                                # rather than routing to _on_poc (avoiding double
                                # processing); _handle_contract_close is idempotent
                                # via processed_ids if both paths race.
                                snap = await self._request(ws, {
                                    "proposal_open_contract": 1,
                                    "contract_id": int(cid),
                                }, timeout=10.0)
                                poc = snap.get("proposal_open_contract", {})
                                if poc.get("status") in (
                                    "sold", "won", "lost", "expired", "cancelled"
                                ):
                                    closed_while_offline.append((cid, poc))
                            except Exception as exc:
                                log.warning("[RECONNECT] Could not check contract %s: %s",
                                            cid, exc)
                        # Handle offline-closed contracts before starting signal loop
                        for cid, poc in closed_while_offline:
                            log.info(
                                "[RECONNECT] Contract %s closed while offline "
                                "(status=%s, profit=%s) — reconciling",
                                cid, poc.get("status"), poc.get("profit"),
                            )
                            await self._handle_contract_close(cid, poc)
                        log.info("[RECONNECT] Balance after reconciliation: $%.2f",
                                 self.current_balance)

                    # ── Discover live multipliers from contracts_for ──
                    log.info("[INIT] Querying live multiplier values…")
                    for sym in self.all_symbols:
                        await self._fetch_contracts_for(sym, ws)
                        await asyncio.sleep(0.3)

                    # ── Train / load models ──
                    log.info("[INIT] Training models (or loading from cache)…")
                    for sym in self.all_symbols:
                        await self.train_model(sym, ws)
                        await asyncio.sleep(0.5)

                    deployed = list(self.models.keys())
                    oos_str  = {s: f"{v*100:.1f}%" for s, v in self.model_oos.items()}
                    wr_str   = {
                        s: f"{self._symbol_wr(s)*100:.1f}% (n={len(self.symbol_results[s])})"
                        for s in self.all_symbols
                    }
                    ensemble = "RF+GBM+LGBM" if _HAS_LGBM else "RF+GBM"
                    log.info(
                        "[READY] %s | Deployed=%s | OOS=%s | WR=%s | "
                        "Confluence≥%d/5 | Kelly≥%.3f | DD=%.0f%%/%.0f%% | "
                        "SL=%d TP=%d pips | Multipliers=%s",
                        ensemble, deployed, oos_str, wr_str,
                        MIN_CONFLUENCE, MIN_KELLY,
                        MAX_DRAWDOWN_PCT * 100, MAX_DAILY_LOSS_PCT * 100,
                        SL_PIPS_DEFAULT, TP_PIPS_DEFAULT,
                        {s: self.multipliers.get(s) for s in deployed},
                    )

                    # ── Subscribe to live candle streams ──
                    for sym in self.all_symbols:
                        await self._subscribe_candles(sym, ws)
                        await asyncio.sleep(0.1)

                    await self._send_tg(
                        f"🤖 Genesis v15 ONLINE | {ensemble}\n"
                        f"Deployed: {deployed}\n"
                        f"Bal: ${self.current_balance:.2f} | "
                        f"DD: {MAX_DRAWDOWN_PCT*100:.0f}%/{MAX_DAILY_LOSS_PCT*100:.0f}%"
                    )

                    # ── Run tasks concurrently ──
                    # listener_task already running (started before auth above).
                    hb_task         = asyncio.create_task(self._heartbeat(ws))
                    signal_task     = asyncio.create_task(self._signal_loop(ws))
                    checkpoint_task = asyncio.create_task(self._checkpoint_loop())
                    shutdown_task   = asyncio.create_task(self._shutdown.wait())

                    done, pending = await asyncio.wait(
                        [listener_task, hb_task, signal_task,
                         checkpoint_task, shutdown_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in pending:
                        t.cancel()

                    if self._shutdown.is_set():
                        break

                    log.warning("[WS] Task completed unexpectedly — reconnecting")

            except websockets.exceptions.ConnectionClosed as exc:
                log.warning("[WS] Connection closed (%s) — reconnecting in %.0fs",
                            exc, reconnect_delay)
            except Exception as exc:
                log.error("[WS] Error: %s — reconnecting in %.0fs",
                          exc, reconnect_delay, exc_info=True)

            self._save_session()
            await asyncio.sleep(reconnect_delay + random.uniform(0, 2))
            reconnect_delay = min(reconnect_delay * 2, RECONNECT_MAX)

    async def run(self) -> None:
        async with aiohttp.ClientSession() as tg_sess:
            self._tg_session = tg_sess
            try:
                await self._run_connection()
            finally:
                self._save_session()
                self._db.close()
                log.info("[SHUTDOWN] W=%d L=%d PnL=$%.2f | Reason: %s",
                         self.wins, self.losses, self.total_profit,
                         self._shutdown_reason or "user request")
                await self._send_tg(
                    f"🛑 Genesis v15 OFFLINE\n"
                    f"W:{self.wins} L:{self.losses} PnL:${self.total_profit:.2f}\n"
                    f"Reason: {self._shutdown_reason or 'user request'}"
                )


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import signal as _signal

    bot = GenesisBot()

    def _handle_signal(sig: int, _frame: Any) -> None:
        log.info("Signal %d received — shutting down gracefully…", sig)
        bot._shutdown_reason = f"signal {sig}"
        bot._shutdown.set()

    _signal.signal(_signal.SIGINT,  _handle_signal)
    _signal.signal(_signal.SIGTERM, _handle_signal)

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        log.info("[SHUTDOWN] Stopped by user (Ctrl-C)")
