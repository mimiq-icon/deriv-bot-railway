# Genesis v14 — Termux Deployment Guide
# ========================================

## 1. Initial Termux Setup (run once)

```bash
# Update packages
pkg update && pkg upgrade -y

# Install Python and build tools
pkg install python python-pip -y
pkg install libzmq freetype harfbuzz libpng -y   # needed for numpy/scipy

# Install git (optional, for version control)
pkg install git -y
```

## 2. Install Python Dependencies

```bash
cd ~/genesis   # or wherever you placed genesis_v14.py

# Option A: pip (standard)
pip install -r requirements.txt --break-system-packages

# Option B: if scikit-learn fails to compile (ARM build issues)
pkg install python-scikit-learn -y
pip install websockets aiohttp python-dotenv numpy pandas joblib --break-system-packages

# Option C: uv (faster)
pip install uv --break-system-packages
uv pip install -r requirements.txt
```

## 3. Configure the Bot

```bash
cp .env.example .env
nano .env   # Fill in DERIV_API_TOKEN and DERIV_APP_ID at minimum
```

## 4. Run the Bot

```bash
# Foreground (good for initial testing)
python genesis_v14.py

# Background with nohup (keeps running after terminal closes)
nohup python genesis_v14.py > logs/bot.log 2>&1 &
echo $! > bot.pid
echo "Bot PID: $(cat bot.pid)"

# To stop:
kill $(cat bot.pid)
```

## 5. Keep Running via Termux Wake Lock

In Termux, go to Notification → Wake Lock to prevent Android from killing the process.
Or use Termux:Boot to auto-start on phone reboot.

## 6. Monitor

```bash
# Watch live logs
tail -f logs/genesis_*.log

# Check SQLite stats
sqlite3 genesis_v14.db "SELECT COUNT(*), SUM(profit), AVG(profit) FROM trades;"
sqlite3 genesis_v14.db "SELECT symbol, COUNT(*), SUM(profit) FROM trades GROUP BY symbol;"
```

## Environment Variables Quick Reference

| Variable            | Default | Description                              |
|---------------------|---------|------------------------------------------|
| DERIV_API_TOKEN     | —       | **Required** — your Deriv API token      |
| DERIV_APP_ID        | 1089    | Deriv App ID (register at api.deriv.com) |
| TG_TOKEN            | —       | Telegram bot token (optional)            |
| TG_CHAT_ID          | —       | Telegram chat ID (optional)              |
| MAX_DAILY_LOSS_PCT  | 0.10    | Daily loss limit (fraction of balance)   |
| MAX_DRAWDOWN_PCT    | 0.50    | Total drawdown limit                     |
| STAKE_RISK_PCT      | 0.005   | Base stake risk per trade                |
| MAX_OPEN            | 2       | Max simultaneous open contracts          |
| MIN_CONFLUENCE      | 3       | Min signals out of 5 required            |
| SL_PIPS_DEFAULT     | 6       | Stop-loss in pips                        |
| TP_PIPS_DEFAULT     | 10      | Take-profit in pips                      |

## Important API Notes

The bot uses the **new Deriv API v4** field names:
- `underlying_symbol` (NOT `symbol`) in proposal requests
- `limit_order.stop_loss/take_profit` as **numbers** (NOT strings)
- **No** `cancellation` field — it's mutually exclusive with `limit_order`
- `price` in buy request is a **number** (NOT string)

## Troubleshooting

**"Cannot create contract"**
→ Usually means wrong payload. This bot has the correct payload per the official schema.

**"frxEURUSD ABORTED — minority X% < 5%"**
→ Normal — the training labels are too one-sided in the current market. The model will retry next retrain cycle.

**"OOS X% < 50.5%"**
→ Model didn't meet the accuracy bar. The bot skips this symbol and retries later.

**Termux killed the process**
→ Enable Termux:Boot, use a wake lock notification, or run on a VPS/Raspberry Pi instead.
# v15 clean deploy Fri May  8 16:18:24 CAT 2026
