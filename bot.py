#!/usr/bin/env python3
"""Crypto backtest bot. Needs: pip install ccxt pandas

Downloads OHLCV candles from a public exchange API (no key) and backtests a
long-only EMA crossover with an ATR stop, against buy and hold.

    python bot.py backtest --symbol BTC/USDT --timeframe 1h --days 180
    python bot.py backtest --symbol ETH/USDT --timeframe 4h --days 365 --fast 12 --slow 26
    python bot.py backtest --exchange kraken --symbol BTC/USD --trades
    python bot.py backtest --csv candles.csv          # offline: timestamp,open,high,low,close,volume

The rules (no look-ahead):
  * Signal on a closed bar: fast EMA crosses above the slow EMA -> buy,
    crosses below -> sell. Fills happen at the NEXT bar's open.
  * Stop: entry - atr_mult * ATR(14) at entry. Checked on each bar's low;
    a gap below the stop fills at that bar's open (the worse price).
  * Fees: --fee per side (default 0.1 %), applied on entry and exit.
    --slippage (default 0.02 %) moves every fill against you.
  * All in, one position at a time, long only. Any open trade is closed at
    the last bar's close so the totals are realised.

Not financial advice. Past results do not predict future ones, and this bot
never places an order: it only reads public candles.
"""

import argparse
import math
import os
import sys
import time

try:
    import ccxt
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    sys.exit("Missing dependency (%s). Run: pip install ccxt pandas" % exc.name)


# Binance's main API answers 451 from some regions (and many cloud hosts);
# its public market-data mirror serves the same candles without that block.
BINANCE_DATA_MIRROR = "https://data-api.binance.vision/api/v3"


def make_exchange(name):
    if not hasattr(ccxt, name):
        sys.exit("Unknown exchange %r. Try binance, kraken, bybit, okx, coinbase." % name)
    cfg = {"enableRateLimit": True}
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy:  # ccxt ignores the proxy environment variables unless told
        cfg["httpsProxy"] = proxy
    ex = getattr(ccxt, name)(cfg)
    if name == "binance":
        ex.urls["api"]["public"] = BINANCE_DATA_MIRROR
        ex.options["fetchMarkets"] = ["spot"]
    return ex


def fetch_candles(ex, symbol, timeframe, days):
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    now = ex.milliseconds()
    since = now - days * 86400 * 1000
    rows = []
    while since < now:
        for attempt in range(4):
            try:
                batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                break
            except ccxt.NetworkError:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][0] + tf_ms
        if nxt <= since:
            break
        since = nxt
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts")
    df = df[df["ts"] + tf_ms <= now]  # drop the still-forming candle
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.drop(columns="ts")


def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    t = df[col]
    df.index = pd.to_datetime(t, unit="ms", utc=True) if t.dtype.kind in "if" else pd.to_datetime(t, utc=True)
    return df[["open", "high", "low", "close", "volume"]].astype(float).sort_index()


def add_indicators(df, fast, slow, atr_len=14):
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1.0 / atr_len, adjust=False).mean()
    above = df["ema_fast"] > df["ema_slow"]
    df["signal"] = 0
    df.loc[above & ~above.shift(1, fill_value=False), "signal"] = 1
    df.loc[~above & above.shift(1, fill_value=False), "signal"] = -1
    df.iloc[:slow, df.columns.get_loc("signal")] = 0  # EMAs not settled yet
    return df


def backtest(df, cash=10000.0, fee=0.001, slippage=0.0002, atr_mult=2.0):
    equity_curve = []
    trades = []
    qty = 0.0
    entry = stop = None
    entry_time = None
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    sig, atr = df["signal"].values, df["atr"].values
    idx = df.index

    def close_pos(i, price, reason):
        nonlocal cash, qty, entry, stop
        fill = price * (1 - slippage)
        cash = qty * fill * (1 - fee)
        cost = qty * entry / (1 - fee)
        trades.append({"entry_time": entry_time, "exit_time": idx[i], "entry": entry,
                       "exit": fill, "pnl": cash - cost, "ret": cash / cost - 1, "reason": reason})
        qty, entry, stop = 0.0, None, None

    for i in range(len(df)):
        if i > 0:
            # act on the previous bar's signal at this bar's open
            if qty == 0 and sig[i - 1] == 1:
                entry = o[i] * (1 + slippage)
                qty = cash * (1 - fee) / entry
                cash = 0.0
                stop = entry - atr_mult * atr[i - 1]
                entry_time = idx[i]
            elif qty > 0 and sig[i - 1] == -1:
                close_pos(i, o[i], "signal")
        if qty > 0 and l[i] <= stop:
            close_pos(i, min(o[i], stop), "stop")
        equity_curve.append(cash + qty * c[i])
    if qty > 0:
        close_pos(len(df) - 1, c[-1], "end")
        equity_curve[-1] = cash
    return pd.Series(equity_curve, index=idx), pd.DataFrame(trades)


def max_drawdown(eq):
    return float((eq / eq.cummax() - 1).min())


def sharpe(eq, bars_per_year):
    r = eq.pct_change().dropna()
    if r.std() == 0 or len(r) < 2:
        return 0.0
    return float(r.mean() / r.std() * math.sqrt(bars_per_year))


def report(df, eq, trades, cash, symbol, timeframe, tf_seconds, show_trades):
    bars_per_year = 365 * 86400 / tf_seconds
    bh = cash * df["close"] / df["open"].iloc[0]
    years = (df.index[-1] - df.index[0]).total_seconds() / (365 * 86400)

    def cagr(final):
        return (final / cash) ** (1 / years) - 1 if years > 0 and final > 0 else float("nan")

    n = len(trades)
    wins = trades[trades["pnl"] > 0] if n else trades
    losses = trades[trades["pnl"] <= 0] if n else trades
    gross_loss = -losses["pnl"].sum() if n else 0.0
    pf = wins["pnl"].sum() / gross_loss if n and gross_loss > 0 else float("inf")

    print("Backtest  %s  %s  %s -> %s  (%d bars)" % (
        symbol, timeframe, df.index[0].strftime("%Y-%m-%d %H:%M"),
        df.index[-1].strftime("%Y-%m-%d %H:%M"), len(df)))
    print("-" * 62)
    print("%-22s %16s %16s" % ("", "Strategy", "Buy & hold"))
    print("%-22s %16s %16s" % ("Final equity", "$%s" % format(eq.iloc[-1], ",.2f"),
                               "$%s" % format(bh.iloc[-1], ",.2f")))
    print("%-22s %15.2f%% %15.2f%%" % ("Total return", (eq.iloc[-1] / cash - 1) * 100,
                                       (bh.iloc[-1] / cash - 1) * 100))
    print("%-22s %15.2f%% %15.2f%%" % ("CAGR", cagr(eq.iloc[-1]) * 100, cagr(bh.iloc[-1]) * 100))
    print("%-22s %15.2f%% %15.2f%%" % ("Max drawdown", max_drawdown(eq) * 100, max_drawdown(bh) * 100))
    print("%-22s %16.2f %16.2f" % ("Sharpe (annualised)", sharpe(eq, bars_per_year), sharpe(bh, bars_per_year)))
    print("-" * 62)
    print("Trades: %d   Win rate: %s   Profit factor: %s" % (
        n, "%.1f%%" % (len(wins) / n * 100) if n else "n/a",
        "%.2f" % pf if n and math.isfinite(pf) else ("inf" if n else "n/a")))
    if n:
        print("Avg win: %+.2f%%   Avg loss: %+.2f%%   Best: %+.2f%%   Worst: %+.2f%%" % (
            wins["ret"].mean() * 100 if len(wins) else 0.0,
            losses["ret"].mean() * 100 if len(losses) else 0.0,
            trades["ret"].max() * 100, trades["ret"].min() * 100))
        held = (trades["exit_time"] - trades["entry_time"]).mean()
        exposure = (trades["exit_time"] - trades["entry_time"]).sum() / (df.index[-1] - df.index[0])
        print("Avg hold: %s   Time in market: %.1f%%   Exits: %s" % (
            str(held).split(".")[0], exposure * 100,
            ", ".join("%s %d" % kv for kv in trades["reason"].value_counts().items())))
    if show_trades and n:
        print("-" * 62)
        for t in trades.itertuples():
            print("%s  ->  %s   %10.2f -> %10.2f   %+7.2f%%  %s" % (
                t.entry_time.strftime("%Y-%m-%d %H:%M"), t.exit_time.strftime("%Y-%m-%d %H:%M"),
                t.entry, t.exit, t.ret * 100, t.reason))
    print("\nNot financial advice. A backtest is not a forecast.")


def cmd_backtest(a):
    if a.fast >= a.slow:
        sys.exit("--fast must be smaller than --slow")
    if a.csv:
        df = load_csv(a.csv)
        tf_seconds = int(pd.Series(df.index).diff().median().total_seconds())
        source = a.csv
    else:
        ex = make_exchange(a.exchange)
        tf_seconds = ex.parse_timeframe(a.timeframe)
        print("Fetching %d days of %s %s candles from %s..." % (a.days, a.symbol, a.timeframe, a.exchange),
              file=sys.stderr)
        try:
            df = fetch_candles(ex, a.symbol, a.timeframe, a.days)
        except ccxt.BaseError as exc:
            sys.exit("Could not fetch candles from %s: %s" % (a.exchange, str(exc)[:300]))
        source = a.exchange
    if len(df) <= a.slow + 2:
        sys.exit("Only %d candles from %s; need more than %d. Raise --days." % (len(df), source, a.slow + 2))
    if a.save:
        df.to_csv(a.save, index_label="timestamp")
    df = add_indicators(df, a.fast, a.slow)
    eq, trades = backtest(df, a.cash, a.fee / 100, a.slippage / 100, a.atr_mult)
    print("Strategy: EMA %d/%d cross, long only, stop %.1f x ATR(14), fee %.3f%%/side, slippage %.3f%%" % (
        a.fast, a.slow, a.atr_mult, a.fee, a.slippage))
    report(df, eq, trades, a.cash, a.symbol, a.timeframe, tf_seconds, a.trades)


def main(argv=None):
    p = argparse.ArgumentParser(description="Crypto backtest bot (ccxt + pandas). Reads public data only.")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest", help="backtest an EMA crossover on exchange candles")
    b.add_argument("--exchange", default="binance")
    b.add_argument("--symbol", default="BTC/USDT")
    b.add_argument("--timeframe", default="1h")
    b.add_argument("--days", type=int, default=180)
    b.add_argument("--fast", type=int, default=20, help="fast EMA length (default 20)")
    b.add_argument("--slow", type=int, default=50, help="slow EMA length (default 50)")
    b.add_argument("--atr-mult", type=float, default=2.0, help="stop distance in ATRs (default 2)")
    b.add_argument("--cash", type=float, default=10000.0)
    b.add_argument("--fee", type=float, default=0.1, help="fee per side in %% (default 0.1)")
    b.add_argument("--slippage", type=float, default=0.02, help="slippage per fill in %% (default 0.02)")
    b.add_argument("--trades", action="store_true", help="list every trade")
    b.add_argument("--csv", help="read candles from a CSV instead of the exchange")
    b.add_argument("--save", help="save the fetched candles to this CSV")
    a = p.parse_args(argv)
    if a.cmd == "backtest":
        cmd_backtest(a)


if __name__ == "__main__":
    main()
