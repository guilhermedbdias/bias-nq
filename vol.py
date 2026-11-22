"""Painel diário: hoje vai mexer-se muito? (expansão do range do NQ)"""
import json, math, os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import yfinance as yf

EXP = 1.5  # expansão = range do dia >= 1,5x a média dos últimos 20 dias


def hist(t, period):
    d = yf.download(t, period=period, interval="1d", auto_adjust=True, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.dropna(subset=["Close"])
    d.index = pd.to_datetime(d.index).tz_localize(None)
    return d


def stat(sub, base):
    n = len(sub)
    if n < 30:
        return {"n": n}
    p = float(sub["t"].mean() * 100)
    se = math.sqrt(p / 100 * (1 - p / 100) / n) * 100
    return {"n": n, "p": round(p, 1), "vs": round(p - base, 1), "se": round(se, 1),
            "range_med": round(float(sub["r"].median()), 2)}


def compute(nq, vxn, today):
    C, H, L = nq["Close"], nq["High"], nq["Low"]
    rng = (H - L) / C
    avg20 = rng.rolling(20).mean()
    vx = vxn["Close"].reindex(nq.index).ffill()
    rank = vx.rolling(250).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True)
    bucket = pd.Series(np.where(rank <= 1 / 3, "baixa", np.where(rank >= 2 / 3, "alta", "média")),
                       index=nq.index).where(rank.notna())
    df = pd.DataFrame({"comp": (rng.rolling(5).mean() / avg20) < 0.75,
                       "big": (rng / avg20.shift(1)) >= EXP,
                       "vx": bucket, "rank": rank,
                       "t": (rng.shift(-1) / avg20) >= EXP,
                       "r": rng.shift(-1) * 100,
                       "ok": rng.shift(-1).notna() & avg20.notna() & rank.notna()})
    closed = df[df.index.date < today]
    last = closed.iloc[-1]
    h = closed.iloc[:-1]
    h = h[h["ok"]]                       # só dias cujo dia seguinte já fechou
    base = float(h["t"].mean() * 100)
    cell = h[(h["comp"] == last["comp"]) & (h["big"] == last["big"]) & (h["vx"] == last["vx"])]
    st = stat(cell, base)
    prev = float(C[closed.index[-1]])
    vxn_last = float(vx[closed.index[-1]])
    if "p" in st:
        label = ("Acima do normal" if st["vs"] > 2 * st["se"] else
                 "Abaixo do normal" if st["vs"] < -2 * st["se"] else "Normal (sem sinal claro)")
    else:
        label = "Poucos dados para este cenário"
    marg = []
    for nome, col, val in [("Compressão (range 5d < 75% da média 20d)", "comp", True),
                           ("Ontem foi expansão (≥ 1,5x a média)", "big", True),
                           ("VXN alto (percentil ≥ 67%)", "vx", "alta"),
                           ("VXN baixo (percentil ≤ 33%)", "vx", "baixa")]:
        s = stat(h[h[col] == val], base)
        s.update({"cond": nome, "ativo": bool(last[col] == val)})
        marg.append(s)
    return {"updated": datetime.now(timezone.utc).isoformat(),
            "ref": str(closed.index[-1].date()), "spot": round(prev, 1),
            "vxn": round(vxn_last, 1), "vxn_pct": round(float(last["rank"]) * 100),
            "estado": {"compressao": bool(last["comp"]), "ontem_expansao": bool(last["big"]),
                       "vxn": str(last["vx"])},
            "base_exp": round(base, 1), "cenario": st, "label": label,
            "range_base": round(float(h["r"].median()), 2),
            "range_pts": round(st["range_med"] * prev / 100) if "range_med" in st else None,
            "sigma_pts": round(prev * vxn_last / 100 / math.sqrt(252)),
            "marginais": marg}


def main():
    today = datetime.now(ZoneInfo("America/New_York")).date()
    out = compute(hist("NQ=F", "10y"), hist("^VXN", "10y"), today)
    os.makedirs("docs", exist_ok=True)
    with open("docs/vol.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(out["label"], out["cenario"])


if __name__ == "__main__":
    main()
