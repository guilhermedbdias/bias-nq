import json, math, os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import yfinance as yf


def hist(t, period):
    d = yf.download(t, period=period, interval="1d", auto_adjust=True, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.dropna(subset=["Close"])
    d.index = pd.to_datetime(d.index).tz_localize(None)
    return d


def clip(x, a=-100, b=100):
    return float(max(a, min(b, x)))


TODAY = datetime.now(ZoneInfo("America/New_York")).date()
nq = hist("NQ=F", "2y")
closed = nq[nq.index.date < TODAY]
spot = float(nq["Close"].iloc[-1])
prev = float(closed["Close"].iloc[-1])
LEVELS = {}


def m_pct():
    vxn = float(hist("^VXN", "10d")["Close"].iloc[-1])
    sd = prev * vxn / 100 / math.sqrt(252)
    z = [-1.645, -0.674, 0, 0.674, 1.645]
    xs = [prev + k * sd for k in z]
    for name, v in zip(["p5", "p25", "p50", "p75", "p95"], xs):
        LEVELS[name] = round(v, 1)
    if spot <= xs[0]:
        pct = 2
    elif spot >= xs[-1]:
        pct = 98
    else:
        pct = float(np.interp(spot, xs, [5, 25, 50, 75, 95]))
    return (50 - pct) * 1.8, f"preço no percentil {pct:.0f} (VXN {vxn:.1f})"


def m_trend():
    c = closed["Close"]
    s20, s50 = c.rolling(20).mean().iloc[-1], c.rolling(50).mean().iloc[-1]
    pts = [spot > s20, spot > s50, s20 > s50, c.iloc[-1] > c.iloc[-6]]
    return sum(25 if p else -25 for p in pts), f"{sum(pts)}/4 condições de alta"


def m_candles():
    r, p = closed.iloc[-1], closed.iloc[-2]
    rng = r.High - r.Low
    pos = (r.Close - r.Low) / rng if rng > 0 else 0.5
    sc = (pos - 0.5) * 100 + (25 if r.Close > p.High else -25 if r.Close < p.Low else 0)
    return sc, f"fechou a {pos*100:.0f}% do range de ontem"


def m_vp():
    d = closed.tail(20)
    tp = ((d.High + d.Low + d.Close) / 3).values
    h, edges = np.histogram(tp, bins=30, weights=d.Volume.values)
    poc = (edges[:-1] + edges[1:])[h.argmax()] / 2
    idx, cum = [], 0
    for i in np.argsort(h)[::-1]:
        idx.append(i)
        cum += h[i]
        if cum >= 0.7 * h.sum():
            break
    vah, val = edges[max(idx) + 1], edges[min(idx)]
    sc = (50 if spot > vah else -50 if spot < val else 0) + (25 if spot > poc else -25)
    return sc, f"VAH {vah:.0f} / POC {poc:.0f} / VAL {val:.0f} (20 dias)"


def m_eq():
    d, tol = closed.tail(30), 0.0015

    def clusters(v):
        return [(v[i] + v[j]) / 2 for i in range(len(v)) for j in range(i + 1, len(v))
                if abs(v[i] - v[j]) / v[i] < tol]
    eqh = [x for x in clusters(d.High.values) if x > spot]
    eql = [x for x in clusters(d.Low.values) if x < spot]
    dh = min(eqh) - spot if eqh else None
    dl = spot - max(eql) if eql else None
    if dh is not None and (dl is None or dh < dl):
        return 50, "equal highs mais perto, acima"
    if dl is not None:
        return -50, "equal lows mais perto, abaixo"
    return 0, "sem equal highs/lows próximos"


def m_fund():
    up = 0
    names = ["NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL"]
    for t in names:
        c = hist(t, "6mo")["Close"]
        up += int(c.iloc[-1] > c.rolling(50).mean().iloc[-1])
    return (up / len(names) - 0.5) * 200, f"{up}/6 big techs acima da média de 50 dias"


def chg(t):
    c = hist(t, "1mo")["Close"]
    return (c.iloc[-1] / c.iloc[-6] - 1) * 100


def m_macro():
    tnx, dxy, vix = chg("^TNX"), chg("DX-Y.NYB"), chg("^VIX")
    sc = -clip(tnx * 3, -35, 35) - clip(dxy * 10, -30, 30) - clip(vix * 1.5, -35, 35)
    return sc, f"5d: yield10 {tnx:+.1f}%, dólar {dxy:+.1f}%, VIX {vix:+.1f}%"


def m_season():
    r = hist("^NDX", "10y")["Close"].pct_change().dropna()
    r = r[r.index.date < TODAY]
    x = r[r.index.dayofweek == TODAY.weekday()]
    t = x.mean() / (x.std() / math.sqrt(len(x)))
    return clip(t * 15, -30, 30), f"média deste dia da semana: {x.mean()*100:+.2f}%"


MODS = [("eq", "EQH / EQL", 6, m_eq), ("fund", "Fundamentals", 4, m_fund),
        ("vp", "Volume Profile", 8, m_vp), ("candles", "Daily Candles", 10, m_candles),
        ("pat", "Known Patterns", 4, None), ("pct", "Percentis", 20, m_pct),
        ("trend", "Trends", 15, m_trend), ("bet", "Betting Markets", 8, None),
        ("macro", "Macro Economics", 15, m_macro), ("season", "Seasonality", 10, m_season)]

mods, total = [], 0
for k, name, w, f in MODS:
    if f is None:
        sc, note = 0, "sem fonte automática"
    else:
        try:
            sc, note = f()
        except Exception as e:
            sc, note = 0, "erro: " + str(e)[:80]
    sc = round(clip(sc))
    c = round(sc / 100 * w)
    total += c
    mods.append({"k": k, "name": name, "weight": w, "score": sc, "contrib": c, "note": note})

up = round(clip(50 + total * 0.72, 5, 95))
label = ("Muito bearish" if total <= -40 else "Bearish" if total <= -10 else
         "Muito bullish" if total >= 40 else "Bullish" if total >= 10 else "Neutro")
out = {"updated": datetime.now(timezone.utc).isoformat(), "spot": round(spot, 1),
       "prev": round(prev, 1), "total": total, "label": label, "up": up,
       "levels": LEVELS, "modules": mods}
os.makedirs("docs", exist_ok=True)
with open("docs/data.json", "w") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
print(label, total)
