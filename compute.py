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
closed = nq[nq.index.date < TODAY]          # só barras já fechadas
spot = float(nq["Close"].iloc[-1])           # preço mais recente
prev = float(closed["Close"].iloc[-1])       # fecho de ontem
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


TODAY_UTC = datetime.now(timezone.utc).date()


def hist_h(t):
    d = yf.download(t, period="10d", interval="1h", auto_adjust=True, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.dropna(subset=["Close"])
    d.index = d.index.tz_convert("UTC")
    return d


def sess(d):
    x = d[d.index.date == TODAY_UTC]
    a = x[(x.index.hour >= 0) & (x.index.hour < 6)]      # Ásia (00-06 UTC)
    l = x[(x.index.hour >= 6) & (x.index.hour < 12)]     # Londres (06-12 UTC)
    return x, a, l


def m_amd():
    """Bias do teu plano (AMD / PO3 / DOL) com o que dá para calcular sozinho."""
    pts, notes = 0, []
    c = closed["Close"]
    s20, s50 = c.rolling(20).mean().iloc[-1], c.rolling(50).mean().iloc[-1]
    if spot > s50 and s20 > s50:
        pts += 25
        notes.append("HTF em alta")
    elif spot < s50 and s20 < s50:
        pts -= 25
        notes.append("HTF em baixa")
    else:
        notes.append("HTF misto")
    pdh, pdl = float(closed["High"].iloc[-1]), float(closed["Low"].iloc[-1])
    iso = closed.index.isocalendar()
    key = iso["year"].astype(int) * 100 + iso["week"].astype(int)
    ti = TODAY.isocalendar()
    cur = int(ti[0]) * 100 + int(ti[1])
    older = key[key < cur]
    pwh = pwl = None
    if len(older):
        pw = closed[key == older.iloc[-1]]
        pwh, pwl = float(pw["High"].max()), float(pw["Low"].min())
    cw = closed[key == cur]
    x = None
    try:
        x, a, l = sess(hist_h("NQ=F"))
        if len(a) < 3 or len(l) < 1:
            x = None
    except Exception:
        x = None
    px, dh, dl = spot, spot, spot
    if x is not None:
        px, dh, dl = float(x["Close"].iloc[-1]), float(x["High"].max()), float(x["Low"].min())
        ah, al = float(a["High"].max()), float(a["Low"].min())
        lh, ll = float(l["High"].max()), float(l["Low"].min())
        sh, sl = lh > ah, ll < al
        if sl and not sh:
            if px > al:
                pts += 25
                notes.append("Londres varreu mínimos da Ásia e recuperou (manipulação em baixa)")
            else:
                pts -= 15
                notes.append("Londres rompeu mínimos da Ásia e ficou abaixo")
        elif sh and not sl:
            if px < ah:
                pts -= 25
                notes.append("Londres varreu máximos da Ásia e recuou (manipulação em alta)")
            else:
                pts += 15
                notes.append("Londres rompeu máximos da Ásia e ficou acima")
        elif sh and sl:
            notes.append("Londres varreu os dois lados da Ásia")
        else:
            notes.append("Londres dentro da Ásia (acumulação em curso)")
        if dh > pdh and px < pdh:
            pts -= 25
            notes.append("varreu PDH e recuou")
        if dl < pdl and px > pdl:
            pts += 25
            notes.append("varreu PDL e recuperou")
        tp = (x["High"] + x["Low"] + x["Close"]) / 3
        if float(x["Volume"].sum()) > 0:
            vw = float((tp * x["Volume"]).sum() / x["Volume"].sum())
            pts += 10 if px > vw else -10
            notes.append(f"{'acima' if px > vw else 'abaixo'} do VWAP ({vw:.0f})")
        try:
            ex, ea, el = sess(hist_h("ES=F"))
            if sl and len(el) and float(el["Low"].min()) >= float(ea["Low"].min()):
                pts += 15
                notes.append("SMT em alta com o ES")
            if sh and len(el) and float(el["High"].max()) <= float(ea["High"].max()):
                pts -= 15
                notes.append("SMT em baixa com o ES")
        except Exception:
            pass
    else:
        notes.append("sem dados intradiários hoje")
    wk_hi = max(float(cw["High"].max()) if len(cw) else dh, dh)
    wk_lo = min(float(cw["Low"].min()) if len(cw) else dl, dl)
    up, dn = {}, {}
    for k, v, hi, lo in [("PDH", pdh, dh, dl), ("PDL", pdl, dh, dl), ("PWH", pwh, wk_hi, wk_lo), ("PWL", pwl, wk_hi, wk_lo)]:
        if v is None:
            continue
        if v > px and hi < v:
            up[k] = v
        if v < px and lo > v:
            dn[k] = v
    ka = min(up, key=lambda k: up[k] - px) if up else None
    kb = min(dn, key=lambda k: px - dn[k]) if dn else None
    if ka and kb:
        toward_up = (up[ka] - px) < (px - dn[kb])
    else:
        toward_up = bool(ka)
    if ka or kb:
        pts += 15 if toward_up else -15
        k = ka if toward_up else kb
        v = up[k] if toward_up else dn[k]
        notes.append(f"DOL mais próximo: {k} {v:.0f}")
    if pdh > pdl:
        pos = (px - pdl) / (pdh - pdl)
        if 0.35 <= pos <= 0.65:
            notes.append("preço no meio do range de ontem: zona de NO TRADE")
        elif pos < 0 or pos > 1:
            notes.append("preço fora do range de ontem")
    if TODAY.weekday() in (1, 2):
        notes.append("terça/quarta: janela típica de manipulação e do high/low da semana")
    return clip(pts), "; ".join(notes) + f". PDH {pdh:.0f} PDL {pdl:.0f}" + (f" PWH {pwh:.0f} PWL {pwl:.0f}" if pwh else "")


MODS = [("amd", "AMD / DOL (plano)", 15, m_amd), ("eq", "EQH / EQL", 6, m_eq), ("fund", "Fundamentals", 4, m_fund),
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

total = round(total * 100 / sum(w for _, _, w, _ in MODS))
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
