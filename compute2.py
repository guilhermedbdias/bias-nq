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


def swings(d, n=2):
    H, L = d["High"].values, d["Low"].values
    sh = [i for i in range(n, len(d) - n) if H[i] == max(H[i - n:i + n + 1])]
    sl = [i for i in range(n, len(d) - n) if L[i] == min(L[i - n:i + n + 1])]
    return sh, sl


def m_struct():
    """Manipulação / estrutura protegida (do teu documento), em dados diários."""
    d = closed.tail(45).reset_index(drop=True)
    sh, sl = swings(d)
    pl, ph, wl, wh = [], [], [], []   # protegidos / fracos (não protegidos)
    for i in sl:                       # lows: ainda não tirados
        lvl = float(d["Low"].iloc[i])
        if float(d["Low"].iloc[i + 1:].min()) < lvl:
            continue
        pr = [j for j in sh if j < i]
        bos = bool(pr) and bool((d["Close"].iloc[i + 1:] > float(d["High"].iloc[pr[-1]])).any())
        (pl if bos else wl).append(lvl)
    for i in sh:                       # highs: ainda não tirados
        lvl = float(d["High"].iloc[i])
        if float(d["High"].iloc[i + 1:].max()) > lvl:
            continue
        pr = [j for j in sl if j < i]
        bos = bool(pr) and bool((d["Close"].iloc[i + 1:] < float(d["Low"].iloc[pr[-1]])).any())
        (ph if bos else wh).append(lvl)
    pts, notes = 0, []
    plb = [x for x in pl if x < spot]
    pha = [x for x in ph if x > spot]
    if plb and not pha:
        pts += 25
        notes.append(f"low protegido (BOS feito) abaixo: {max(plb):.0f}")
    elif pha and not plb:
        pts -= 25
        notes.append(f"high protegido (BOS feito) acima: {min(pha):.0f}")
    elif plb and pha:
        notes.append("há lows e highs protegidos (range)")
    else:
        notes.append("sem estrutura protegida clara")
    wlb = [x for x in wl if x < spot]
    wha = [x for x in wh if x > spot]
    dl = spot - max(wlb) if wlb else None
    dh = min(wha) - spot if wha else None
    if dl is not None and (dh is None or dl < dh):
        notes.append(f"estrutura fraca mais perto abaixo ({max(wlb):.0f}): manipulação tende a ir para baixo")
    elif dh is not None:
        notes.append(f"estrutura fraca mais perto acima ({min(wha):.0f}): manipulação tende a ir para cima")
    # ciclos AMD múltiplos (últimos 5 dias): varreu e fechou de volta
    t = closed.tail(6)
    bull = bear = 0
    for k in range(1, len(t)):
        r, p = t.iloc[k], t.iloc[k - 1]
        if r.Low < p.Low and r.Close > p.Low:
            bull += 1
        if r.High > p.High and r.Close < p.High:
            bear += 1
    pts += clip((bull - bear) * 15, -30, 30)
    notes.append(f"últimos 5 dias: {bull} ciclos de alta, {bear} de baixa")
    # ontem: manipulação protegida ou não
    r, p = closed.iloc[-1], closed.iloc[-2]
    if r.Low < p.Low and r.Close > p.Low:
        pts += 20
        notes.append("ontem varreu o low e fechou acima (manipulação protegida)")
    elif r.High > p.High and r.Close < p.High:
        pts -= 20
        notes.append("ontem varreu o high e fechou abaixo (manipulação protegida)")
    elif r.Low < p.Low and r.Close < p.Low:
        pts -= 10
        notes.append("ontem fechou abaixo do low anterior (não protegido)")
    elif r.High > p.High and r.Close > p.High:
        pts += 10
        notes.append("ontem fechou acima do high anterior (não protegido)")
    # semana: OHLC semanal
    iso = closed.index.isocalendar()
    key = iso["year"].astype(int) * 100 + iso["week"].astype(int)
    ti = TODAY.isocalendar()
    cur = int(ti[0]) * 100 + int(ti[1])
    ks = sorted(set(key[key < cur]))
    if len(ks) >= 2:
        w1, w0 = closed[key == ks[-1]], closed[key == ks[-2]]
        l1, h1, c1 = float(w1["Low"].min()), float(w1["High"].max()), float(w1["Close"].iloc[-1])
        l0, h0 = float(w0["Low"].min()), float(w0["High"].max())
        if l1 < l0 and c1 > l0:
            pts += 20
            notes.append("semana passada varreu o low semanal e fechou acima")
        elif h1 > h0 and c1 < h0:
            pts -= 20
            notes.append("semana passada varreu o high semanal e fechou abaixo")
        else:
            notes.append("sem varrimento semanal")
    return clip(pts), "; ".join(notes)


MODS = [("amd", "AMD / DOL (plano)", 15, m_amd), ("struct", "Manipulação / estrutura", 15, m_struct),
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
