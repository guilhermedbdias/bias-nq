"""
Backtest do painel de bias: recalcula o score de cada dia do passado,
só com a informação disponível nesse dia, e compara com o movimento do NQ no dia seguinte.
Sinal calculado no fecho do dia t  ->  alvo: fecho(t+1) / fecho(t) - 1
"""
import json, math, os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf

PERIOD = "8y"
W = {"eq": 6, "fund": 4, "vp": 8, "candles": 10, "pat": 4, "pct": 20,
     "trend": 15, "bet": 8, "macro": 15, "season": 10}
NAMES = {"eq": "EQH / EQL", "fund": "Fundamentals", "vp": "Volume Profile",
         "candles": "Daily Candles", "pat": "Known Patterns", "pct": "Percentis",
         "trend": "Trends", "bet": "Betting Markets", "macro": "Macro Economics",
         "season": "Seasonality"}


def hist(t, period=PERIOD):
    d = yf.download(t, period=period, interval="1d", auto_adjust=True, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.dropna(subset=["Close"])
    d.index = pd.to_datetime(d.index).tz_localize(None)
    return d


def clip(x, a=-100, b=100):
    return float(max(a, min(b, x)))


def build(nq, vxn, tnx, dxy, vix, ndx, techs):
    idx = nq.index
    C, H, L = nq["Close"], nq["High"], nq["Low"]
    n = len(nq)

    def al(s):
        return s.reindex(idx).ffill()

    S = pd.DataFrame(0.0, index=idx, columns=list(W))

    # Percentis: onde fechou o dia t face ao intervalo esperado (VXN de t-1)
    vxn_p = al(vxn["Close"]).shift(1)
    z = (C - C.shift(1)) / (C.shift(1) * vxn_p / 100 / math.sqrt(252))
    pc = z.map(lambda v: np.nan if pd.isna(v) else min(98, max(2, 100 * 0.5 * (1 + math.erf(v / math.sqrt(2))))))
    S["pct"] = ((50 - pc) * 1.8).fillna(0)

    # Trends
    s20, s50 = C.rolling(20).mean(), C.rolling(50).mean()
    pts = (2 * (C > s20) - 1) + (2 * (C > s50) - 1) + (2 * (s20 > s50) - 1) + (2 * (C > C.shift(5)) - 1)
    S["trend"] = (25 * pts).where(s50.notna(), 0)

    # Daily Candles
    rng = (H - L).where((H - L) > 0)
    pos = ((C - L) / rng).fillna(0.5)
    S["candles"] = ((pos - 0.5) * 100 + 25 * (C > H.shift(1)) - 25 * (C < L.shift(1))).fillna(0)

    # Macro
    def chg(df):
        c = al(df["Close"])
        return (c / c.shift(5) - 1) * 100
    S["macro"] = (-(chg(tnx) * 3).clip(-35, 35) - (chg(dxy) * 10).clip(-30, 30)
                  - (chg(vix) * 1.5).clip(-35, 35)).fillna(0).clip(-100, 100)

    # Fundamentals (proxy: big techs acima da média de 50 dias)
    cols = []
    for df in techs:
        c = df["Close"]
        m = c.rolling(50).mean()
        cols.append((c > m).astype(float).where(m.notna()).reindex(idx).ffill())
    if cols:
        S["fund"] = ((pd.concat(cols, axis=1).mean(axis=1) - 0.5) * 200).fillna(0)

    # Volume Profile e EQH/EQL (janelas deslizantes)
    Hv, Lv, Cv, Vv = H.values, L.values, C.values, nq["Volume"].values
    vp, eq = np.zeros(n), np.zeros(n)
    for i in range(30, n):
        s = Cv[i]
        w = slice(i - 19, i + 1)
        tp = (Hv[w] + Lv[w] + Cv[w]) / 3
        h, edges = np.histogram(tp, bins=30, weights=Vv[w])
        if h.sum() > 0:
            poc = (edges[:-1] + edges[1:])[h.argmax()] / 2
            ids, cum = [], 0
            for j in np.argsort(h)[::-1]:
                ids.append(j)
                cum += h[j]
                if cum >= 0.7 * h.sum():
                    break
            vah, val = edges[max(ids) + 1], edges[min(ids)]
            vp[i] = (50 if s > vah else -50 if s < val else 0) + (25 if s > poc else -25)
        w2 = slice(i - 29, i + 1)

        def cl(v):
            return [(v[a] + v[b]) / 2 for a in range(len(v)) for b in range(a + 1, len(v))
                    if abs(v[a] - v[b]) / v[a] < 0.0015]
        eqh = [x for x in cl(Hv[w2]) if x > s]
        eql = [x for x in cl(Lv[w2]) if x < s]
        dh = min(eqh) - s if eqh else None
        dl = s - max(eql) if eql else None
        if dh is not None and (dl is None or dh < dl):
            eq[i] = 50
        elif dl is not None:
            eq[i] = -50
    S["vp"], S["eq"] = vp, eq

    # Seasonality: média do dia da semana do DIA SEGUINTE, só com passado
    r = ndx["Close"].pct_change().dropna()
    rd, rv, rw = r.index.values, r.values, r.index.dayofweek.values
    nxt_dow = pd.Series(idx.dayofweek, index=idx).shift(-1)
    se = np.zeros(n)
    for i in range(60, n - 1):
        m = (rd <= idx.values[i]) & (rw == int(nxt_dow.iloc[i]))
        x = rv[m]
        if len(x) >= 30 and x.std() > 0:
            se[i] = clip(x.mean() / (x.std() / math.sqrt(len(x))) * 15, -30, 30)
    S["season"] = se

    contrib = pd.DataFrame({k: (S[k].clip(-100, 100) / 100 * W[k]).round() for k in W})
    total = contrib.sum(axis=1)
    nxt = C.shift(-1) / C - 1
    return S, total, nxt


BINS = [(-999, -30, "≤ −30"), (-29, -15, "−29 a −15"), (-14, 14, "−14 a +14"),
        (15, 29, "+15 a +29"), (30, 999, "≥ +30")]


def summarize(S, total, nxt):
    ok = nxt.notna() & (np.arange(len(nxt)) >= 60)
    S, total, nxt = S[ok], total[ok], nxt[ok]
    out = {"dias": int(len(nxt)),
           "base": {"sobe_pct": round(float((nxt > 0).mean() * 100), 1),
                    "media_pct": round(float(nxt.mean() * 100), 3)},
           "faixas": [], "modulos": []}
    for lo, hi, lab in BINS:
        m = (total >= lo) & (total <= hi)
        if m.sum() == 0:
            continue
        out["faixas"].append({"faixa": lab, "dias": int(m.sum()),
                              "sobe_pct": round(float((nxt[m] > 0).mean() * 100), 1),
                              "media_pct": round(float(nxt[m].mean() * 100), 3)})
    for k in W:
        if (S[k] == 0).all():
            continue
        p, q = S[k] > 10, S[k] < -10
        row = {"modulo": NAMES[k], "dias_bull": int(p.sum()), "dias_bear": int(q.sum())}
        if p.sum() >= 20 and q.sum() >= 20:
            up_p, up_q = float((nxt[p] > 0).mean() * 100), float((nxt[q] > 0).mean() * 100)
            row.update({"sobe_se_bull": round(up_p, 1), "sobe_se_bear": round(up_q, 1),
                        "vantagem": round(up_p - up_q, 1)})
        out["modulos"].append(row)
    return out


def main():
    nq = hist("NQ=F")
    args = [hist("^VXN"), hist("^TNX"), hist("DX-Y.NYB"), hist("^VIX"), hist("^NDX", "12y")]
    techs = []
    for t in ["NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL"]:
        try:
            techs.append(hist(t))
        except Exception:
            pass
    S, total, nxt = build(nq, *args, techs)
    res = {"gerado": datetime.now(timezone.utc).isoformat(),
           "inicio": str(nq.index[60].date()), "fim": str(nq.index[-2].date()),
           "tudo": summarize(S, total, nxt),
           "ultimos_2_anos": summarize(S.tail(520), total.tail(520), nxt.tail(520))}
    os.makedirs("docs", exist_ok=True)
    with open("docs/backtest.json", "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res["tudo"]["faixas"], ensure_ascii=False))


if __name__ == "__main__":
    main()
