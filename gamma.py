import json, math, os
from datetime import datetime, timezone, date
import numpy as np
import pandas as pd

TICKER = "QQQ"


def bs_gamma(S, K, T, iv):
    S, K, T, iv = (np.asarray(x, dtype=float) for x in (S, K, T, iv))
    sq = iv * np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * iv * iv * T) / sq
    return np.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) / (S * sq)


def prep(calls, puts, T):
    rows = []
    for df, sgn in ((calls, 1), (puts, -1)):
        d = df[["strike", "openInterest", "impliedVolatility"]].copy()
        d = d.fillna(0)
        d = d[(d.openInterest > 0) & (d.impliedVolatility > 0.02)]
        d["sgn"], d["T"] = sgn, T
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def gex_at(opt, S):
    g = bs_gamma(S, opt.strike.values, opt["T"].values, opt.impliedVolatility.values)
    return float((opt.sgn.values * g * opt.openInterest.values * 100 * S * S * 0.01).sum())


def levels(opt, spot, pain_src):
    o = opt[(opt.strike > spot * 0.85) & (opt.strike < spot * 1.15)].copy()
    g = bs_gamma(spot, o.strike.values, o["T"].values, o.impliedVolatility.values)
    o["gex"] = o.sgn.values * g * o.openInterest.values * 100 * spot * spot * 0.01
    by = o.groupby("strike").agg(
        call=("gex", lambda x: x[x > 0].sum()), put=("gex", lambda x: x[x < 0].sum()))
    by["net"] = by.call + by.put
    up = by[by.index >= spot]
    dn = by[by.index <= spot]
    cw = float(up.call.idxmax()) if len(up) and up.call.max() > 0 else None
    pw = float(dn.put.idxmin()) if len(dn) and dn.put.min() < 0 else None
    grid = np.linspace(spot * 0.93, spot * 1.07, 57)
    vals = [gex_at(opt, s) for s in grid]
    best = None
    for i in range(len(grid) - 1):
        a, b = vals[i], vals[i + 1]
        if a == 0 or a * b < 0:
            x = grid[i] if a == 0 else grid[i] + (grid[i + 1] - grid[i]) * (0 - a) / (b - a)
            if best is None or abs(x - spot) < abs(best - spot):
                best = x
    zg = float(best) if best is not None else None
    ks = np.array(sorted(pain_src.strike.unique()))
    c = pain_src[pain_src.sgn > 0].groupby("strike").openInterest.sum().reindex(ks, fill_value=0).values
    p = pain_src[pain_src.sgn < 0].groupby("strike").openInterest.sum().reindex(ks, fill_value=0).values
    pay = [(np.maximum(k - ks, 0) * c).sum() + (np.maximum(ks - k, 0) * p).sum() for k in ks]
    mp = float(ks[int(np.argmin(pay))]) if len(ks) else None
    tot = float(by.net.sum())
    near = by[(by.index > spot * 0.97) & (by.index < spot * 1.03)]
    return {"call_wall": cw, "put_wall": pw, "zero_gamma": zg, "max_pain": mp, "net_total": tot,
            "pin": float(by.net.abs().idxmax()) if len(by) else None,
            "strikes": [{"k": float(k), "call": round(float(r.call)), "put": round(float(r.put)), "net": round(float(r.net))}
                        for k, r in near.iterrows()]}


def main():
    import yfinance as yf
    t = yf.Ticker(TICKER)
    spot = float(t.history(period="5d")["Close"].iloc[-1])
    nq = float(yf.Ticker("NQ=F").history(period="5d")["Close"].iloc[-1])
    ratio = nq / spot
    today = date.today()
    exps = []
    for e in t.options:
        d = datetime.strptime(e, "%Y-%m-%d").date()
        if d >= today and (d - today).days <= 7:
            exps.append((e, d))
    if not exps:
        raise SystemExit("sem expiries")
    frames = []
    for e, d in exps:
        ch = t.option_chain(e)
        T = max((d - today).days, 0.5) / 365
        o = prep(ch.calls, ch.puts, T)
        o["exp"] = e
        frames.append(o)
    allo = pd.concat(frames, ignore_index=True)
    first = allo[allo.exp == exps[0][0]]
    out = {"updated": datetime.now(timezone.utc).isoformat(), "ticker": TICKER, "spot": round(spot, 2),
           "nq": round(nq, 1), "ratio": round(ratio, 4), "sets": {}}
    for name, src, lab in (("1d", first, exps[0][0]), ("7d", allo, "até " + exps[-1][0])):
        L = levels(src, spot, src)
        for k in ("call_wall", "put_wall", "zero_gamma", "max_pain", "pin"):
            if L[k] is not None:
                L[k + "_nq"] = round(L[k] * ratio)
                L[k] = round(L[k], 2)
        for s in L["strikes"]:
            s["nq"] = round(s["k"] * ratio)
        L["regime"] = "positivo" if L["net_total"] > 0 else "negativo"
        L["expiry"] = lab
        out["sets"][name] = L
    os.makedirs("docs", exist_ok=True)
    with open("docs/gamma.json", "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("ok", out["sets"]["1d"]["call_wall"], out["sets"]["1d"]["put_wall"])


if __name__ == "__main__":
    main()
