import json, os
from datetime import datetime, timezone, time as dtime
import numpy as np
import pandas as pd

STEP = 10  # pontos por nível de preço no perfil


def profile(d):
    vol = {}
    for h, l, v in zip(d["High"].values, d["Low"].values, d["Volume"].values):
        if not (v > 0) or np.isnan(h) or np.isnan(l):
            continue
        lo, hi = int(np.floor(l / STEP)), int(np.floor(h / STEP))
        n = hi - lo + 1
        for b in range(lo, hi + 1):
            vol[b] = vol.get(b, 0.0) + v / n
    return vol


def value_area(vol):
    if not vol:
        return None
    tot = sum(vol.values())
    poc = max(vol, key=vol.get)
    lo = hi = poc
    acc = vol[poc]
    for _ in range(2000):
        if acc >= 0.7 * tot:
            break
        ups = [k for k in vol if k > hi]
        dns = [k for k in vol if k < lo]
        if not ups and not dns:
            break
        nu = min(ups) if ups else None
        nd = max(dns) if dns else None
        if nd is None or (nu is not None and (vol[nu] >= vol[nd] if nu - hi == 1 and lo - nd == 1 else nu - hi <= lo - nd)):
            hi = nu
            acc += vol[nu]
        else:
            lo = nd
            acc += vol[nd]
    return {"poc": poc * STEP + STEP / 2, "vah": hi * STEP + STEP, "val": lo * STEP}


def compute(d5, daily, now_utc):
    d5 = d5.copy()
    d5.index = d5.index.tz_convert("America/New_York")
    now_et = now_utc.astimezone(d5.index.tz)
    rth = d5.between_time(dtime(9, 30), dtime(15, 59))
    sessions = sorted(set(rth.index.date))
    active = bool(sessions) and sessions[-1] == now_et.date() and now_et.time() >= dtime(9, 30)
    cur = sessions[-1] if active else None
    prev = (sessions[-2] if active else sessions[-1]) if len(sessions) >= (2 if active else 1) else None
    last = float(d5["Close"].iloc[-1])
    L = []

    def add(name, price, grp):
        if price is not None and not np.isnan(price):
            L.append({"name": name, "price": round(float(price), 2), "group": grp})

    info = {}
    if prev is not None:
        p = rth[rth.index.date == prev]
        add("Máx. sessão anterior (PDH)", p["High"].max(), "prior")
        add("Mín. sessão anterior (PDL)", p["Low"].min(), "prior")
        add("Fecho sessão anterior", p["Close"].iloc[-1], "prior")
        info["prev_date"] = str(prev)
        pr = p.between_time(dtime(9, 30), dtime(10, 29))
        if len(pr):
            add("IB anterior máx.", pr["High"].max(), "session")
            add("IB anterior mín.", pr["Low"].min(), "session")
        va = value_area(profile(p))
        if va:
            add("POC (sessão anterior)", va["poc"], "profile")
            add("VAH (sessão anterior)", va["vah"], "profile")
            add("VAL (sessão anterior)", va["val"], "profile")
        ps = [s for s in sessions if s <= prev][-5:]
        comp = rth[np.isin(rth.index.date, ps)]
        vc = value_area(profile(comp))
        if vc:
            add("POC (5 sessões)", vc["poc"], "profile")
            add("VAH (5 sessões)", vc["vah"], "profile")
            add("VAL (5 sessões)", vc["val"], "profile")
    if cur is not None:
        c = rth[rth.index.date == cur]
        add("Máx. de hoje", c["High"].max(), "session")
        add("Mín. de hoje", c["Low"].min(), "session")
        orr = c.between_time(dtime(9, 30), dtime(9, 59))
        ib = c.between_time(dtime(9, 30), dtime(10, 29))
        if len(orr):
            add("Opening range máx.", orr["High"].max(), "session")
            add("Opening range mín.", orr["Low"].min(), "session")
        if now_et.time() >= dtime(10, 30) and len(ib):
            add("IB máx.", ib["High"].max(), "session")
            add("IB mín.", ib["Low"].min(), "session")
        tp = (c["High"] + c["Low"] + c["Close"]) / 3
        if c["Volume"].sum() > 0:
            add("VWAP (hoje)", (tp * c["Volume"]).sum() / c["Volume"].sum(), "profile")
    if prev is not None:
        end = pd.Timestamp(datetime.combine(prev, dtime(16, 0))).tz_localize(d5.index.tz)
        start = pd.Timestamp(datetime.combine(cur, dtime(9, 30))).tz_localize(d5.index.tz) if cur else None
        on = d5[d5.index >= end]
        if start is not None:
            on = on[on.index < start]
        if len(on):
            add("Máx. overnight", on["High"].max(), "session")
            add("Mín. overnight", on["Low"].min(), "session")
    u = d5.copy()
    u.index = u.index.tz_convert("UTC")
    today_u = u.index[-1].date()
    x = u[u.index.date == today_u]
    a, lo_ = x[x.index.hour < 6], x[(x.index.hour >= 6) & (x.index.hour < 12)]
    if len(a) > 3:
        add("Ásia máx.", a["High"].max(), "session")
        add("Ásia mín.", a["Low"].min(), "session")
    if len(lo_) > 3:
        add("Londres máx.", lo_["High"].max(), "session")
        add("Londres mín.", lo_["Low"].min(), "session")
    dd = daily.copy()
    dd.index = pd.to_datetime(dd.index).tz_localize(None)
    iso = dd.index.isocalendar()
    key = iso["year"].astype(int) * 100 + iso["week"].astype(int)
    ks = sorted(set(key))
    if len(ks) >= 2:
        pw = dd[key == ks[-2]]
        add("Máx. semana anterior (PWH)", pw["High"].max(), "week")
        add("Mín. semana anterior (PWL)", pw["Low"].min(), "week")
    vol = profile(rth[rth.index.date == prev]) if prev is not None else {}
    lo_b, hi_b = last * 0.97, last * 1.03
    buckets = [{"p": b * STEP, "v": round(v)} for b, v in sorted(vol.items()) if lo_b <= b * STEP <= hi_b]
    return {"updated": now_utc.isoformat(), "last": round(last, 2), "active": active,
            "info": info, "levels": sorted(L, key=lambda z: -z["price"]), "step": STEP, "buckets": buckets}


def main():
    import yfinance as yf
    d5 = yf.download("NQ=F", period="10d", interval="5m", auto_adjust=True, progress=False)
    if isinstance(d5.columns, pd.MultiIndex):
        d5.columns = d5.columns.get_level_values(0)
    d5 = d5.dropna(subset=["Close"])
    daily = yf.download("NQ=F", period="1mo", interval="1d", auto_adjust=True, progress=False)
    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = daily.columns.get_level_values(0)
    out = compute(d5, daily.dropna(subset=["Close"]), datetime.now(timezone.utc))
    os.makedirs("docs", exist_ok=True)
    with open("docs/levels.json", "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("ok", len(out["levels"]), "níveis")


if __name__ == "__main__":
    main()
