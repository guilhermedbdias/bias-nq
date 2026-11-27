import json, os
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from levels import profile, value_area   # mesmas funções dos Níveis (perfil + 70% value area)


def only_rth(d, minute_bars):
    """mantém só 09:30-16:00 de Nova Iorque."""
    if d.empty:
        return d
    m = d.index.hour * 60 + d.index.minute
    return d[(m >= 570) & (m < 960)]


def levels_of(d, tag, out):
    """calcula POC/VAH/VAL do perfil composto de d e acrescenta a out."""
    va = value_area(profile(d)) if len(d) else None
    if not va:
        return
    out.append((f"{tag} VAH", va["vah"]))
    out.append((f"{tag} POC", va["poc"]))
    out.append((f"{tag} VAL", va["val"]))


def build(d5, d1h, gamma, now):
    """d5 = barras de 5 min (30 dias), d1h = barras de 1 hora (90 dias), gamma = dict do gamma.json ou None."""
    out = []
    for d in (d5, d1h):
        d.index = d.index.tz_convert("America/New_York")
    cut30 = now - timedelta(days=30)
    cut90 = now - timedelta(days=90)
    a30 = d5[d5.index >= cut30]
    a90 = d1h[d1h.index >= cut90]
    levels_of(a30, "VP30 ETH", out)
    levels_of(only_rth(a30, True), "VP30 RTH", out)
    levels_of(a90, "VP90 ETH", out)
    levels_of(only_rth(a90, False), "VP90 RTH", out)
    if gamma and gamma.get("sets"):
        names = [("call_wall_nq", "Call Wall"), ("put_wall_nq", "Put Wall"), ("zero_gamma_nq", "Zero Gamma"),
                 ("max_pain_nq", "Max Pain"), ("pin_nq", "Pin")]
        for sname in ("1d", "7d"):
            s = gamma["sets"].get(sname)
            if not s:
                continue
            for key, nm in names:
                if s.get(key):
                    out.append((f"Gamma {sname} {nm}", float(s[key])))
    out = [(n, round(float(p), 2)) for n, p in out if p is not None and not np.isnan(p)]
    return sorted(out, key=lambda z: -z[1])


def dl(ticker, period, interval):
    import yfinance as yf
    d = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d.dropna(subset=["Close"])


def main():
    now = datetime.now(timezone.utc)
    d5 = dl("NQ=F", "35d", "5m")       # 5m só existe até 60 dias
    d1h = dl("NQ=F", "100d", "60m")    # 1h dá para 90 dias
    gamma = None
    try:
        with open("docs/gamma.json") as fh:
            gamma = json.load(fh)
    except Exception:
        print("sem gamma.json (corre primeiro o workflow Gamma)")
    L = build(d5, d1h, gamma, pd.Timestamp(now).tz_convert("America/New_York"))
    os.makedirs("docs", exist_ok=True)
    with open("docs/tv_levels.txt", "w") as fh:
        fh.write("\n".join(f"{n}={p}" for n, p in L))
    with open("docs/tv_levels.json", "w") as fh:
        json.dump({"updated": now.isoformat(), "levels": [{"name": n, "price": p} for n, p in L]}, fh, ensure_ascii=False, indent=1)
    print("ok", len(L), "níveis")


if __name__ == "__main__":
    main()
