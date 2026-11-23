"""Painel AMD: acumulação / manipulação / distribuição (regras objetivas sobre o NQ)."""
import json, math, os
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import yfinance as yf

NOMES = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta"]
TIPOS = ["A", "M", "D", "O"]


def hist(t, period, interval="1d"):
    d = yf.download(t, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d.dropna(subset=["Close"])


def classify(nq):
    """A = acumulação (dia dentro do anterior / pouco range); M = manipulação (varre um extremo
    do dia anterior e fecha de volta dentro do range); D = distribuição (fecha fora do range
    anterior, com range normal ou grande e fecho perto de um extremo); O = outro."""
    C, H, L = nq["Close"], nq["High"], nq["Low"]
    rng = H - L
    rel = rng / rng.rolling(20).mean().shift(1)
    pH, pL = H.shift(1), L.shift(1)
    sH, sL = H > pH, L < pL
    pos = ((C - L) / rng.where(rng > 0)).fillna(0.5)
    M = (sH | sL) & (C <= pH) & (C >= pL)
    D = ((C > pH) | (C < pL)) & (rel >= 1.0) & ((pos >= 0.7) | (pos <= 0.3))
    A = (~sH & ~sL) | ((rel < 0.8) & ~M & ~D)
    df = pd.DataFrame(index=nq.index)
    df["tipo"] = np.where(M, "M", np.where(D, "D", np.where(A, "A", "O")))
    df["lado"] = np.where(D & (C > pH), "alta", np.where(D & (C < pL), "baixa",
                 np.where(M & sH & ~sL, "max", np.where(M & sL & ~sH, "min", ""))))
    s20, s50 = C.rolling(20).mean(), C.rolling(50).mean()
    df["htf"] = np.where((C > s50) & (s20 > s50), "alta", np.where((C < s50) & (s20 < s50), "baixa", "misto"))
    df["ok"] = rel.notna() & s50.notna()
    df["wd"] = df.index.weekday
    df["prev_tipo"] = df["tipo"].shift(1)
    df["prev_htf"] = df["htf"].shift(1)
    return df


def dist(sub):
    n = len(sub)
    if n == 0:
        return {"n": 0}
    out = {"n": n}
    for t in TIPOS:
        out[t] = round(float((sub["tipo"] == t).mean() * 100), 1)
    d, m = sub[sub["tipo"] == "D"], sub[sub["tipo"] == "M"]
    out["d_alta"] = round(float((d["lado"] == "alta").mean() * 100)) if len(d) >= 20 else None
    out["m_max"] = round(float((m["lado"] == "max").mean() * 100)) if len(m) >= 20 else None
    return out


def compare(cond, base):
    if cond.get("n", 0) < 30:
        return None
    c = dict(cond)
    for t in TIPOS:
        p = cond[t] / 100
        se = math.sqrt(p * (1 - p) / cond["n"]) * 100
        c[t + "_vs"] = round(cond[t] - base[t], 1)
        c[t + "_star"] = bool(abs(cond[t] - base[t]) > 2 * se)
    return c


def ltf(today_utc):
    try:
        d = hist("NQ=F", "10d", "1h")
        d.index = d.index.tz_convert("UTC")
        day = d[d.index.date == today_utc]
        asia = day[(day.index.hour >= 0) & (day.index.hour < 6)]
        lon = day[(day.index.hour >= 6) & (day.index.hour < 12)]
        if len(asia) < 3 or len(lon) < 1:
            return None
        ah, al = float(asia["High"].max()), float(asia["Low"].min())
        lh, ll, last = float(lon["High"].max()), float(lon["Low"].min()), float(lon["Close"].iloc[-1])
        sh, sl = lh > ah, ll < al
        if sl and not sh:
            txt = ("Londres varreu os mínimos da Ásia e recuperou: possível manipulação em baixa."
                   if last > al else "Londres rompeu os mínimos da Ásia e ficou abaixo: possível distribuição em baixa.")
        elif sh and not sl:
            txt = ("Londres varreu os máximos da Ásia e recuou: possível manipulação em alta."
                   if last < ah else "Londres rompeu os máximos da Ásia e ficou acima: possível distribuição em alta.")
        elif sh and sl:
            txt = "Londres varreu os dois lados do intervalo asiático."
        else:
            txt = "Londres ainda dentro do intervalo asiático: acumulação em curso."
        return {"asia_high": round(ah, 1), "asia_low": round(al, 1), "lon_high": round(lh, 1),
                "lon_low": round(ll, 1), "ultimo": round(last, 1), "leitura": txt}
    except Exception:
        return None


def compute(nq, today_ny, today_utc):
    df = classify(nq)
    closed = df[df.index.date < today_ny]
    h = closed[closed["ok"] & closed["prev_tipo"].notna() & closed["prev_htf"].notna()]
    last = closed.iloc[-1]
    wd = today_ny.weekday()
    monday = today_ny - timedelta(days=wd) if wd <= 2 else today_ny + timedelta(days=7 - wd)
    dias, first = [], True
    for k in range(3):
        d = monday + timedelta(days=k)
        item = {"nome": NOMES[k], "data": d.strftime("%d/%m")}
        if d < today_ny:
            row = df[df.index.date == d]
            if len(row) == 0:
                item["status"] = "sem sessão"
            else:
                item.update({"status": "fechado", "tipo": str(row["tipo"].iloc[0]), "lado": str(row["lado"].iloc[0])})
        else:
            item["status"] = "hoje" if d == today_ny else "previsto"
            base = dist(h[h["wd"] == k])
            if first:
                sub = h[(h["prev_tipo"] == last["tipo"]) & (h["prev_htf"] == last["htf"])]
                item["contexto"] = f"dia anterior = {last['tipo']}, HTF {last['htf']}"
                first = False
            else:
                sub = h[(h["wd"] == k) & (h["prev_htf"] == last["htf"])]
                item["contexto"] = f"HTF {last['htf']} (depende do que acontecer antes)"
            item["base"], item["cond"] = base, compare(dist(sub), base)
        dias.append(item)
    return {"updated": datetime.now(timezone.utc).isoformat(), "ref": str(closed.index[-1].date()),
            "ultimo_tipo": str(last["tipo"]), "ultimo_lado": str(last["lado"]), "htf": str(last["htf"]),
            "dias": dias, "ltf": ltf(today_utc), "n_hist": int(len(h))}


def main():
    nq = hist("NQ=F", "10y")
    nq.index = pd.to_datetime(nq.index).tz_localize(None)
    now = datetime.now(timezone.utc)
    out = compute(nq, now.astimezone(ZoneInfo("America/New_York")).date(), now.date())
    os.makedirs("docs", exist_ok=True)
    with open("docs/amd.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out["dias"], ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()
