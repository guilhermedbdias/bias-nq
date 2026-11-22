"""Teste de ablação: bias completo vs. sem cada módulo vs. só cada módulo."""
import json, math, os
from datetime import datetime, timezone
import pandas as pd
import backtest as bt

TH = 15  # score >= +15 conta como sinal bullish, <= -15 como bearish


def scaled(S, mods):
    w = sum(bt.W[k] for k in mods)
    c = sum((S[k].clip(-100, 100) / 100 * bt.W[k]).round() for k in mods)
    return c / w * 100  # reescala para -100..+100 mesmo sem alguns módulos


def evaluate(score, nxt):
    ok = nxt.notna() & score.notna()
    s, r = score[ok], nxt[ok]
    up = float((r > 0).mean() * 100)
    bull, bear = s >= TH, s <= -TH

    def blk(m, win, base):
        n = int(m.sum())
        if n < 20:
            return {"n": n}
        p = float(win[m].mean() * 100)
        se = math.sqrt(p / 100 * (1 - p / 100) / n) * 100
        return {"n": n, "win": round(p, 1), "vs_base": round(p - base, 1), "se": round(se, 1)}

    nb, ns = int(bull.sum()), int(bear.sum())
    base_c = (nb * up + ns * (100 - up)) / max(nb + ns, 1)
    return {"base_sobe": round(up, 1),
            "bull": blk(bull, r > 0, up),
            "bear": blk(bear, r < 0, 100 - up),
            "comb": blk(bull | bear, (bull & (r > 0)) | (bear & (r < 0)), base_c)}


def run(S, nxt):
    allm = list(bt.W)
    res = {"dias": int(nxt.notna().sum()), "completo": evaluate(scaled(S, allm), nxt), "modulos": []}
    for k in allm:
        row = {"k": k, "nome": bt.NAMES[k], "testavel": not bool((S[k] == 0).all())}
        if row["testavel"]:
            row["sem"] = evaluate(scaled(S, [m for m in allm if m != k]), nxt)
            row["so"] = evaluate(S[k].clip(-100, 100), nxt)
        res["modulos"].append(row)
    return res


def main():
    nq = bt.hist("NQ=F")
    args = [bt.hist("^VXN"), bt.hist("^TNX"), bt.hist("DX-Y.NYB"), bt.hist("^VIX"), bt.hist("^NDX", "12y")]
    techs = []
    for t in ["NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL"]:
        try:
            techs.append(bt.hist(t))
        except Exception:
            pass
    S, _, nxt = bt.build(nq, *args, techs)
    mid = 60 + (len(S) - 60) // 2
    res = {"gerado": datetime.now(timezone.utc).isoformat(),
           "inicio": str(S.index[60].date()), "fim": str(S.index[-2].date()),
           "tudo": run(S.iloc[60:], nxt.iloc[60:]),
           "metade1": run(S.iloc[60:mid], nxt.iloc[60:mid]),
           "metade2": run(S.iloc[mid:], nxt.iloc[mid:])}
    os.makedirs("docs", exist_ok=True)
    with open("docs/ablation.json", "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res["tudo"]["completo"], ensure_ascii=False))


if __name__ == "__main__":
    main()
