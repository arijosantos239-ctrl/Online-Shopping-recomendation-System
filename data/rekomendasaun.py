"""Sistema rekomendasaun sosa online (item-based collaborative filtering).

Uza: python rekomendasaun.py <pasta_dadus> [user_id] [total]
Ezemplu: python rekomendasaun.py ./dadus 500000221 5
"""
import sys
import numpy as np
import pandas as pd

PESU = {"view": 1.0, "cart": 3.0, "remove_from_cart": -1.0, "purchase": 5.0}


def carrega(pasta):
    klik = pd.read_csv(f"{pasta}/clickstream_synthetic.csv")
    produtu = pd.read_csv(f"{pasta}/products.csv")
    return klik, produtu


def matriz_interasaun(klik):
    klik = klik.copy()
    klik["pesu"] = klik["event_type"].map(PESU)
    m = klik.pivot_table(index="user_id", columns="product_id",
                         values="pesu", aggfunc="sum", fill_value=0.0)
    return m.clip(lower=0)


def similaridade(m):
    x = m.values.T                                  # produtu x uzuariu
    norma = np.linalg.norm(x, axis=1, keepdims=True)
    norma[norma == 0] = 1
    xn = x / norma
    s = xn @ xn.T
    np.fill_diagonal(s, 0)
    return pd.DataFrame(s, index=m.columns, columns=m.columns)


def rekomenda(user_id, m, sim, produtu, total=5):
    if user_id not in m.index:                      # uzuariu foun: produtu populár
        popular = m.sum().sort_values(ascending=False).head(total).index
        return produtu.set_index("product_id").loc[popular].assign(skor=m.sum()[popular])
    perfil = m.loc[user_id]
    skor = sim.values @ perfil.values
    skor = pd.Series(skor, index=sim.index)
    skor[perfil[perfil > 0].index] = -1             # la rekomenda buat ne'ebé uzuariu hatene ona
    top = skor.sort_values(ascending=False).head(total)
    return produtu.set_index("product_id").loc[top.index].assign(skor=top.round(3))


def avalia(klik, total=10):
    """Teste: hamoos sosa ikus husi uzuariu ida-idak, hare se sistema hatene fali."""
    k = klik.sort_values("event_time")
    sosa = k[k.event_type == "purchase"]
    ikus = sosa.groupby("user_id").tail(1)
    treinu = k.drop(ikus.index)
    m = matriz_interasaun(treinu)
    sim = similaridade(m)
    hit, n = 0, 0
    for _, r in ikus.iterrows():
        if r.user_id not in m.index:
            continue
        perfil = m.loc[r.user_id]
        skor = pd.Series(sim.values @ perfil.values, index=sim.index)
        skor[perfil[perfil > 0].index] = -1
        n += 1
        hit += r.product_id in skor.sort_values(ascending=False).head(total).index
    return hit / n if n else 0.0, n


if __name__ == "__main__":
    pasta = sys.argv[1] if len(sys.argv) > 1 else "."
    klik, produtu = carrega(pasta)
    m = matriz_interasaun(klik)
    sim = similaridade(m)
    uid = int(sys.argv[2]) if len(sys.argv) > 2 else int(m.index[0])
    tot = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    print(f"Uzuariu: {uid}")
    print(rekomenda(uid, m, sim, produtu, tot)[["product_name", "category_code", "brand", "price", "skor"]].to_string())
    taxa, n = avalia(klik)
    print(f"\nTeste (hit-rate@10): {taxa:.1%} husi uzuariu {n}")
