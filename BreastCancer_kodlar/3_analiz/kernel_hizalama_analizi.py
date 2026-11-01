"""Hakem 2, Yorum 14: Kuantum durum mesafesi ile Öklid mesafesi arasındaki korelasyon ve
kuantum çekirdeği ile RBF çekirdeği arasındaki çekirdek hizalaması (Wine ve Breast Cancer, 5-katlı CV).
Açısal kodlama (AngleEmbedding, R_X) dolanıklık içermediğinden sadakat analitik olarak
k(x,x') = prod_i cos^2((x_i - x'_i)/2) şeklinde hesaplanır (Adjoint devresiyle birebir aynı)."""
import json, numpy as np
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from scipy.stats import spearmanr, pearsonr

def pipe(Xtr):
    sc = StandardScaler().fit(Xtr); p = PCA(4, random_state=42).fit(sc.transform(Xtr))
    mm = MinMaxScaler((-np.pi, np.pi)).fit(p.transform(sc.transform(Xtr)))
    return mm.transform(p.transform(sc.transform(Xtr)))
def qk(A, B):
    d = A[:, None, :] - B[None, :, :]; return np.prod(np.cos(d / 2) ** 2, axis=2)
def align(K1, K2): return np.sum(K1 * K2) / np.sqrt(np.sum(K1 * K1) * np.sum(K2 * K2))
def calign(K1, K2):
    n = len(K1); H = np.eye(n) - 1 / n; return align(H @ K1 @ H, H @ K2 @ H)

out = {}
for name, ld in [("Wine", load_wine), ("BreastCancer", load_breast_cancer)]:
    d = ld(); X, y = d.data, d.target
    if name == "BreastCancer": y = (y == 0).astype(int)
    r = {k: [] for k in ["spearman", "pearson", "align_rbf_scale", "centered_align_rbf_scale", "align_rbf_g025"]}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        A = pipe(X[tr]); Kq = qk(A, A); Dq = np.sqrt(np.maximum(0, 2 - 2 * Kq)); De = euclidean_distances(A)
        iu = np.triu_indices(len(A), 1)
        r["spearman"].append(spearmanr(Dq[iu], De[iu]).correlation); r["pearson"].append(pearsonr(Dq[iu], De[iu])[0])
        g = 1 / (A.shape[1] * A.var()); Kr = rbf_kernel(A, gamma=g)
        r["align_rbf_scale"].append(align(Kq, Kr)); r["centered_align_rbf_scale"].append(calign(Kq, Kr))
        r["align_rbf_g025"].append(align(Kq, rbf_kernel(A, gamma=0.25)))
    out[name] = {k: [float(np.mean(v)), float(np.std(v))] for k, v in r.items()}
    print(name, {k: f"{m:.3f} ± {s:.3f}" for k, (m, s) in out[name].items()})
json.dump(out, open("kernel_hizalama_sonuclari.json", "w"), indent=1)
