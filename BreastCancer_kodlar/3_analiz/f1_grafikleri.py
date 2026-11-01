"""Hakem 1, Yorum 9: Her veri seti için yöntem bazlı F1 karşılaştırma grafikleri (Makale Şekil 1–3).
Değerler makaledeki çizelgelerden alınmıştır (5-katlı CV ortalaması ± SD; YSA ve AE için 30. epoch,
Wine çok sınıflı sınıflandırıcı için 50. epoch). Çizelgeler güncellenirse değerleri burada değiştirin."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
C, Q = "#4C72B0", "#55A868"
D = {
"Wine": [("SVM", [("Klasik\nSVM\n(Linear)", 97.93, 1.69, "c"), ("Klasik\nSVM\n(RBF)", 100.00, 0.00, "c"), ("QSVM\n(Adjoint)", 93.83, 9.02, "q"), ("QSVM\n(Swap\nTest)", 93.83, 9.02, "q")]),
         ("k-NN", [("Klasik k-NN", 97.89, 2.83, "c"), ("QKNN\n(Adjoint)", 95.14, 6.75, "q"), ("QKNN\n(Swap\nTest)", 95.14, 6.75, "q")]),
         ("Yapay Sinir Ağı", [("Klasik ANN", 68.11, 15.89, "c"), ("Klasik\nKontrol", 90.07, 12.88, "c"), ("QNN\n(2 Kübit)", 63.19, 16.53, "q"), ("QNN\n(4 Kübit)", 87.74, 13.44, "q"), ("QNN\n(8 Kübit)", 97.22, 1.74, "q")]),
         ("Kümeleme", [("Klasik\nK-Means", 97.19, 1.79, "c"), ("Kuantum\nÇekirdek", 90.95, 3.27, "q"), ("Swap-Test\nQ-Means", 93.86, 3.17, "q")]),
         ("Çok Sınıflı\nSınıflandırıcı", [("Klasik\nNN\n(4-8-8-3)", 97.31, 0.14, "c"), ("Kuantum\nQNN\n(2 Katman)", 82.05, 8.08, "q")])],
"Fraud": [("SVM", [("Klasik\nSVM\n(Linear)", 88.70, 5.79, "c"), ("Klasik\nSVM\n(RBF)", 89.19, 5.93, "c"), ("QSVM\n(Adjoint)", 88.10, 5.58, "q"), ("QSVM\n(Swap\nTest)", 88.10, 5.58, "q")]),
          ("k-NN", [("Klasik k-NN", 89.27, 6.20, "c"), ("QKNN\n(Adjoint)", 87.62, 5.60, "q"), ("QKNN\n(Swap\nTest)", 87.62, 5.60, "q")]),
          ("Yapay Sinir Ağı", [("Klasik ANN", 92.70, 2.29, "c"), ("Klasik\nKontrol", 92.54, 2.16, "c"), ("QNN\n(2 Kübit)", 92.53, 1.91, "q"), ("QNN\n(4 Kübit)", 92.47, 2.08, "q"), ("QNN\n(8 Kübit)", 92.64, 2.26, "q")]),
          ("Kümeleme", [("Klasik\nK-Means", 48.79, 8.52, "c"), ("Kuantum\nÇekirdek", 62.37, 9.40, "q"), ("Swap-Test\nQ-Means", 55.14, 9.59, "q")]),
          ("Autoencoder", [("Klasik AE\n(Sığ)", 69.04, 20.95, "c"), ("Klasik AE\n(Derin)", 84.18, 5.62, "c"), ("Kuantum\nQAE", 83.07, 5.02, "q")])],
"BreastCancer": [("SVM", [("Klasik\nSVM\n(Linear)", 94.68, 2.88, "c"), ("Klasik\nSVM\n(RBF)", 94.58, 2.05, "c"), ("QSVM\n(Adjoint)", 94.02, 2.72, "q"), ("QSVM\n(Swap\nTest)", 94.02, 2.72, "q")]),
                 ("k-NN", [("Klasik k-NN", 91.73, 1.86, "c"), ("QKNN\n(Adjoint)", 89.98, 1.78, "q"), ("QKNN\n(Swap\nTest)", 89.98, 1.78, "q")]),
                 ("Yapay Sinir Ağı", [("Klasik ANN", 77.38, 38.71, "c"), ("Klasik\nKontrol", 95.64, 2.06, "c"), ("QNN\n(2 Kübit)", 96.30, 2.26, "q"), ("QNN\n(4 Kübit)", 96.89, 1.60, "q"), ("QNN\n(8 Kübit)", 95.36, 2.38, "q")]),
                 ("Kümeleme", [("Klasik\nK-Means", 88.01, 2.93, "c"), ("Kuantum\nÇekirdek", 70.18, 7.27, "q"), ("Swap-Test\nQ-Means", 87.34, 1.43, "q")]),
                 ("Autoencoder", [("Klasik AE\n(Sığ)", 75.24, 15.33, "c"), ("Klasik AE\n(Derin)", 79.54, 10.74, "c"), ("Kuantum\nQAE", 11.51, 9.80, "q")])],
}
plt.rcParams.update({"font.family": "serif", "font.size": 9})
for key, fams in D.items():
    fig, ax = plt.subplots(figsize=(13, 5.0), dpi=300)
    x = 0; ticks = []; labels = []; fam_pos = []
    for fname, models in fams:
        start = x
        for lbl, m, sd, k in models:
            lo, hi = min(sd, m), min(sd, 100 - m)
            ax.bar(x, m, width=0.9, color=C if k == "c" else Q, edgecolor="#222", linewidth=0.6,
                   yerr=[[lo], [hi]], capsize=2.5, error_kw=dict(elinewidth=0.8, ecolor="#333"))
            ax.text(x, min(m + hi, 100) + 1.5, f"{m:.1f}".replace(".", ","), ha="center", va="bottom", fontsize=7.5)
            ticks.append(x); labels.append(lbl); x += 1.18
        fam_pos.append(((start + x - 1.18) / 2, fname, start - 0.5, x - 0.68)); x += 0.9
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=7.4)
    ax.set_ylim(0, 112); ax.set_ylabel("F1-Skoru (%)", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4); ax.set_axisbelow(True)
    for cx, fn, a, b in fam_pos:
        ax.annotate(fn, xy=(cx, -0.235), xycoords=("data", "axes fraction"), ha="center", va="top", fontsize=9, fontweight="bold")
        ax.annotate("", xy=(a + 0.15, -0.21), xytext=(b - 0.15, -0.21), xycoords=("data", "axes fraction"), arrowprops=dict(arrowstyle="-", color="#555", lw=0.8))
    for cx, fn, a, b in fam_pos[1:]: ax.axvline(a - 0.2, color="#bbb", lw=0.6)
    ax.legend(handles=[Patch(color=C, label="Klasik"), Patch(color=Q, label="Kuantum / Hibrit")], loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8.5, ncol=2, frameon=False)
    plt.subplots_adjust(bottom=0.28, left=0.055, right=0.995, top=0.92)
    plt.savefig(f"sekil_F1_{key}.png", dpi=300); plt.close()
print("tamam")
