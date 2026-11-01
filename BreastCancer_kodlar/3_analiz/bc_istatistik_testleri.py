"""Hakem 2, Yorum 4 – Breast Cancer: klasik ve kuantum model çiftleri için katman bazlı eşleştirilmiş t-testi (F1)
ve Bonferroni düzeltmesi (Çizelge 25'in Meme Kanseri satırları). Girdi: ../2_sonuclar/bc_tum_sonuclar.json"""
import json, numpy as np
from scipy.stats import ttest_rel, wilcoxon
R = json.load(open("../2_sonuclar/bc_tum_sonuclar.json"))
S, N, C, A = R["svm_knn"], R["ann_qnn"], R["kumeleme"], R["anomali"]["sonuclar"]
nn = lambda m, ep: [x["f1"] for x in N[m]["checkpoints"][ep]]
P = [("Klasik SVM (RBF)", "QSVM (Adjoint)", S["Klasik SVM (RBF)"]["f1"], S["QSVM (Adjoint Kernel)"]["f1"]),
     ("Klasik SVM (Linear)", "QSVM (Adjoint)", S["Klasik SVM (Linear)"]["f1"], S["QSVM (Adjoint Kernel)"]["f1"]),
     ("Klasik k-NN", "QKNN (Adjoint)", S["Klasik k-NN (Öklid)"]["f1"], S["QKNN (Adjoint Kernel)"]["f1"]),
     ("Saf Klasik ANN (30 ep)", "Hibrit QNN-4 (30 ep)", nn("Saf Klasik ANN", "30"), nn("Hibrit QNN (4 Kübit)", "30")),
     ("Klasik Kontrol (30 ep)", "Hibrit QNN-4 (30 ep)", nn("Klasik Kontrol (4 boyutlu darboğaz)", "30"), nn("Hibrit QNN (4 Kübit)", "30")),
     ("Klasik Kontrol (200 ep)", "Hibrit QNN-4 (200 ep)", nn("Klasik Kontrol (4 boyutlu darboğaz)", "200"), nn("Hibrit QNN (4 Kübit)", "200")),
     ("Klasik K-Means", "Kuantum Çekirdek Kümeleme", C["Klasik K-Means"]["f1"], C["Kuantum Çekirdek Kümeleme"]["f1"]),
     ("Klasik K-Means", "Swap-Test Q-Means", C["Klasik K-Means"]["f1"], C["Swap Test K-Means"]["f1"]),
     ("Klasik AE (Derin, 30 ep)", "Kuantum QAE (30 ep)", A["Klasik AE (Derin 4-8-2-8-4)"]["30"]["f1"], A["Kuantum QAE (2 Katman)"]["30"]["f1"])]
m = len(P)
print(f"{'Model A':<26} | {'Model B':<26} | {'Fark':<14} | {'p':>7} | {'p (Bonf.)':>9} | Anlamlı")
for a, b, x, y in P:
    x, y = np.array(x) * 100, np.array(y) * 100; d = x - y
    t, p = ttest_rel(x, y); pb = min(1.0, p * m)
    print(f"{a:<26} | {b:<26} | {d.mean():5.2f} ± {d.std(ddof=1):5.2f} | {p:7.4f} | {pb:9.4f} | {'Evet' if pb < 0.05 else 'Hayır'}")
