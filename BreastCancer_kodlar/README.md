# Meme Kanseri (Breast Cancer Wisconsin, Diagnostic) – Kod Paketi

## 1_defter / BreastCancer_tum_deneyler.ipynb
Tüm Breast Cancer deneyleri tek defterde (Google Colab'da "Tümünü çalıştır"): veri hazırlama ve 5 katlamalı bölme,
kuantum sadakat matrisleri (Adjoint / Swap-Test), SVM/QSVM ve k-NN/QKNN, kümeleme (Hungarian eşleme),
ANN / Klasik Kontrol / QNN 2-4-8 (200 epoch, 30/100/200 kontrol noktaları, öğrenme eğrileri), çoklu tohum ANN çökme
analizi (5 tohum × 3 öğrenme oranı, ölü ReLU tespiti), Klasik AE / QAE anomali tespiti (PR-AUC dahil).

Hücre numaraları defterin başındaki çizelgedeki numaralarla aynıdır.

| Makale çizelgesi | Kaynak |
|---|---|
| Çizelge 14 – SVM / QSVM | defter, hücre 3 |
| Çizelge 15 – k-NN / QKNN | defter, hücre 3 |
| Çizelge 16 – ANN / Klasik Kontrol / QNN | defter, hücre 5 |
| Çizelge 17 – Çoklu tohum analizi | defter, hücre 6 |
| Çizelge 18 – Kümeleme | defter, hücre 4 |
| Çizelge 19 – Autoencoder | defter, hücre 7 |
| Çizelge 24 – Meme Kanseri satırı | 3_analiz/kernel_hizalama_analizi.py |
| Çizelge 25 – Meme Kanseri satırları | 3_analiz/bc_istatistik_testleri.py |
| Yanıt Şekli 1–3 | 2_sonuclar/*.png |

## 2_sonuclar
`bc_tum_sonuclar.json` (ham sonuçlar ve ortam bilgisi), `bc_cv_fold_indices.json` (5 katlamanın bölme indeksleri), grafikler.

## 3_analiz
- `bc_istatistik_testleri.py`: Eşleştirilmiş t-testleri ve Bonferroni düzeltmesi (bc_tum_sonuclar.json'dan).
- `kernel_hizalama_analizi.py`: Kuantum çekirdek–RBF hizalaması ve mesafe korelasyonları (Wine ve Breast Cancer).
- `f1_grafikleri.py`: Veri seti bazında yöntem gruplu F1 grafikleri.

Tüm koşumlarda random_state = 42. Ortam: Google Colab, 2 çekirdekli Intel Xeon 2,20 GHz, 13,61 GB RAM.
