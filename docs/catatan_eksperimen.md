# Catatan Eksperimen

Detail setiap run (semua kombinasi hyperparameter, skor per fold, artefak) ada di dashboard MLflow,
experiment `klasifikasi-komentar-ig`. File di `results/metrics/` dan `results/figures/` adalah salinan
yang ikut di-commit untuk paper.

## 2026-11-02: Baseline 3 model (Bim)

**Setup bersama** (`configs/default.yaml`): split stratified 80/20 (5.252 latih / 1.314 uji, seed 42).
Hyperparameter dipilih dengan stratified 5-fold CV di data latih, skor macro-F1. Data uji hanya dipakai
sekali untuk konfigurasi terbaik. Preprocessing ringan (`src/preprocess.py`), fitur TF-IDF
word 1-2 gram / char_wb 2-5 gram / gabungan.

| Model | Konfigurasi terbaik (CV) | CV macro-F1 | Test macro-F1 | Akurasi | Fit (s) |
|---|---|---|---|---|---|
| SVM (LinearSVC) | word+char, C=0.3, class_weight=balanced | 0.594 ± 0.031 | **0.578** | **0.768** | 0.3 |
| Naive Bayes | word+char, MultinomialNB, alpha=0.01 | 0.561 ± 0.044 | 0.545 | 0.717 | 0.2 |
| XGBoost | word+char, max_depth=8, sample_weight=balanced | 0.565 | 0.538 | 0.762 | 28.7 (GPU) |

F1 per kelas di data uji:

| Kelas (n uji) | SVM | NB | XGBoost |
|---|---|---|---|
| TUDUH (464) | 0.793 | 0.751 | 0.771 |
| KRITIK (297) | 0.709 | 0.641 | 0.696 |
| TANYA (281) | 0.821 | 0.779 | 0.837 |
| REAKSI (238) | 0.800 | 0.746 | 0.802 |
| LAIN (22) | 0.000 | 0.121 | 0.000 |
| DUKUNG (12) | 0.348 | 0.231 | 0.125 |

**Temuan untuk Discussion**
- Fitur n-gram karakter konsisten lebih baik daripada kata saja di ketiga model
  (CV terbaik word → word+char: SVM 0.568 → 0.594, NB 0.488 → 0.561, XGB 0.521 → 0.565). Ini wajar
  untuk teks slang dan salah ketik.
- Penanganan ketidakseimbangan membantu: class_weight=balanced pada SVM (0.571 → 0.594) dan
  sample_weight=balanced pada XGBoost (0.542 → 0.565). ComplementNB justru kalah dari MultinomialNB.
- Empat kelas besar sudah cukup baik (F1 0.70-0.84). Macro-F1 ditarik turun oleh LAIN dan DUKUNG.
  LAIN tidak punya pola leksikal yang khas (kategori "sisa"); 41% diprediksi REAKSI oleh SVM.
  DUKUNG sering terbaca sebagai KRITIK (5 dari 12): kemungkinan sarkasme atau label yang ambigu,
  perlu analisis kesalahan manual (`results/metrics/*_test_pred.csv`).
- Data uji hanya punya 12 DUKUNG dan 22 LAIN, sehingga F1 kelas tersebut sangat tidak stabil (1 prediksi
  bisa menggeser F1 beberapa poin). Contohnya NB: memperluas grid alpha ke 0.01 sedikit menaikkan
  CV (0.5597 → 0.5606) tetapi menurunkan test (0.559 → 0.545). Konfigurasi tetap dipilih
  berdasarkan CV, bukan test.
- XGBoost ±85x lebih lambat dilatih dan ±5x lebih lambat memprediksi daripada SVM/NB, tanpa
  keunggulan macro-F1.

**Catatan / keterbatasan**
- Grid XGBoost kecil (12 kombinasi). max_depth=8 berada di batas atas grid.
- Perbedaan antar model belum diuji signifikansinya (rencana: McNemar pada prediksi uji, atau paired
  test pada skor per fold di `*_cv.csv`; fold sama untuk semua model).
- 96% data berasal dari 2 post; generalisasi lintas post belum diuji.
