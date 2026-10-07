# XGBoost Model — Prediksi Factor of Safety (FK) & Klasifikasi Risiko Disposal Tambang

> Bagian dari **Sistem Hybrid NARX-XGBoost** untuk prediksi multi-horizon kondisi disposal tambang (Muka Air Tanah oleh NARX, Factor of Safety oleh XGBoost).

---

## 1. Ringkasan

Notebook ini melatih **dua model XGBoost terpisah namun saling melengkapi**, berbagi fitur input dan preprocessing yang sama:

| | Model Regresi | Model Klasifikasi |
|---|---|---|
| Target | Nilai Factor of Safety (FK), kontinu | Kategori risiko (Danger / Warning / Safe) |
| Tujuan | Memprediksi angka FK presisi | Memprediksi kelas risiko langsung |
| Interpretasi | SHAP (TreeExplainer) | Confusion Matrix, ROC Curve |

Kedua model memakai **monotonic constraints** berbasis prinsip Mohr-Coulomb & kesetimbangan batas (limit equilibrium), bukan dibiarkan belajar arah hubungan secara bebas dari data — menjamin hasil model tetap konsisten secara fisika geoteknik.

---

## 2. Dataset

- **File**: `data/dataset_xgboost.csv`
- **Jumlah sampel**: 288 baris (230 training / 58 test, split 80:20 terstratifikasi)
- **Fitur input (6)**:

| Fitur | Satuan | Arah Monotonic Constraint |
|---|---|---|
| `kohesi_kpa` | kPa | +1 (FK naik searah kohesi naik) |
| `sudut_geser_deg` | ° | +1 |
| `unit_weight_kNm3` | kN/m³ | 0 (sengaja tanpa constraint — efek ganda, lihat Section 2 notebook) |
| `tinggi_disposal_m` | m | −1 |
| `osa_deg` (Overall Slope Angle) | ° | −1 |
| `mat_rl` (Muka Air Tanah) | m RL | −1 |

- **Target regresi**: `fk` (Factor of Safety, kontinu)
- **Target klasifikasi** (diturunkan dari FK, bukan kolom terpisah):

| Kategori | Rentang FK |
|---|---|
| 🔴 Danger | FK < 1,10 |
| 🟡 Warning | 1,10 ≤ FK < 1,30 |
| 🟢 Safe | FK ≥ 1,30 |

---

## 3. Metodologi

1. **Validasi & EDA** — pengecekan struktur data, korelasi fitur-target.
2. **Tidak ada feature engineering** — fitur dipakai langsung dari parameter geoteknik mentah (berbeda dari model NARX yang punya fitur turunan curah hujan kumulatif); XGBoost berbasis pohon sudah menangkap interaksi non-linear antar-fitur secara native.
3. **Tuning hyperparameter** — Optuna (Bayesian/TPE), 100 trial, 5-fold cross-validation (Stratified untuk klasifikasi), monotonic constraints diterapkan **sejak awal pencarian** (bukan ditambahkan belakangan).
4. **Training final** — pada seluruh training set dengan hyperparameter terbaik hasil Optuna.
5. **Interpretasi SHAP** (model regresi) — ranking kepentingan fitur, dependence plot + verifikasi arah monotonic constraint, waterfall plot untuk sampel individual.
6. **Perbandingan dengan vs tanpa monotonic constraint** — mengukur kuantitatif trade-off konsistensi fisika (Physics Inconsistency Index).
7. **Pengukuran efisiensi operasional** — waktu training & inferensi.

---

## 4. Hasil Evaluasi

> ⚠️ **Catatan reproducibility**: angka berikut adalah hasil eksekusi aktual notebook ini. Nilai dapat sedikit berbeda (±0,01–0,02 pada skala metrik) jika dijalankan ulang di environment dengan versi XGBoost/Optuna berbeda, meski `random_state` sudah dikunci — ini keterbatasan reproducibility yang terdokumentasi pada library tersebut, bukan indikasi bug. Cantumkan versi library persis (lihat Bagian 7) saat melaporkan hasil.

### Model Regresi (FK)

| Metrik | Target | Hasil | Status |
|---|---|---|---|
| R² | ≥ 0,80 | **0,9630** | ✅ Lulus |
| RMSE | Serendah mungkin | 0,0848 | Informatif |
| MAE | — *(lihat Bagian 6)* | 0,0625 | — |
| MAPE | — | 6,65% | — |
| VAF | — | 96,39% | — |
| Physics Inconsistency Index | — | 0,037 | — |

Benchmark literatur: Meizhou Landslide (Khan et al.) R²=0,96; Surrogate-Li R²=0,9989 — hasil model **sebanding**.

### Model Klasifikasi (Risiko)

| Metrik | Target | Hasil | Status |
|---|---|---|---|
| Accuracy | — | 0,8966 | Informatif |
| Cohen's Kappa | ≥ 0,60 (Moderate+, Landis & Koch 1977) | **0,8369** (Almost Perfect) | ✅ Lulus |
| F1-Score (macro) | ≥ 0,60 | **0,8662** | ✅ Lulus |
| AUC-ROC (OvR, macro) | ≥ 0,70 (Hosmer et al. 2013) | **0,9743** | ✅ Lulus |
| TPR / Recall (macro) | ≥ 0,60 | **0,8804** | ✅ Lulus |
| FPR (macro) | ≤ 0,30 | **0,0458** | ✅ Lulus |

Benchmark literatur Kappa: GBM-Zhou et al. (2019) = 0,7324 (Substantial) — model kita **mengungguli**.

**Keputusan akhir: 6/6 kriteria bertarget lulus → LAYAK DIGUNAKAN.**

### Ranking SHAP Feature Importance (Model Regresi)

| Rank | Fitur | Mean |SHAP| |
|---|---|---|
| 1 | `kohesi_kpa` | 0,2722 |
| 2 | `mat_rl` | 0,1630 |
| 3 | `tinggi_disposal_m` | 0,1362 |
| 4 | `sudut_geser_deg` | 0,1333 |
| 5 | `osa_deg` | 0,0742 |
| 6 | `unit_weight_kNm3` | 0,0248 |

> Catatan: ranking fitur ke-5/ke-6 (OSA vs unit weight) **sensitif terhadap environment eksekusi** dan versi dataset — lihat diskusi reproducibility di atas.

---

## 5. Efisiensi Operasional

| | Training Time | Inference Time |
|---|---|---|
| Model Regresi | 0,158 detik | 0,067 ms/sampel |
| Model Klasifikasi | 0,302 detik | 0,128 ms/sampel |

Waktu inferensi sub-milidetik per sampel — layak untuk pemantauan near-real-time pada sistem hybrid.

---

## 6. Keterbatasan & Catatan Terbuka

- **Threshold MAE/RMSE belum ditetapkan secara resmi di CONFIG** pada versi notebook ini — sedang dalam proses penentuan via analisis sensitivitas empiris (jarak FK aktual ke batas kategori risiko terdekat). Rekomendasi sementara dari backtesting: **MAE ≤ 0,07** (titik sebelum lonjakan tajam tingkat misklasifikasi risiko). Lihat riwayat diskusi metodologi untuk detail perhitungan.
- **Threshold MAPE (10%) dan VAF (80%) didefinisikan di `CONFIG` tapi tidak dipakai sebagai kriteria gating** di tabel Summary — bersifat vestigial, perlu dibersihkan atau diaktifkan secara sadar.
- Beberapa threshold (R²≥0,80) adalah **konvensi angka bulat**, bukan dikutip dari literatur spesifik — perlu disebutkan apa adanya di bagian metodologi skripsi, bukan diklaim sebagai rujukan literatur.
- Model dilatih pada 288 sampel — ukuran dataset relatif kecil untuk 6 fitur; disarankan validasi lanjutan jika dataset bertambah signifikan di masa depan.

---

## 7. Struktur Notebook

52 sel, 17 Section, format dokumentasi **PENJELASAN → DASAR/JUSTIFIKASI → MEKANISME → IMPLEMENTASI → HASIL** di setiap bagian:

1. Import Library
2. Konfigurasi Terpusat (CONFIG)
3. Memuat Dataset
4. EDA & Analisis Korelasi
5. Pemilihan Fitur, Target, & Label Risiko
6. Stratified Train-Test Split
7. Tuning Hyperparameter Regresi (Optuna)
8. Training Model Regresi Final
9. Evaluasi Regresi (R², RMSE, + metrik pelengkap)
10. Visualisasi Evaluasi Regresi
11. Interpretasi SHAP (A–E: komputasi, bar chart, beeswarm, dependence+verifikasi constraint, waterfall)
12. Tuning & Training Model Klasifikasi
13. Evaluasi Klasifikasi (A–D: metrik lengkap, confusion matrix, reliability curve, ROC+per-class)
14. Perbandingan Model: Dengan vs Tanpa Monotonic Constraint
15. Efisiensi Operasional
16. Simpan Model, Scaler, & Experiment Log
17. Summary & Status Kelayakan

---

## 8. Cara Menjalankan

```bash
pip install xgboost optuna shap scikit-learn pandas numpy matplotlib seaborn

jupyter nbconvert --to notebook --execute --inplace xgboost.ipynb
```

**Struktur folder yang dibutuhkan**:
```
project/
├── xgboost.ipynb
├── data/
│   └── dataset_xgboost.csv
├── models/XGBoost/        # output: model, scaler
├── outputs/XGBoost/       # output: figure, tabel CSV
└── logs/XGBoost/          # output: experiment_log.json
```

**Versi library saat hasil di atas dihasilkan** — *isi sesuai environment Anda sendiri saat menjalankan, demi reproducibility yang dapat diverifikasi*:
```
xgboost      : ___
optuna       : ___
scikit-learn : ___
shap         : ___
```

---

## 9. Output yang Dihasilkan

- `models/XGBoost/XGBoost_Regressor_FK.json`, `XGBoost_Classifier_Risiko.json`
- `outputs/XGBoost/*.png` — seluruh figure (scatter, residual, SHAP, confusion matrix, ROC, dsb.)
- `outputs/XGBoost/final_summary_table.csv`, `model_comparison_table.csv`, `efficiency_summary.csv`
- `logs/XGBoost/experiment_log.json` — log eksperimen lengkap (hyperparameter, seluruh metrik)

---

## 10. Referensi Utama

- Chen & Guestrin (2016) — *XGBoost: A Scalable Tree Boosting System*
- Lundberg & Lee (2017) — SHAP (SHapley Additive exPlanations)
- Landis & Koch (1977) — interpretasi Cohen's Kappa
- Hosmer, Lemeshow & Sturdivant (2013) — *Applied Logistic Regression* (interpretasi AUC-ROC)
- *Interpretable and Calibrated XGBoost Framework for Risk-Informed Probabilistic Prediction of Slope Stability* — preseden arsitektur & jumlah trial Optuna
- GBM-Zhou et al. (2019) — benchmark Cohen's Kappa
