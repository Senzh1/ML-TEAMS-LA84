# Prediktor Stres Akademik Mahasiswa

Aplikasi machine learning berbasis web untuk memprediksi tingkat stres akademik mahasiswa berdasarkan 20 faktor psikologis, kesehatan, lingkungan, dan akademik.

**Project Based Learning — Machine Learning**
Binus University | Kelas LA84 | Semester Genap 2025/2026

---

## Daftar Isi

- [Tentang Proyek](#tentang-proyek)
- [Struktur Proyek](#struktur-proyek)
- [Prasyarat](#prasyarat)
- [Instalasi dan Penggunaan](#instalasi-dan-penggunaan)
- [Dataset](#dataset)
- [Deskripsi Fitur](#deskripsi-fitur)
- [Model Machine Learning](#model-machine-learning)
- [Metrik Evaluasi](#metrik-evaluasi)
- [Troubleshooting](#troubleshooting)
- [Referensi](#referensi)
- [Tim Pengembang](#tim-pengembang)

---

## Tentang Proyek

Proyek ini membangun sistem prediksi tingkat stres mahasiswa menggunakan algoritma **Random Forest** dan **SVM (Support Vector Machine)**. Sistem menganalisis 20 faktor gaya hidup dan kondisi mahasiswa untuk mengklasifikasikan tingkat stres ke dalam tiga kategori:

| Kelas | Label    |
|-------|----------|
| 0     | Rendah   |
| 1     | Sedang   |
| 2     | Tinggi   |

Hasil prediksi disertai probabilitas per kelas serta rekomendasi tindakan yang dapat diambil mahasiswa. Antarmuka web dibangun menggunakan Streamlit.

---

## Struktur Proyek

```
student_stress_predictor/
├── app.py                 # Aplikasi web Streamlit (UI + prediksi)
├── preprocessing.py       # Prapemrosesan data, scaler, metadata fitur
├── train_models.py        # Training RF + SVM, evaluasi, dan plot
├── requirements.txt       # Dependensi Python
├── StressLevelDataset.csv # Dataset (1.100 baris, 21 kolom)
├── README.md              # Dokumentasi proyek
└── artifacts/             # Dibuat otomatis oleh train_models.py
    ├── random_forest.pkl
    ├── svm.pkl
    ├── scaler.pkl
    ├── feature_cols.pkl
    ├── cm_rf.png
    ├── cm_svm.png
    ├── feature_importance.png
    └── comparison.png
```

---

## Prasyarat

- Python 3.10 atau lebih baru
- pip (package manager)
- Terminal / Command Prompt

---

## Instalasi dan Penggunaan

### 1. Masuk ke direktori proyek

```bash
cd student_stress_predictor
```

### 2. Install dependensi

```bash
pip install -r requirements.txt
```

### 3. Training model

```bash
python train_models.py
```

Perintah di atas akan menghasilkan:
- Laporan evaluasi (accuracy, sensitivity, specificity) di terminal
- File model dan scaler di `artifacts/` (`.pkl`)
- Grafik confusion matrix, feature importance, dan perbandingan model di `artifacts/` (`.png`)

> Jika direktori `artifacts/` sudah berisi file model, langkah ini bisa dilewati.

### 4. Jalankan aplikasi web

```bash
python -m streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`.

---

## Dataset

| Properti         | Detail                                |
|------------------|---------------------------------------|
| Nama             | Student Stress Factors                |
| Sumber           | [Kaggle — rxnach](https://www.kaggle.com/datasets/rxnach/student-stress-factors-a-comprehensive-analysis) |
| File             | `StressLevelDataset.csv`              |
| Jumlah baris     | 1.100                                 |
| Jumlah kolom     | 21 (20 fitur + 1 target)             |
| Tipe data        | Semua numerik (integer)               |

Distribusi kelas target (`stress_level`):

| Kelas | Label   | Jumlah Sampel |
|-------|---------|---------------|
| 0     | Rendah  | 373           |
| 1     | Sedang  | 358           |
| 2     | Tinggi  | 369           |

---

## Deskripsi Fitur

### Psikologis

| Fitur                    | Skala   | Deskripsi                                   |
|--------------------------|---------|---------------------------------------------|
| `anxiety_level`          | 0–21    | Tingkat kecemasan (adaptasi GAD-7)          |
| `self_esteem`            | 0–30    | Tingkat kepercayaan diri                    |
| `mental_health_history`  | 0–1     | Riwayat gangguan kesehatan mental (biner)   |
| `depression`             | 0–27    | Tingkat depresi (adaptasi PHQ-9)            |

### Kesehatan Fisik

| Fitur               | Skala | Deskripsi                                                     |
|----------------------|-------|---------------------------------------------------------------|
| `headache`           | 0–5   | Frekuensi sakit kepala                                        |
| `blood_pressure`     | 1–3   | Tekanan darah (1 = Normal, 2 = Pre-hipertensi, 3 = Hipertensi) |
| `sleep_quality`      | 0–5   | Kualitas tidur (0 = sangat buruk, 5 = sangat baik)            |
| `breathing_problem`  | 0–5   | Frekuensi masalah pernapasan                                  |

### Lingkungan

| Fitur               | Skala | Deskripsi                                       |
|----------------------|-------|-------------------------------------------------|
| `noise_level`        | 0–5   | Tingkat kebisingan lingkungan belajar            |
| `living_conditions`  | 0–5   | Kualitas kondisi tempat tinggal                 |
| `safety`             | 0–5   | Rasa aman di lingkungan                         |
| `basic_needs`        | 0–5   | Tingkat pemenuhan kebutuhan dasar               |

### Akademik dan Sosial

| Fitur                          | Skala | Deskripsi                                |
|--------------------------------|-------|------------------------------------------|
| `academic_performance`         | 0–5   | Tingkat prestasi akademik                |
| `study_load`                   | 0–5   | Beban belajar (SKS/tugas)                |
| `teacher_student_relationship` | 0–5   | Kualitas hubungan dosen–mahasiswa        |
| `future_career_concerns`       | 0–5   | Tingkat kekhawatiran masa depan karier   |
| `social_support`               | 0–3   | Dukungan dari keluarga/teman             |
| `peer_pressure`                | 0–5   | Tekanan dari teman sebaya                |
| `extracurricular_activities`   | 0–5   | Keterlibatan dalam organisasi/ekskul     |
| `bullying`                     | 0–5   | Frekuensi pengalaman perundungan         |

---

## Model Machine Learning

### Random Forest (Model Utama)

| Parameter           | Nilai            |
|---------------------|------------------|
| `n_estimators`      | 300              |
| `max_depth`         | None (unlimited) |
| `min_samples_split` | 3                |
| `min_samples_leaf`  | 1                |
| `class_weight`      | `"balanced"`     |
| `random_state`      | 42               |

Random Forest dipilih sebagai model utama karena tahan terhadap overfitting pada dataset berukuran sedang (Breiman, 2001) dan menyediakan feature importance secara langsung.

### SVM — Support Vector Machine (Baseline)

| Parameter     | Nilai       |
|---------------|-------------|
| `kernel`      | RBF         |
| `C`           | 10.0        |
| `gamma`       | `"scale"`   |
| `class_weight`| `"balanced"`|
| `probability` | True        |
| `random_state`| 42          |

SVM digunakan sebagai model pembanding terhadap baseline dari penelitian Ahuja & Banga (2019).

---

## Metrik Evaluasi

| Metrik        | Formula              | Keterangan                                        |
|---------------|----------------------|---------------------------------------------------|
| Accuracy      | (TP + TN) / Total    | Proporsi prediksi benar secara keseluruhan        |
| Sensitivity   | TP / (TP + FN)       | Kemampuan mendeteksi kasus positif (macro-averaged) |
| Specificity   | TN / (TN + FP)       | Kemampuan mendeteksi kasus negatif (macro-averaged) |

Baseline penelitian (Ahuja & Banga, 2019):

| Model          | Accuracy Baseline |
|----------------|-------------------|
| SVM            | 85.71%            |
| Random Forest  | 83.33%            |

---

## Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `ModuleNotFoundError: No module named 'streamlit'` | Jalankan `pip install -r requirements.txt` |
| `streamlit: command not found` | Gunakan `python -m streamlit run app.py` |
| "Model belum dilatih" muncul di browser | Jalankan `python train_models.py` terlebih dahulu |
| CSV tidak terbaca dengan benar | Pastikan delimiter CSV sesuai — jika menggunakan titik koma, ubah `pd.read_csv(csv_path)` menjadi `pd.read_csv(csv_path, sep=';')` di `preprocessing.py` |
| Port 8501 sudah digunakan | Jalankan `streamlit run app.py --server.port 8502` |

---

## Referensi

1. Ahuja, R., & Banga, A. (2019). Mental stress detection in university students using machine learning algorithms. *Procedia Computer Science*, 152, 349–353. https://doi.org/10.1016/j.procs.2019.05.007
2. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324
3. Dataset: [Student Stress Factors — Kaggle](https://www.kaggle.com/datasets/rxnach/student-stress-factors-a-comprehensive-analysis)

---

## Tim Pengembang

| Nama                         | Peran            |
|------------------------------|------------------|
| Aaron Christian              | Anggota Kelompok |
| Michael Arthara Immanuel     | Anggota Kelompok |
| Mikha Agasthya               | Anggota Kelompok |

Binus University — Kelas LA84
Project Based Learning — Machine Learning, Semester Genap 2025/2026

---

*Disclaimer: Aplikasi ini hanya untuk tujuan edukasi dan penelitian, bukan alat diagnosis klinis.*
