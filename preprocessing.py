import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Metadata kolom dataset
ALL_FEATURES = [
    "anxiety_level",
    "self_esteem",
    "mental_health_history",
    "depression",
    "headache",
    "blood_pressure",
    "sleep_quality",
    "breathing_problem",
    "noise_level",
    "living_conditions",
    "safety",
    "basic_needs",
    "academic_performance",
    "study_load",
    "teacher_student_relationship",
    "future_career_concerns",
    "social_support",
    "peer_pressure",
    "extracurricular_activities",
    "bullying",
]
TARGET_COL = "stress_level"

FEATURE_LABELS = {
    "anxiety_level":                 "Tingkat Kecemasan",
    "self_esteem":                   "Kepercayaan Diri",
    "mental_health_history":         "Riwayat Kes. Mental",
    "depression":                    "Tingkat Depresi",
    "headache":                      "Frekuensi Sakit Kepala",
    "blood_pressure":                "Tekanan Darah",
    "sleep_quality":                 "Kualitas Tidur",
    "breathing_problem":             "Masalah Pernapasan",
    "noise_level":                   "Tingkat Kebisingan",
    "living_conditions":             "Kondisi Tempat Tinggal",
    "safety":                        "Rasa Aman",
    "basic_needs":                   "Pemenuhan Keb. Dasar",
    "academic_performance":          "Prestasi Akademik",
    "study_load":                    "Beban Belajar",
    "teacher_student_relationship":  "Hub. Dosen-Mahasiswa",
    "future_career_concerns":        "Kekhawatiran Karier",
    "social_support":                "Dukungan Sosial",
    "peer_pressure":                 "Tekanan Teman Sebaya",
    "extracurricular_activities":    "Aktivitas Ekskul",
    "bullying":                      "Perundungan (Bullying)",
}

# (min, max, default)
FEATURE_RANGES = {
    "anxiety_level":                 (0, 21, 10),
    "self_esteem":                   (0, 30, 18),
    "mental_health_history":         (0,  1,  0),
    "depression":                    (0, 27, 11),
    "headache":                      (0,  5,  2),
    "blood_pressure":                (1,  3,  2),
    "sleep_quality":                 (0,  5,  3),
    "breathing_problem":             (0,  5,  2),
    "noise_level":                   (0,  5,  2),
    "living_conditions":             (0,  5,  3),
    "safety":                        (0,  5,  3),
    "basic_needs":                   (0,  5,  3),
    "academic_performance":          (0,  5,  3),
    "study_load":                    (0,  5,  3),
    "teacher_student_relationship":  (0,  5,  3),
    "future_career_concerns":        (0,  5,  3),
    "social_support":                (0,  5,  3),
    "peer_pressure":                 (0,  5,  3),
    "extracurricular_activities":    (0,  5,  2),
    "bullying":                      (0,  5,  2),
}


def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File tidak ditemukan: {csv_path}")
    df = pd.read_csv(csv_path, sep=";")
    print(f"[INFO] Dataset dimuat: {df.shape[0]} baris x {df.shape[1]} kolom")
    return df

def preprocess(df, scaler=None, fit_scaler=True):
    df = df.copy()
    for col in ALL_FEATURES:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    X = df[ALL_FEATURES].values
    y = df[TARGET_COL].values if TARGET_COL in df.columns else None

    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, y, scaler, ALL_FEATURES


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


def preprocess_single_input(raw_input, scaler, feature_cols):
    row = [float(raw_input.get(col, 0)) for col in feature_cols]
    return scaler.transform(np.array([row]))


def save_scaler(scaler, path="artifacts/scaler.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    print(f"[INFO] Scaler disimpan: {path}")


def load_scaler(path="artifacts/scaler.pkl"):
    return joblib.load(path)
