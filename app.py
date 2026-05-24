import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Prediktor Stres Mahasiswa",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global CSS
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    /* Theme variables - light mode defaults */
    :root {
        --bg-primary: #ffffff;
        --bg-secondary: #f8fafc;
        --bg-tertiary: #f1f5f9;
        --border-color: #e2e8f0;
        --text-primary: #1e293b;
        --text-secondary: #475569;
        --text-muted: #64748b;
        --text-faint: #94a3b8;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --border-color: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #cbd5e1;
            --text-muted: #94a3b8;
            --text-faint: #64748b;
            --bg-card-dark: #1e293b;
            --border-dark: rgba(255, 255, 255, 0.08);
        }
    }

    /* Streamlit dark theme selector */
    [data-testid="stAppViewContainer"][data-theme="dark"],
    .stApp[data-theme="dark"],
    [data-theme="dark"] {
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --bg-tertiary: #334155;
        --border-color: #334155;
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --text-muted: #94a3b8;
        --text-faint: #64748b;
        --bg-card-dark: #1e293b;
        --border-dark: rgba(255, 255, 255, 0.08);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .site-header h1 {
        font-weight: 700;
        margin: 0 0 0.4rem 0; letter-spacing: -0.02em;
        color: #f8fafc;
    }
            
    .site-header p {
        color: #94a3b8;
        margin-top: 0;
        font-size: 0.95rem;
    }

    .section-heading {
        font-size: 1.2rem; font-weight: 600;
        color: var(--text-primary);
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--border-color);
    }

    .result-card {
        background-color: var(--bg-card-dark, #1e293b);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        border: 1px solid var(--border-dark, rgba(255,255,255,0.08));
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-height: 120px;
    }
    
    .border-low { border-left: 5px solid #16a34a !important; }
    .border-medium { border-left: 5px solid #d97706 !important; }
    .border-high { border-left: 5px solid #dc2626 !important; }

    .result-stress-title {
        font-size: 2.2rem; 
        font-weight: 800; 
        margin: 0;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    .result-confidence-sub {
        font-size: 1.1rem;
        margin-top: 0.4rem;
        font-weight: 500;
    }
            
    div[data-testid="stHorizontalBlock"] {
        display: flex !important;
        align-items: stretch !important;
    }

    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    div[data-testid="column"]:has(.stPyplot) {
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
    }

    div[data-testid="column"] .stPyplot img {
        width: 100% !important;
        height: auto !important;
        max-width: 400px;
        object-fit: contain !important;
    }

    .model-info-card {
        background-color: var(--bg-card-dark, #1e293b);
        padding: 1rem 2rem;
        border-radius: 12px;
        border: 1px solid var(--border-dark, rgba(255,255,255,0.08));
        display: flex;
        flex-direction: row;
        justify-content: space-between; 
        align-items: center;
        min-height: auto;
        margin-top: 1rem;
    }

    .model-info-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin-bottom: 0;
    }

    .model-info-value {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
    }
    
    .metric-card {
        background-color: var(--bg-card-dark, #1e293b);
        border: 1px solid var(--border-dark, rgba(255,255,255,0.08));
        border-radius: 10px;
        padding: 1rem 0.6rem;
        text-align: center;
        margin-top: 1rem;
    }
    .metric-card .value { font-size: 1.2rem; font-weight: 700; }
    .metric-card .label { font-size: 0.8rem; color: var(--text-muted); margin-top: 0.2rem; }

    .rec-item {
        background: var(--bg-secondary); border: 1px solid var(--border-color);
        border-left: 3px solid #3b82f6;
        border-radius: 8px !important;
        padding: 0.9rem 1rem !important; 
        margin-bottom: 0.5rem !important;
        font-size: 0.88rem !important; 
        color: var(--text-secondary, #cbd5e1) !important; 
        line-height: 1.5 !important;
        display: block !important;
    }
    .rec-item.warning { border-left-color: #f59e0b !important; }
    .rec-item.danger { border-left-color: #ef4444 !important; }

    .site-footer {
        font-size: 0.75rem; color: var(--text-faint); text-align: center;
        margin-top: 3rem; padding: 1.2rem 0; border-top: 1px solid var(--border-color);
    }

    .nav-label {
        font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.1em; color: var(--text-faint); margin-bottom: 0.3rem;
    }

    .scale-guide {
        width: 100%; font-size: 0.78rem;
        border-collapse: collapse; margin: 0.5rem 0;
    }
    .scale-guide th {
        background: var(--bg-tertiary); padding: 0.4rem 0.6rem;
        text-align: left; font-weight: 600; color: var(--text-primary);
    }
    .scale-guide td {
        padding: 0.35rem 0.6rem; border-bottom: 1px solid var(--border-color);
        color: var(--text-secondary);
    }

    .info-box {
        background: var(--bg-secondary); border: 1px solid var(--border-color);
        border-radius: 10px; padding: 1.4rem 1.5rem; margin-bottom: 1rem;
    }
    .info-box h3 {
        font-size: 1.05rem; font-weight: 600; color: var(--text-primary);
        margin: 0 0 0.6rem 0;
    }
    .info-box p {
        font-size: 0.88rem; color: var(--text-secondary); line-height: 1.6; margin: 0.4rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Constants
ARTIFACT_DIR = "artifacts"
LABEL_MAP    = {0: "RENDAH", 1: "SEDANG", 2: "TINGGI"}
CARD_CLASS   = {0: "result-low", 1: "result-medium", 2: "result-high"}

from preprocessing import FEATURE_LABELS, FEATURE_RANGES, ALL_FEATURES

FEATURE_HELP = {
    "anxiety_level":                "Skala 0-21 (GAD-7 adaptasi). Semakin tinggi = semakin cemas.",
    "self_esteem":                  "Skala 0-30. Semakin tinggi = kepercayaan diri lebih baik.",
    "mental_health_history":        "0 = Tidak ada riwayat | 1 = Ada riwayat gangguan mental.",
    "depression":                   "Skala 0-27 (PHQ-9 adaptasi). Semakin tinggi = lebih depresi.",
    "headache":                     "Frekuensi sakit kepala: 0=tidak pernah, 5=sangat sering.",
    "blood_pressure":               "1=Normal | 2=Pre-hipertensi | 3=Hipertensi.",
    "sleep_quality":                "0=sangat buruk, 5=sangat baik.",
    "breathing_problem":            "0=tidak pernah, 5=sangat sering.",
    "noise_level":                  "Kebisingan lingkungan belajar: 0=sangat tenang, 5=sangat bising.",
    "living_conditions":            "0=sangat buruk, 5=sangat baik.",
    "safety":                       "Rasa aman di lingkungan: 0=tidak aman, 5=sangat aman.",
    "basic_needs":                  "Pemenuhan kebutuhan dasar: 0=tidak terpenuhi, 5=sangat terpenuhi.",
    "academic_performance":         "0=sangat rendah, 5=sangat tinggi.",
    "study_load":                   "Beban SKS/tugas: 0=sangat ringan, 5=sangat berat.",
    "teacher_student_relationship": "Kualitas hubungan dosen-mahasiswa: 0=sangat buruk, 5=sangat baik.",
    "future_career_concerns":       "Kekhawatiran karier: 0=tidak khawatir, 5=sangat khawatir.",
    "social_support":               "Dukungan keluarga/teman: 0=tidak ada, 5=sangat besar.",
    "peer_pressure":                "Tekanan teman sebaya: 0=tidak ada, 5=sangat besar.",
    "extracurricular_activities":   "Keterlibatan ekskul/organisasi: 0=tidak ada, 5=sangat aktif.",
    "bullying":                     "Pengalaman perundungan: 0=tidak pernah, 5=sangat sering.",
}

RECOMMENDATIONS = {
    0: [
        ("info", "Pertahankan rutinitas positif Anda — kesehatan mental Anda dalam kondisi baik."),
        ("info", "Lanjutkan pola tidur dan gaya hidup aktif yang mendukung keseimbangan belajar."),
        ("info", "Manfaatkan momentum ini untuk menetapkan target akademik lebih tinggi."),
        ("info", "Luangkan waktu untuk hobi dan aktivitas kreatif — ini menjaga keseimbangan mental."),
    ],
    1: [
        ("warning", "Identifikasi sumber stres utama Anda dan buat rencana aksi konkret."),
        ("warning", "Prioritaskan kualitas tidur — hindari begadang tidak perlu."),
        ("info", "Gunakan teknik Pomodoro (25 menit fokus + 5 menit istirahat) saat belajar."),
        ("info", "Perbanyak interaksi sosial positif; ceritakan kegelisahan ke orang terpercaya."),
        ("info", "Batasi konsumsi kafein dan paparan media sosial yang memicu kecemasan."),
    ],
    2: [
        ("danger", "Segera konsultasi dengan konselor psikologi kampus atau profesional kesehatan mental."),
        ("danger", "Kurangi beban komitmen non-esensial sementara — fokus pada pemulihan."),
        ("warning", "Praktikkan teknik pernapasan dalam (4-7-8) atau meditasi mindfulness setiap hari."),
        ("warning", "Bangun sistem dukungan: hubungi keluarga, teman dekat, atau mentor akademik."),
        ("info", "Olahraga ringan 20-30 menit setiap hari terbukti menurunkan kadar kortisol."),
        ("info", "Hindari isolasi — stres tinggi memburuk jika dihadapi sendirian."),
    ],
}


@st.cache_resource
def load_artifacts():
    paths = {k: os.path.join(ARTIFACT_DIR, f)
             for k, f in [("rf","random_forest.pkl"),("svm","svm.pkl"),
                           ("sc","scaler.pkl"),("fc","feature_cols.pkl")]}
    if not os.path.exists(paths["rf"]):
        return None, None, None, None, "not_trained"
    rf   = joblib.load(paths["rf"])
    svm  = joblib.load(paths["svm"]) if os.path.exists(paths["svm"]) else None
    sc   = joblib.load(paths["sc"])
    fc   = joblib.load(paths["fc"])
    return rf, svm, sc, fc, "ok"


def predict(model, scaler, feature_cols, raw_input):
    row = [float(raw_input.get(col, 0)) for col in feature_cols]
    X   = scaler.transform(np.array([row]))
    lbl = int(model.predict(X)[0])
    prb = model.predict_proba(X)[0]
    return lbl, prb


def proba_donut(proba, predicted_label):
    labels = ["Rendah", "Sedang", "Tinggi"]
    colors = ["#22c55e", "#f59e0b", "#ef4444"]
    explode = [0.03 if i == predicted_label else 0 for i in range(3)]

    fig, ax = plt.subplots(figsize=(4, 4), facecolor="none", layout="constrained")
    
    wedges, texts, autotexts = ax.pie(
        proba, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.78, explode=explode,
        wedgeprops=dict(width=0.4, edgecolor="#1e293b", linewidth=2),
        textprops=dict(fontsize=9, fontfamily="sans-serif", color="#f8fafc"),
    )
    
    for t in autotexts:
        t.set_fontsize(8.5)
        t.set_fontweight(600)
        t.set_color("#ffffff")
        
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    return fig

# Page: Prediksi
def page_prediksi(active_model, model_name, scaler, feature_cols):
    st.markdown('<div class="section-heading">Isi Data Kondisi Anda</div>',
                unsafe_allow_html=True)

    tabs = st.tabs(["Psikologis", "Kesehatan Fisik", "Lingkungan", "Akademik & Sosial"])

    input_vals = {}

    psych_cols  = ["anxiety_level", "self_esteem", "mental_health_history", "depression"]
    health_cols = ["headache", "blood_pressure", "sleep_quality", "breathing_problem"]
    env_cols    = ["noise_level", "living_conditions", "safety", "basic_needs"]
    acad_cols   = ["academic_performance", "study_load", "teacher_student_relationship",
                   "future_career_concerns", "social_support", "peer_pressure",
                   "extracurricular_activities", "bullying"]

    def render_slider(feat, col_obj):
        lo, hi, default = FEATURE_RANGES[feat]
        label = FEATURE_LABELS.get(feat, feat)
        help_txt = FEATURE_HELP.get(feat, "")
        if feat == "mental_health_history":
            val = col_obj.selectbox(label, [0, 1],
                                    format_func=lambda x: "Tidak Ada" if x == 0 else "Ada",
                                    help=help_txt)
        elif feat == "blood_pressure":
            val = col_obj.selectbox(label, [1, 2, 3],
                                    format_func=lambda x: {1: "Normal", 2: "Pre-hipertensi", 3: "Hipertensi"}[x],
                                    help=help_txt)
        else:
            val = col_obj.slider(label, lo, hi, default, help=help_txt)
        return val

    with tabs[0]:
        c1, c2 = st.columns(2)
        for i, feat in enumerate(psych_cols):
            input_vals[feat] = render_slider(feat, c1 if i % 2 == 0 else c2)

    with tabs[1]:
        c1, c2 = st.columns(2)
        for i, feat in enumerate(health_cols):
            input_vals[feat] = render_slider(feat, c1 if i % 2 == 0 else c2)

    with tabs[2]:
        c1, c2 = st.columns(2)
        for i, feat in enumerate(env_cols):
            input_vals[feat] = render_slider(feat, c1 if i % 2 == 0 else c2)

    with tabs[3]:
        c1, c2 = st.columns(2)
        for i, feat in enumerate(acad_cols):
            input_vals[feat] = render_slider(feat, c1 if i % 2 == 0 else c2)

    st.markdown("")
    predict_btn = st.button("Prediksi Tingkat Stres", type="primary", width="stretch")

    if predict_btn:
        label, proba = predict(active_model, scaler, feature_cols, input_vals)
        confidence   = proba[label] * 100

        st.markdown("---")
        st.markdown('<div class="section-heading">Hasil Prediksi</div>',
                    unsafe_allow_html=True)

        BORDER_CLASS = {
            0: "border-low",
            1: "border-medium",
            2: "border-high"
        }

        COLOR_MAP = {
            0: "#16a34a",
            1: "#d97706",
            2: "#dc2626"
        }

        col_r, col_c = st.columns([6, 4])

        with col_r:
            # main stress
            st.markdown(f"""
            <div class="result-card {BORDER_CLASS[label]}" style="margin-bottom: 1rem;">
                <div class="result-stress-title" style="color: {COLOR_MAP[label]};">
                    {LABEL_MAP[label]}
                </div>
                <div class="result-confidence-sub">
                    Confidence Level: <b>{confidence:.1f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # stress levels
            mc1, mc2, mc3 = st.columns(3)
            for col_w, cls_lbl, cls_col, p in zip(
                [mc1, mc2, mc3], ["Rendah", "Sedang", "Tinggi"],
                ["#16a34a", "#d97706", "#dc2626"], proba
            ):
                col_w.markdown(f"""
                <div class="metric-card" style="margin-top: 0;">
                    <div class="value" style="color:{cls_col};">{p*100:.1f}%</div>
                    <div class="label">{cls_lbl}</div>
                </div>""", unsafe_allow_html=True)

            # model
            st.markdown(f"""
            <div class="model-info-card">
                <div class="model-info-label">Model Prediksi</div>
                <div class="model-info-value">{model_name}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_c:
            # donut chart
            fig = proba_donut(proba, label)
            st.pyplot(fig)
            plt.close(fig)

        # Recommendations
        st.markdown("---")
        st.markdown('<div class="section-heading">Rekomendasi</div>', unsafe_allow_html=True)

        recs = RECOMMENDATIONS[label]
        rc1, rc2 = st.columns(2)

        for i, (severity, text) in enumerate(recs):
            target_col = rc1 if i % 2 == 0 else rc2
            target_col.markdown(
                f'<div class="rec-item {severity}">{text}</div>',
                unsafe_allow_html=True
            )

        st.markdown(" ")
        st.markdown(" ")
        
        # Input summary
        with st.expander("Ringkasan Data yang Dimasukkan"):
            rows = [{"Fitur": FEATURE_LABELS.get(k, k), "Nilai": v}
                    for k, v in input_vals.items()]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# Page: Tentang Model
def page_tentang_model():
    st.markdown('<div class="section-heading">Tentang Model Machine Learning</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <h3>Random Forest <span class="tag green">Model Utama</span></h3>
        <p>
            Random Forest adalah algoritma <i>ensemble learning</i> yang bekerja dengan membangun
            ratusan decision tree secara independen, lalu menggabungkan hasil prediksi masing-masing
            tree melalui <i>majority voting</i>. Setiap tree dilatih pada subset data yang berbeda
            (teknik <i>bagging</i>), sehingga model akhir lebih stabil dan tahan terhadap overfitting
            dibandingkan satu decision tree tunggal.
        </p>
        <p>
            Keunggulan utama Random Forest dalam proyek ini adalah kemampuannya menyediakan
            <b>feature importance</b> secara langsung, yaitu skor yang menunjukkan seberapa besar
            kontribusi setiap variabel dalam menentukan prediksi. Ini membantu mengidentifikasi
            faktor gaya hidup mana yang paling berpengaruh terhadap tingkat stres mahasiswa.
        </p>
        <table class="param-table">
            <tr><th>Parameter</th><th>Nilai</th><th>Keterangan</th></tr>
            <tr><td>n_estimators</td><td>300</td><td>Jumlah decision tree dalam ensemble</td></tr>
            <tr><td>max_depth</td><td>None</td><td>Kedalaman tree tidak dibatasi</td></tr>
            <tr><td>min_samples_split</td><td>3</td><td>Minimum sampel untuk membagi node</td></tr>
            <tr><td>min_samples_leaf</td><td>1</td><td>Minimum sampel di setiap leaf node</td></tr>
            <tr><td>class_weight</td><td>balanced</td><td>Bobot kelas disesuaikan otomatis</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <h3>Support Vector Machine (SVM) <span class="tag amber">Baseline</span></h3>
        <p>
            SVM adalah algoritma klasifikasi yang bekerja dengan mencari <i>hyperplane</i> optimal
            yang memisahkan kelas-kelas data dengan margin terbesar. Untuk data yang tidak bisa
            dipisahkan secara linear, SVM menggunakan <i>kernel trick</i> (dalam proyek ini: kernel RBF)
            untuk memproyeksikan data ke dimensi yang lebih tinggi di mana pemisahan linear menjadi mungkin.
        </p>
        <p>
            SVM digunakan sebagai model pembanding karena merupakan algoritma yang sama dengan
            yang digunakan dalam penelitian baseline oleh Ahuja & Banga (2019), yang mencapai
            akurasi 85.71% pada dataset serupa. Dengan demikian, kita bisa membandingkan performa
            model kita secara langsung terhadap hasil penelitian yang sudah dipublikasikan.
        </p>
        <table class="param-table">
            <tr><th>Parameter</th><th>Nilai</th><th>Keterangan</th></tr>
            <tr><td>kernel</td><td>RBF</td><td>Radial Basis Function untuk non-linear separation</td></tr>
            <tr><td>C</td><td>10.0</td><td>Regularization parameter (toleransi terhadap misklasifikasi)</td></tr>
            <tr><td>gamma</td><td>scale</td><td>Koefisien kernel, otomatis berdasarkan jumlah fitur</td></tr>
            <tr><td>class_weight</td><td>balanced</td><td>Bobot kelas disesuaikan otomatis</td></tr>
            <tr><td>probability</td><td>True</td><td>Mengaktifkan estimasi probabilitas per kelas</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Perbandingan dengan Baseline Penelitian</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <p>
            Hasil prediksi model dibandingkan dengan baseline dari penelitian:<br>
            <b>Ahuja, R., & Banga, A. (2019).</b> <i>Mental stress detection in university students
            using machine learning algorithms.</i> Procedia Computer Science, 152, 349-353.
        </p>
        <table class="param-table">
            <tr><th>Model</th><th>Akurasi Baseline (2019)</th><th>Akurasi Proyek Ini</th><th>Selisih</th></tr>
            <tr><td>Random Forest</td><td>83.33%</td><td>88.18%</td><td>+4.85%</td></tr>
            <tr><td>SVM</td><td>85.71%</td><td>87.27%</td><td>+1.56%</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)


# Page: Performa Model
def page_performa():
    st.markdown('<div class="section-heading">Analisis Performa Model</div>',
                unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["CM Random Forest", "CM SVM", "Feature Importance"])
    with t1:
        p = os.path.join(ARTIFACT_DIR, "cm_rf.png")
        if os.path.exists(p):
            st.image(p)
        else:
            st.info("Jalankan train_models.py untuk menghasilkan grafik.")
    with t2:
        p = os.path.join(ARTIFACT_DIR, "cm_svm.png")
        if os.path.exists(p):
            st.image(p)
        else:
            st.info("Jalankan train_models.py untuk menghasilkan grafik.")
    with t3:
        p = os.path.join(ARTIFACT_DIR, "feature_importance.png")
        if os.path.exists(p):
            st.image(p)
            st.caption("Fitur dengan importance tertinggi adalah kontributor terbesar dalam prediksi tingkat stres.")
        else:
            st.info("Jalankan train_models.py untuk menghasilkan grafik.")

    p = os.path.join(ARTIFACT_DIR, "comparison.png")
    if os.path.exists(p):
        st.markdown("")
        st.image(p, caption="Perbandingan Model vs. Baseline Penelitian")


# Main
def main():
    # Header
    st.markdown("""
    <div class="site-header">
        <h1>Prediktor Stres Akademik Mahasiswa</h1>
        <p>Masukkan kondisi Anda untuk mendapatkan prediksi tingkat stres.</p>
    </div>""", unsafe_allow_html=True)
    
    rf, svm, scaler, feature_cols, status = load_artifacts()
    if status == "not_trained":
        st.error("Model belum dilatih. Jalankan `python train_models.py` terlebih dahulu.")
        st.code("python train_models.py")
        st.stop()

    # Sidebar: Navigation + Settings
    with st.sidebar:
        st.markdown('<div class="nav-label">Navigasi</div>', unsafe_allow_html=True)
        page = st.radio(
            "Halaman",
            ["Prediksi Stres", "Tentang Model", "Performa Model"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("## Pengaturan")
        use_rf = st.radio(
            "Model Prediksi:",
            ["Random Forest (Utama)", "SVM (Baseline)"]
        ) == "Random Forest (Utama)"
        active_model = rf if use_rf else (svm or rf)
        model_name   = "Random Forest" if use_rf else "SVM"

        st.markdown("---")
        st.markdown("### Panduan Skala")
        st.markdown("""
        <table class="scale-guide">
            <tr><th>Nilai</th><th>Arti Umum</th></tr>
            <tr><td>0</td><td>Tidak pernah / Tidak ada</td></tr>
            <tr><td>1-2</td><td>Rendah / Jarang</td></tr>
            <tr><td>3</td><td>Sedang / Kadang-kadang</td></tr>
            <tr><td>4-5</td><td>Tinggi / Sering</td></tr>
        </table>
        """, unsafe_allow_html=True)
        st.caption("Kecemasan: 0-21 | Depresi: 0-27 | Kepercayaan Diri: 0-30")

    # Page routing
    if page == "Prediksi Stres":
        page_prediksi(active_model, model_name, scaler, feature_cols)
    elif page == "Tentang Model":
        page_tentang_model()
    elif page == "Performa Model":
        page_performa()




if __name__ == "__main__":
    main()
