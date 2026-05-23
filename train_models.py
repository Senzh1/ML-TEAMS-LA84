import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessing import load_dataset, preprocess, split_data, save_scaler, FEATURE_LABELS

ARTIFACT_DIR  = "artifacts"
CLASS_NAMES   = ["Rendah (0)", "Sedang (1)", "Tinggi (2)"]
BASELINE_SVM  = 85.71
BASELINE_RF   = 83.33
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# Metrik kustom
def sensitivity_specificity(y_true, y_pred, n_classes=3):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    sens_list, spec_list = [], []
    for i in range(n_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP
        sens_list.append(TP / (TP + FN) if (TP + FN) > 0 else 0.0)
        spec_list.append(TN / (TN + FP) if (TN + FP) > 0 else 0.0)
    return np.mean(sens_list), np.mean(spec_list), sens_list, spec_list


def evaluate_model(name, model, X_test, y_test):
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    macro_sens, macro_spec, per_sens, per_spec = sensitivity_specificity(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred)
    report   = classification_report(y_test, y_pred, target_names=CLASS_NAMES)

    print(f"\n{'='*60}")
    print(f"  MODEL: {name}")
    print(f"{'='*60}")
    print(f"  Accuracy    : {accuracy:.2f}%")
    print(f"  Sensitivity : {macro_sens*100:.2f}%  (macro-avg)")
    print(f"  Specificity : {macro_spec*100:.2f}%  (macro-avg)")
    print(f"\n  Per-Class:")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"    {cls:15s}  Sensitivity: {per_sens[i]*100:.2f}%  |  Specificity: {per_spec[i]*100:.2f}%")
    print(f"\n  Classification Report:\n{report}")

    return {"name": name, "model": model, "accuracy": accuracy,
            "sensitivity": macro_sens*100, "specificity": macro_spec*100,
            "cm": cm, "y_pred": y_pred}


# Plot helpers
def plot_confusion_matrix(cm, name, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Rendah","Sedang","Tinggi"],
                yticklabels=["Rendah","Sedang","Tinggi"],
                ax=ax, linewidths=0.5)
    ax.set_title(f"Confusion Matrix — {name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Aktual"); ax.set_xlabel("Prediksi")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  [SAVED] {path}")


def plot_feature_importance(rf_model, feature_cols, path):
    imp     = rf_model.feature_importances_
    idx     = np.argsort(imp)[::-1]
    labels  = [FEATURE_LABELS.get(feature_cols[i], feature_cols[i]) for i in idx]
    values  = imp[idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors  = sns.color_palette("viridis", len(labels))
    bars    = ax.barh(labels[::-1], values[::-1], color=colors[::-1],
                      edgecolor="white", height=0.65)
    for bar, v in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{v:.4f}", va="center", fontsize=8)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title("Feature Importance — Random Forest", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, values.max() * 1.18)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  [SAVED] {path}")
    return list(zip([feature_cols[i] for i in idx], values))


def plot_comparison(results, path):
    metrics = ["accuracy", "sensitivity", "specificity"]
    labels  = ["Accuracy (%)", "Sensitivity (%)", "Specificity (%)"]
    x, w    = np.arange(len(metrics)), 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, res in enumerate(results):
        vals = [res[m] for m in metrics]
        bars = ax.bar(x + i*w, vals, w, label=res["name"],
                      color=["#2196F3","#FF5722"][i], edgecolor="white", alpha=0.87)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)
    ax.axhline(BASELINE_SVM, color="red",  linestyle="--", lw=1.4,
               label=f"Baseline SVM Ahuja 2019 ({BASELINE_SVM}%)")
    ax.axhline(BASELINE_RF,  color="navy", linestyle=":",  lw=1.4,
               label=f"Baseline RF Ahuja 2019 ({BASELINE_RF}%)")
    ax.set_xticks(x + w/2); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Score (%)"); ax.set_ylim(0, 115)
    ax.set_title("Perbandingan Performa Model vs. Baseline", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  [SAVED] {path}")


# Main
def train_and_evaluate(csv_path="StressLevelDataset.csv"):
    print("\n[STEP 1] Memuat dan memproses dataset ...")
    df = load_dataset(csv_path)
    X, y, scaler, feature_cols = preprocess(df, fit_scaler=True)
    X_train, X_test, y_train, y_test = split_data(X, y)
    save_scaler(scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(ARTIFACT_DIR, "feature_cols.pkl"))

    print(f"  Sampel    : {len(df)}")
    print(f"  Fitur     : {len(feature_cols)}")
    print(f"  Train/Test: {X_train.shape[0]} / {X_test.shape[0]}")
    for cls in range(3):
        lbl = ["Rendah","Sedang","Tinggi"][cls]
        print(f"  Kelas {cls} ({lbl}): {(y==cls).sum()} sampel")

    print("\n[STEP 2] Melatih model ...")
    print("  -> Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None,
        min_samples_split=3, min_samples_leaf=1,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    print("  -> SVM ...")
    svm = SVC(kernel="rbf", C=10.0, gamma="scale",
              class_weight="balanced", probability=True, random_state=42)
    svm.fit(X_train, y_train)

    print("\n[STEP 3] Evaluasi model ...")
    rf_res  = evaluate_model("Random Forest", rf,  X_test, y_test)
    svm_res = evaluate_model("SVM",           svm, X_test, y_test)

    print("\n[STEP 4] Menyimpan model ...")
    joblib.dump(rf,  os.path.join(ARTIFACT_DIR, "random_forest.pkl"))
    joblib.dump(svm, os.path.join(ARTIFACT_DIR, "svm.pkl"))
    print(f"  [SAVED] {ARTIFACT_DIR}/random_forest.pkl")
    print(f"  [SAVED] {ARTIFACT_DIR}/svm.pkl")

    print("\n[STEP 5] Membuat grafik ...")
    plot_confusion_matrix(rf_res["cm"],  "Random Forest",
                          os.path.join(ARTIFACT_DIR, "cm_rf.png"))
    plot_confusion_matrix(svm_res["cm"], "SVM",
                          os.path.join(ARTIFACT_DIR, "cm_svm.png"))
    fi_ranking = plot_feature_importance(
        rf, feature_cols, os.path.join(ARTIFACT_DIR, "feature_importance.png"))
    plot_comparison([rf_res, svm_res],
                    os.path.join(ARTIFACT_DIR, "comparison.png"))

    print("\n[STEP 6] Feature Importance (Top 10):")
    print(f"  {'Rank':<5} {'Fitur':<32} {'Label':<28} {'Importance':>10}")
    print("  " + "-"*78)
    for rank, (feat, imp) in enumerate(fi_ranking[:10], 1):
        lbl = FEATURE_LABELS.get(feat, feat)
        print(f"  {rank:<5} {feat:<32} {lbl:<28} {imp:>10.4f}")

    print("\n" + "="*60)
    print("  RINGKASAN vs. BASELINE (Ahuja & Banga, 2019)")
    print("="*60)
    for res in [rf_res, svm_res]:
        base = BASELINE_RF if "Forest" in res["name"] else BASELINE_SVM
        diff = res["accuracy"] - base
        mark = "MELAMPAUI" if diff >= 0 else "Di bawah"
        print(f"  {res['name']:20s} | Acc: {res['accuracy']:.2f}% | "
              f"Baseline: {base}% | Delta: {diff:+.2f}% | {mark}")
    print("="*60)
    print(f"\nSelesai. Artefak tersimpan di: {ARTIFACT_DIR}/")
    return rf, svm, scaler, feature_cols


if __name__ == "__main__":
    train_and_evaluate("StressLevelDataset.csv")
