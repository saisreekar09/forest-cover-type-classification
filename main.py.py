"""
Forest Cover Type Classification
Hybrid Feature Selection + Multi-Model Comparison

Models compared:
  1. Random Forest - all 54 features
  2. Random Forest - hybrid top 10 features
  3. Linear SVM - all 54 features
  4. Linear SVM - hybrid top 10 features
  5. KNN - all 54 features
  6. KNN - hybrid top 10 features
  7. XGBoost - all 54 features
  8. XGBoost - hybrid top 10 features

Run:
    pip install numpy pandas scikit-learn matplotlib xgboost
    python main.py

The script downloads Covertype from OpenML on first run and saves
CSV results, per-model confusion matrices, per-class metrics and plots
in the "results" folder. Five-fold CV remains for Hybrid Random Forest only.
"""

from pathlib import Path
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

RANDOM_STATE = 42
N_ESTIMATORS = 100
TOP_K = 10
N_SPLITS = 5
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


def make_rf():
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def get_hybrid_feature_scores(X_part, y_part):
    """Combine normalized RF importance and mutual-information scores."""
    selector = make_rf()
    selector.fit(X_part, y_part)
    rf_scores = selector.feature_importances_
    mi_scores = mutual_info_classif(
        X_part, y_part, random_state=RANDOM_STATE
    )

    rf_norm = MinMaxScaler().fit_transform(
        rf_scores.reshape(-1, 1)
    ).ravel()
    mi_norm = MinMaxScaler().fit_transform(
        mi_scores.reshape(-1, 1)
    ).ravel()

    alpha = 0.5
    hybrid_scores = alpha * rf_norm + (1 - alpha) * mi_norm

    return pd.DataFrame({
        "Feature": X_part.columns,
        "Random_Forest_Score": rf_scores,
        "Mutual_Information_Score": mi_scores,
        "RF_Normalized": rf_norm,
        "MI_Normalized": mi_norm,
        "Hybrid_Score": hybrid_scores,
    }).sort_values("Hybrid_Score", ascending=False)


def evaluate_model(name, model, X_fit, y_fit, X_eval, y_eval,
                   fit_labels=None, prediction_shift=0):
    """Fit one model and return its evaluation metrics and predictions."""
    print(f"\nTraining: {name}")
    start_fit = time.perf_counter()
    model.fit(X_fit, fit_labels if fit_labels is not None else y_fit)
    fit_seconds = time.perf_counter() - start_fit

    start_predict = time.perf_counter()
    predictions = model.predict(X_eval)
    predict_seconds = time.perf_counter() - start_predict

    predictions = np.asarray(predictions).astype(int) + prediction_shift

    result = {
        "Model": name,
        "Accuracy": accuracy_score(y_eval, predictions),
        "Macro_Precision": precision_score(
            y_eval, predictions, average="macro", zero_division=0
        ),
        "Macro_Recall": recall_score(
            y_eval, predictions, average="macro", zero_division=0
        ),
        "Macro_F1": f1_score(
            y_eval, predictions, average="macro", zero_division=0
        ),
        "Fit_Time_Seconds": fit_seconds,
        "Prediction_Time_Seconds": predict_seconds,
    }

    print(
        f"Accuracy={result['Accuracy']:.5f} | "
        f"Macro-F1={result['Macro_F1']:.5f} | "
        f"Fit={fit_seconds:.1f}s | Predict={predict_seconds:.1f}s"
    )
    return result, predictions


def main():
    if XGBClassifier is None:
        raise ImportError(
            "XGBoost is required. Install it with: pip install xgboost"
        )

    # ---------------- LOAD DATA ----------------
    print("\nLoading Covertype dataset from OpenML...")
    dataset = fetch_openml(name="covertype", version=3, as_frame=True)
    X = dataset.data.apply(pd.to_numeric).copy()
    y = pd.to_numeric(dataset.target).astype(int)

    print(f"Dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
    print("\nClass distribution:")
    print(y.value_counts().sort_index())

    # ---------------- STRATIFIED SPLIT ----------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50,
        stratify=y_temp, random_state=RANDOM_STATE
    )

    print("\nSplit sizes:")
    print(f"Training:   {len(X_train):,}")
    print(f"Validation: {len(X_val):,}")
    print(f"Testing:    {len(X_test):,}")

    # ---------------- HYBRID FEATURE SELECTION ----------------
    print("\nCalculating hybrid feature scores on training data...")
    feature_scores = get_hybrid_feature_scores(X_train, y_train)
    top_features = feature_scores["Feature"].head(TOP_K).tolist()
    feature_scores.to_csv(
        OUTPUT_DIR / "Feature_Importance.csv", index=False
    )

    print("\nSelected top 10 features:")
    for rank, feature in enumerate(top_features, 1):
        print(f"{rank:2d}. {feature}")

    pd.DataFrame({
        "Rank": range(1, TOP_K + 1),
        "Selected_Feature": top_features,
    }).to_csv(OUTPUT_DIR / "Selected_Top_10_Features.csv", index=False)

    # ---------------- PREPROCESSING FOR SVM AND KNN ----------------
    # Standardize the 10 quantitative attributes only. Leave binary
    # wilderness/soil indicators unchanged. Fit scaler on training data.
    continuous_features = [
        c for c in X.columns
        if not c.startswith("Wilderness_Area")
        and not c.startswith("Soil_Type")
    ]

    scaler = StandardScaler()
    X_train_scaled = X_train.copy().astype(np.float32)
    X_val_scaled = X_val.copy().astype(np.float32)
    X_test_scaled = X_test.copy().astype(np.float32)

    X_train_scaled.loc[:, continuous_features] = scaler.fit_transform(
        X_train[continuous_features]
    ).astype(np.float32)
    X_val_scaled.loc[:, continuous_features] = scaler.transform(
        X_val[continuous_features]
    ).astype(np.float32)
    X_test_scaled.loc[:, continuous_features] = scaler.transform(
        X_test[continuous_features]
    ).astype(np.float32)

    # ---------------- DEFINE EIGHT MODEL CONFIGURATIONS ----------------
    configs = [
        (
            "Random Forest - All 54 Features",
            make_rf(),
            X_train,
            X_val,
            None,
            0,
        ),
        (
            "Random Forest - Hybrid Top 10",
            make_rf(),
            X_train[top_features],
            X_val[top_features],
            None,
            0,
        ),
        (
            "Linear SVM - All 54 Features",
            LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=5000,
            ),
            X_train_scaled,
            X_val_scaled,
            None,
            0,
        ),
        (
            "Linear SVM - Hybrid Top 10",
            LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=5000,
            ),
            X_train_scaled[top_features],
            X_val_scaled[top_features],
            None,
            0,
        ),
        (
            "KNN - All 54 Features",
            KNeighborsClassifier(
                n_neighbors=5, weights="distance", n_jobs=-1
            ),
            X_train_scaled,
            X_val_scaled,
            None,
            0,
        ),
        (
            "KNN - Hybrid Top 10",
            KNeighborsClassifier(
                n_neighbors=5, weights="distance", n_jobs=-1
            ),
            X_train_scaled[top_features],
            X_val_scaled[top_features],
            None,
            0,
        ),
        (
            "XGBoost - All 54 Features",
            XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective="multi:softprob",
                num_class=7,
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            X_train,
            X_val,
            y_train - 1,
            1,
        ),
        (
            "XGBoost - Hybrid Top 10",
            XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective="multi:softprob",
                num_class=7,
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            X_train[top_features],
            X_val[top_features],
            y_train - 1,
            1,
        ),
    ]

    # ---------------- VALIDATION COMPARISON ----------------
    validation_rows = []
    validation_predictions = {}

    for name, model, X_fit, X_eval, fit_labels, shift in configs:
        result, predictions = evaluate_model(
            name, model, X_fit, y_train, X_eval, y_val,
            fit_labels=fit_labels,
            prediction_shift=shift,
        )
        result["Number_of_Features"] = (
            TOP_K if "Top 10" in name else X.shape[1]
        )
        validation_rows.append(result)
        validation_predictions[name] = predictions

    validation_df = pd.DataFrame(validation_rows)
    validation_df.to_csv(
        OUTPUT_DIR / "Model_Comparison_Validation.csv", index=False
    )

    print("\n========== VALIDATION MODEL COMPARISON ==========")
    print(validation_df[
        ["Model", "Number_of_Features", "Accuracy",
         "Macro_Precision", "Macro_Recall", "Macro_F1",
         "Fit_Time_Seconds", "Prediction_Time_Seconds"]
    ].to_string(index=False))

    # ---------------- FIVE-FOLD CV FOR PROPOSED HYBRID RF ----------------
    print("\nRunning 5-fold stratified CV for Hybrid Random Forest...")
    cv = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    fold_rows = []

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(X_train, y_train), start=1
    ):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_valid = X_train.iloc[valid_idx]
        y_fold_valid = y_train.iloc[valid_idx]

        fold_scores = get_hybrid_feature_scores(
            X_fold_train, y_fold_train
        )
        fold_features = fold_scores["Feature"].head(TOP_K).tolist()

        fold_model = make_rf()
        fold_model.fit(
            X_fold_train[fold_features], y_fold_train
        )
        fold_pred = fold_model.predict(
            X_fold_valid[fold_features]
        )

        fold_acc = accuracy_score(y_fold_valid, fold_pred)
        fold_f1 = f1_score(
            y_fold_valid, fold_pred, average="macro", zero_division=0
        )
        fold_rows.append({
            "Fold": fold,
            "Accuracy": fold_acc,
            "Macro_F1": fold_f1,
        })
        print(
            f"Fold {fold}: Accuracy={fold_acc:.5f}, Macro-F1={fold_f1:.5f}"
        )

    cv_df = pd.DataFrame(fold_rows)
    cv_df.to_csv(
        OUTPUT_DIR / "Cross_Validation_Results.csv", index=False
    )
    cv_summary = pd.DataFrame([{
        "Model": "Hybrid Random Forest - Top 10",
        "Folds": N_SPLITS,
        "Mean_Accuracy": cv_df["Accuracy"].mean(),
        "Std_Accuracy": cv_df["Accuracy"].std(ddof=1),
        "Mean_Macro_F1": cv_df["Macro_F1"].mean(),
        "Std_Macro_F1": cv_df["Macro_F1"].std(ddof=1),
    }])
    cv_summary.to_csv(
        OUTPUT_DIR / "Cross_Validation_Summary.csv", index=False
    )

    # ---------------- FINAL TEST COMPARISON ----------------
    # Refit preprocessing and feature ranking on train + validation only.
    X_train_final = pd.concat([X_train, X_val])
    y_train_final = pd.concat([y_train, y_val])

    final_scores = get_hybrid_feature_scores(
        X_train_final, y_train_final
    )
    final_features = final_scores["Feature"].head(TOP_K).tolist()
    final_scores.to_csv(
        OUTPUT_DIR / "Final_Feature_Importance.csv", index=False
    )

    final_scaler = StandardScaler()
    X_train_final_scaled = X_train_final.copy().astype(np.float32)
    X_test_final_scaled = X_test.copy().astype(np.float32)
    X_train_final_scaled.loc[:, continuous_features] = (
        final_scaler.fit_transform(
            X_train_final[continuous_features]
        ).astype(np.float32)
    )
    X_test_final_scaled.loc[:, continuous_features] = (
        final_scaler.transform(
            X_test[continuous_features]
        ).astype(np.float32)
    )

    final_configs = [
        (
            "Random Forest - All 54 Features",
            make_rf(), X_train_final, X_test, None, 0
        ),
        (
            "Random Forest - Hybrid Top 10",
            make_rf(), X_train_final[final_features],
            X_test[final_features], None, 0
        ),
        (
            "Linear SVM - All 54 Features",
            LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=5000,
            ),
            X_train_final_scaled, X_test_final_scaled, None, 0
        ),
        (
            "Linear SVM - Hybrid Top 10",
            LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=5000,
            ),
            X_train_final_scaled[final_features],
            X_test_final_scaled[final_features], None, 0
        ),
        (
            "KNN - All 54 Features",
            KNeighborsClassifier(
                n_neighbors=5, weights="distance", n_jobs=-1
            ),
            X_train_final_scaled, X_test_final_scaled, None, 0
        ),
        (
            "KNN - Hybrid Top 10",
            KNeighborsClassifier(
                n_neighbors=5, weights="distance", n_jobs=-1
            ),
            X_train_final_scaled[final_features],
            X_test_final_scaled[final_features], None, 0
        ),
        (
            "XGBoost - All 54 Features",
            XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                objective="multi:softprob", num_class=7,
                eval_metric="mlogloss", tree_method="hist",
                random_state=RANDOM_STATE, n_jobs=-1
            ),
            X_train_final, X_test, y_train_final - 1, 1
        ),
        (
            "XGBoost - Hybrid Top 10",
            XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                objective="multi:softprob", num_class=7,
                eval_metric="mlogloss", tree_method="hist",
                random_state=RANDOM_STATE, n_jobs=-1
            ),
            X_train_final[final_features],
            X_test[final_features], y_train_final - 1, 1
        ),
    ]

    test_rows = []
    classwise_rows = []

    labels = sorted(y.unique())
    for name, model, X_fit, X_eval, fit_labels, shift in final_configs:
        result, predictions = evaluate_model(
            name, model, X_fit, y_train_final, X_eval, y_test,
            fit_labels=fit_labels,
            prediction_shift=shift,
        )
        result["Number_of_Features"] = (
            TOP_K if "Top 10" in name else X.shape[1]
        )
        test_rows.append(result)

        # Save a detailed classification report for every model.
        safe_name = (
            name.replace(" - ", "_")
                .replace(" ", "_")
                .replace("/", "_")
        )
        report = classification_report(
            y_test, predictions, labels=labels,
            output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(
            OUTPUT_DIR / f"{safe_name}_Classification_Report.csv"
        )

        # Save per-class precision, recall, F1 and support.
        for class_label in labels:
            class_result = report[str(class_label)]
            classwise_rows.append({
                "Model": name,
                "Class": class_label,
                "Precision": class_result["precision"],
                "Recall": class_result["recall"],
                "F1_Score": class_result["f1-score"],
                "Support": class_result["support"],
            })

        # Save confusion matrix as CSV and PNG for every model.
        cm = confusion_matrix(y_test, predictions, labels=labels)
        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual_Class_{c}" for c in labels],
            columns=[f"Predicted_Class_{c}" for c in labels],
        )
        cm_df.to_csv(OUTPUT_DIR / f"{safe_name}_Confusion_Matrix.csv")

        fig, ax = plt.subplots(figsize=(9, 7))
        ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=labels
        ).plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
        ax.set_title(f"{name} - Test Confusion Matrix")
        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / f"{safe_name}_Confusion_Matrix.png",
            dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

        # Also save row-normalized confusion matrix for easier class comparison.
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = np.divide(
            cm, row_sums, out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0
        )
        fig, ax = plt.subplots(figsize=(9, 7))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm_normalized, display_labels=labels
        )
        disp.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=True)
        ax.set_title(f"{name} - Normalized Test Confusion Matrix")
        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / f"{safe_name}_Normalized_Confusion_Matrix.png",
            dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(
        OUTPUT_DIR / "Model_Comparison_Test.csv", index=False
    )

    classwise_df = pd.DataFrame(classwise_rows)
    classwise_df.to_csv(
        OUTPUT_DIR / "Per_Class_Metrics_All_Models.csv", index=False
    )

    print("\\n========== FINAL TEST MODEL COMPARISON ==========")
    print(test_df[
        ["Model", "Number_of_Features", "Accuracy",
         "Macro_Precision", "Macro_Recall", "Macro_F1",
         "Fit_Time_Seconds", "Prediction_Time_Seconds"]
    ].to_string(index=False))

    # ---------------- AUTOMATIC COMPARISON PLOTS ----------------
    plot_specs = [
        ("Accuracy", "Accuracy_Comparison.png", "Test Accuracy"),
        ("Macro_Precision", "Macro_Precision_Comparison.png", "Macro Precision"),
        ("Macro_Recall", "Macro_Recall_Comparison.png", "Macro Recall"),
        ("Macro_F1", "Macro_F1_Comparison.png", "Macro F1-score"),
        ("Fit_Time_Seconds", "Training_Time_Comparison.png", "Training Time (seconds)"),
        ("Prediction_Time_Seconds", "Prediction_Time_Comparison.png", "Prediction Time (seconds)"),
    ]

    for metric, filename, ylabel in plot_specs:
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(test_df["Model"], test_df[metric])
        ax.set_title(f"Test Set - {ylabel} Comparison")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=35)
        if "Time" in metric:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        for bar, value in zip(bars, test_df[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}" if "Time" not in metric else f"{value:.2f}",
                ha="center", va="bottom", fontsize=8, rotation=0
            )
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Combined grouped comparison of all four classification metrics.
    metric_columns = ["Accuracy", "Macro_Precision", "Macro_Recall", "Macro_F1"]
    x_pos = np.arange(len(test_df))
    width = 0.19
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, metric in enumerate(metric_columns):
        ax.bar(
            x_pos + (i - 1.5) * width, test_df[metric],
            width, label=metric.replace("_", " ")
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(test_df["Model"], rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Overall Test Performance - All Four Metrics")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "Combined_Classification_Metrics.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # Per-class precision, recall and F1 heatmaps across all eight models.
    for metric in ["Precision", "Recall", "F1_Score"]:
        pivot = classwise_df.pivot(
            index="Class", columns="Model", values=metric
        )
        fig, ax = plt.subplots(figsize=(15, 6))
        image = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"Class {c}" for c in pivot.index])
        ax.set_title(f"Per-Class {metric.replace('_', ' ')} - All Models")
        ax.set_xlabel("Classification Model")
        ax.set_ylabel("Cover Type Class")
        fig.colorbar(image, ax=ax, label=metric.replace("_", " "))
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.iloc[i, j]:.2f}",
                        ha="center", va="center", fontsize=7)
        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / f"Per_Class_{metric}_All_Models.png",
            dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

    # Final hybrid feature ranking plot.
    top_final = final_scores.head(TOP_K).sort_values("Hybrid_Score")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_final["Feature"], top_final["Hybrid_Score"])
    ax.set_xlabel("Hybrid Feature Score")
    ax.set_title("Top 10 Features - Hybrid Feature Selection")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "Hybrid_Top_10_Feature_Importance.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # Five-fold stability plot and summary are for the proposed Hybrid RF only.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cv_df["Fold"], cv_df["Accuracy"], marker="o", label="Accuracy")
    ax.plot(cv_df["Fold"], cv_df["Macro_F1"], marker="s", label="Macro F1")
    ax.set_xticks(cv_df["Fold"])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_title("Hybrid Random Forest - Five-Fold Stability")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "Five_Fold_Stability.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # Save class-wise report separately for the proposed Hybrid RF as well.
    hybrid_name = "Random_Forest_Hybrid_Top_10"
    hybrid_report_path = OUTPUT_DIR / f"{hybrid_name}_Classification_Report.csv"
    pd.read_csv(hybrid_report_path, index_col=0).to_csv(
        OUTPUT_DIR / "Hybrid_RF_Classification_Report.csv"
    )
    hybrid_cm_path = OUTPUT_DIR / f"{hybrid_name}_Confusion_Matrix.csv"
    pd.read_csv(hybrid_cm_path, index_col=0).to_csv(
        OUTPUT_DIR / "Hybrid_RF_Confusion_Matrix.csv"
    )
    import shutil
    shutil.copy2(
        OUTPUT_DIR / f"{hybrid_name}_Confusion_Matrix.png",
        OUTPUT_DIR / "Hybrid_RF_Confusion_Matrix.png"
    )

    cv_summary.to_csv(
        OUTPUT_DIR / "Hybrid_RF_CV_Summary.csv", index=False
    )

    print("\nCross-validation summary for proposed model:")
    print(cv_summary.to_string(index=False))
    print(f"\nAll result files saved in: {OUTPUT_DIR.resolve()}")
    print("Project execution completed.")


if __name__ == "__main__":
    main()
