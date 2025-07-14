import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage
from os.path import exists
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import defaultdict


def precision_at_k(hits, correct_class, k):
    """Return True if the correct class appears in the top-k hits."""
    top_k_hits = hits[:k]
    return any(hit["class_name"] == correct_class for hit in top_k_hits)


def evaluate_model(results):
    """Evaluate precision@k and average query time."""
    evaluation_data = []

    for result in results:
        model_path = result["model_path"]
        category = result["category"]
        class_name = result["class_name"]
        topk = result["topk"]
        hits = result["hits"]
        time = result["time_second"]

        correct_class = class_name

        evaluation_data.append(
            {
                "model_path": model_path,
                "category": category,
                "class_name": class_name,
                "topk": topk,
                "precision_at_1": precision_at_k(hits, correct_class, 1),
                "precision_at_3": precision_at_k(hits, correct_class, 3),
                "precision_at_5": precision_at_k(hits, correct_class, 5),
                "precision_at_10": precision_at_k(hits, correct_class, 10),
                "query_time": time,
            }
        )

    df = pd.DataFrame(evaluation_data)

    grouped = (
        df.groupby(["model_path", "category", "class_name"])
        .agg(
            total_queries=("topk", "count"),
            avg_time=("query_time", "mean"),
            precision_at_1=("precision_at_1", "mean"),
            precision_at_3=("precision_at_3", "mean"),
            precision_at_5=("precision_at_5", "mean"),
            precision_at_10=("precision_at_10", "mean"),
        )
        .reset_index()
    )

    return grouped


def plot_category_level_charts(df, output_dir):
    """Create precision@k and avg_time charts per category. Returns paths for embedding."""
    output_paths = []

    for category, group in df.groupby("category"):
        for k in [1, 3, 5, 10]:
            plt.figure(figsize=(10, 6))
            sns.barplot(
                data=group, x="class_name", y=f"precision_at_{k}", hue="model_path"
            )
            plt.title(f"Precision@{k} for Category: {category}")
            plt.ylabel("Precision")
            plt.ylim(0, 1.05)
            plt.xticks(rotation=30)
            plt.tight_layout()
            out_path = output_dir / f"{category}_precision_at_{k}.png"
            plt.savefig(out_path)
            plt.close()
            output_paths.append((str(out_path), f"G{2 + len(output_paths)*18}"))

        # Avg time per class
        plt.figure(figsize=(10, 6))
        sns.barplot(data=group, x="class_name", y="avg_time", hue="model_path")
        plt.title(f"Average Query Time for Category: {category}")
        plt.ylabel("Time (s)")
        plt.xticks(rotation=30)
        plt.tight_layout()
        out_path = output_dir / f"{category}_avg_time.png"
        plt.savefig(out_path)
        plt.close()
        output_paths.append((str(out_path), f"G{2 + len(output_paths)*18}"))

    return output_paths


def sanitize_filename(name: str) -> str:
    """Replace unsafe characters with underscores."""
    return re.sub(r"[^\w\-_.]", "_", name)


def plot_class_level_precision(df, output_dir):
    """
    Creates a line/bar chart showing precision@1,3,5,10 for each class
    under each category (e.g. pest, disease). One chart per class.
    """
    for (category, class_name), group in df.groupby(["category", "class_name"]):
        plt.figure(figsize=(8, 5))

        topk_values = [1, 3, 5, 10]
        precisions = [
            group["precision_at_1"].values[0],
            group["precision_at_3"].values[0],
            group["precision_at_5"].values[0],
            group["precision_at_10"].values[0],
        ]

        plt.plot(topk_values, precisions, marker="o", linestyle="-")
        plt.title(f"{class_name} ({category}) - Precision@K")
        plt.xlabel("Top-K")
        plt.ylabel("Precision")
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.xticks(topk_values)
        plt.tight_layout()

        # Sanitize file name
        safe_class_name = sanitize_filename(class_name)
        safe_category = sanitize_filename(category)
        file_name = f"{safe_category}_{safe_class_name}_precision_topk.png"
        path = output_dir / file_name

        plt.savefig(path)
        plt.close()


def plot_confusion_matrix_top1(results, output_dir):
    """
    Plot and save the confusion matrix for top-1 predictions.
    Highlights missing and misclassified classes.
    """
    true_labels = []
    pred_labels = []

    for result in results:
        if len(result["hits"]) > 1:
            continue
        true_class = result["class_name"]
        top1_pred = result["hits"][0]["class_name"] if result["hits"] else "None"

        true_labels.append(true_class)
        pred_labels.append(top1_pred)

    # Get sorted list of all unique classes for consistency
    all_classes = sorted(set(true_labels + pred_labels))

    cm = confusion_matrix(true_labels, pred_labels, labels=all_classes)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=all_classes,
        yticklabels=all_classes,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Top-1 Prediction)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_path = output_dir / "confusion_matrix_top1.png"
    plt.savefig(out_path)
    plt.close()

    print(f"📊 Confusion matrix saved to: {out_path.resolve()}")


def plot_confusion_matrix_top1_by_category(results, output_dir):
    """
    Plot and save confusion matrices for top-1 predictions,
    separately for each category (e.g., disease, pest).
    """
    # Step 1: Organize results by category
    category_to_labels = defaultdict(lambda: {"true": [], "pred": []})

    for result in results:
        if len(result["hits"]) > 1:
            continue
        category = result["category"]
        true_class = result["class_name"]
        pred_class = result["hits"][0]["class_name"] if result["hits"] else "None"

        category_to_labels[category]["true"].append(true_class)
        category_to_labels[category]["pred"].append(pred_class)

    # Step 2: Generate and save confusion matrix for each category
    for category, labels in category_to_labels.items():
        true_labels = labels["true"]
        pred_labels = labels["pred"]

        # Get all unique classes in this category
        all_classes = sorted(set(true_labels + pred_labels))
        cm = confusion_matrix(true_labels, pred_labels, labels=all_classes)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=all_classes,
            yticklabels=all_classes,
        )

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix (Top-1) - Category: {category}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Save
        out_path = (
            output_dir / f"confusion_matrix_top1_{sanitize_filename(category)}.png"
        )
        plt.savefig(out_path)
        plt.close()

        print(
            f"📊 Saved confusion matrix for category '{category}' to: {out_path.resolve()}"
        )


def generate_summary_sheet(evaluation_df: pd.DataFrame, xlsx_path: Path):
    """
    Create a summary sheet grouped by model_path and category,
    with average precision@k and query time.
    """
    summary_df = (
        evaluation_df.groupby(["model_path", "category"])
        .agg(
            avg_precision_at_1=("precision_at_1", "mean"),
            avg_precision_at_3=("precision_at_3", "mean"),
            avg_precision_at_5=("precision_at_5", "mean"),
            avg_precision_at_10=("precision_at_10", "mean"),
            avg_query_time=("avg_time", "mean"),
            num_classes=("class_name", "nunique"),
        )
        .reset_index()
    )

    # Append the summary to the existing Excel file
    with pd.ExcelWriter(
        xlsx_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        summary_df.to_excel(writer, index=False, sheet_name="Summary")


if __name__ == "__main__":
    file_paths = [
        "dataset/images/250610_dataset/valid_topK/clip_vit_base_patch32.json",
        "dataset/images/250610_dataset/valid_topK/dinov2_base.json",
        "dataset/images/250610_dataset/valid_topK/dinov2_large.json",
        "dataset/images/250610_dataset/valid_topK/siglip2_base_patch16_224.json",
        "dataset/images/250610_dataset/valid_topK/siglip2_large_patch16_256.json",
        "dataset/images/250610_dataset/valid_topK/clip_vit_large_patch14.json",
        "dataset/images/250610_dataset/valid_topK/tulip_b_16_224.json",
        "dataset/images/250610_dataset/valid_topK/tulip_so400m_14_384.json",
    ]
    file_paths = [Path(file_path) for file_path in file_paths]

    for file_path in file_paths:
        # 📁 Output directory
        output_dir = Path("./evaluation_outputs") / file_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # 🔄 Load results
        with file_path.open(mode="r") as f:
            results = json.load(f)

        # ✅ Evaluate results
        evaluation_df = evaluate_model(results)
        print(evaluation_df.to_string(index=False))

        # 💾 Save metrics to Excel
        xlsx_path = output_dir / f"{file_path.stem}.xlsx"
        evaluation_df.to_excel(xlsx_path, index=False, sheet_name="Detailed")

        # ➕ Add summary sheet
        generate_summary_sheet(evaluation_df, xlsx_path)

        plot_category_level_charts(evaluation_df, output_dir)

        plot_confusion_matrix_top1(results, output_dir)

        plot_confusion_matrix_top1_by_category(results, output_dir)

        print(f"\n✔ Evaluation complete! Excel saved to: {xlsx_path.resolve()}")
