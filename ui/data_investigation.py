import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import Dict

from src.core.config import Config


def get_category_counts(base_dir: Path) -> Dict[str, int]:
    """Counts the number of image files in each subdirectory of a given directory."""
    counts = {}
    if not base_dir.is_dir():
        return counts

    for category_dir in base_dir.iterdir():
        if category_dir.is_dir():
            image_dir = category_dir / "images"
            if image_dir.is_dir():
                image_files = [
                    f
                    for f in image_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in Config.VALID_IMAGE_EXTENSIONS
                ]
                counts[category_dir.name] = len(image_files)
    return counts


def create_report_dashboard(title: str, category_name: str, counts: Dict[str, int]):
    """Creates a dashboard with a table, chart, and summary statistics."""
    st.header(f"{category_name.capitalize()} Report")

    if not counts:
        st.warning(f"No data found for the '{category_name}' category.")
        return

    df = pd.DataFrame(list(counts.items()), columns=["Subcategory", "Image Count"])
    df = df.sort_values("Image Count", ascending=False).reset_index(drop=True)

    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        st.subheader("Data Overview")
        st.dataframe(
            df,
            use_container_width=True,
            height=400,
            column_config={
                "Subcategory": st.column_config.TextColumn(
                    "Subcategory", width="large"
                ),
                "Image Count": st.column_config.NumberColumn("Images", format="%d"),
            },
        )

    with col2:
        st.subheader("Summary")
        total_images = df["Image Count"].sum()
        num_subcategories = len(df)
        avg_images = total_images / num_subcategories if num_subcategories > 0 else 0

        st.metric("Total Images", f"{total_images:,}")
        st.metric("Number of Subcategories", f"{num_subcategories}")
        st.metric("Average Images per Subcategory", f"{avg_images:.1f}")

    # Create visualization
    st.subheader("Image Distribution")
    fig = px.bar(
        df.head(20),  # Show top 20 for cleaner visualization
        x="Image Count",
        y="Subcategory",
        orientation="h",
        title=f"Top 20 Subcategories by Image Count",
        color="Image Count",
        color_continuous_scale=px.colors.sequential.Viridis,
        labels={"Subcategory": category_name, "Image Count": "Number of Images"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)


def create_reports_page(cfg: Config):
    """Creates a consolidated reports page with selectable categories."""
    st.title("📊 Dataset Reports")
    st.markdown("Explore statistics and distributions of the image datasets.")

    try:
        available_categories = [
            d.name for d in cfg.PROCESSED_DATASET_DIR.iterdir() if d.is_dir()
        ]
    except FileNotFoundError:
        st.error(f"Dataset directory not found: {cfg.PROCESSED_DATASET_DIR}")
        return

    if not available_categories:
        st.warning("No data categories found in the processed dataset directory.")
        return

    # Create tabs for each category
    tabs = st.tabs([cat.capitalize() for cat in available_categories])

    for i, category in enumerate(available_categories):
        with tabs[i]:
            category_path = cfg.PROCESSED_DATASET_DIR / category
            counts = get_category_counts(category_path)
            create_report_dashboard(f"{category.capitalize()} Report", category, counts)
