import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import Dict


def count_files_in_directories(base_path: str) -> Dict[str, int]:
    """Count files in each subdirectory"""
    counts = {}
    base_dir = Path(base_path)

    if not base_dir.exists():
        return counts

    for item in base_dir.iterdir():
        if item.is_dir():
            # Count all image files in the directory
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
            file_count = sum(
                1
                for f in item.rglob("*")
                if f.is_file() and f.suffix.lower() in image_extensions
            )
            counts[item.name] = file_count

    return counts


def create_disease_report_page():
    """Page 1: Disease Report with file counts"""
    st.title("🥭 Durian Disease Report")
    st.caption("📊 Disease Statistics")

    # File counting section
    disease_path = "/Users/mac/Documents/PROJECTS/image_retrieval/dataset/images/original_dataset/disease"

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📁 Disease Categories File Count")

        if st.button("🔄 Refresh File Counts", use_container_width=True):
            st.session_state.disease_file_counts = count_files_in_directories(
                disease_path
            )

        if "disease_file_counts" not in st.session_state:
            st.session_state.disease_file_counts = count_files_in_directories(
                disease_path
            )

        if st.session_state.disease_file_counts:
            # Create DataFrame for better display
            df = pd.DataFrame(
                [
                    {"Category": name, "File Count": count}
                    for name, count in st.session_state.disease_file_counts.items()
                ]
            )

            # Sort by file count descending
            df = df.sort_values("File Count", ascending=False)

            # Display as a beautiful table
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Category": st.column_config.TextColumn(
                        "Disease Category", width="medium"
                    ),
                    "File Count": st.column_config.NumberColumn(
                        "Number of Images", width="small"
                    ),
                },
            )

            # Create visualization
            fig = px.bar(
                df,
                x="File Count",
                y="Category",
                orientation="h",
                title="Distribution of Images Across Disease Categories",
                color="File Count",
                color_continuous_scale="viridis",
            )
            fig.update_layout(
                height=600,
                showlegend=False,
                xaxis_title="Number of Images",
                yaxis_title="Disease Category",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No disease categories found or directory doesn't exist.")

    with col2:
        st.subheader("📈 Summary Statistics")

        if st.session_state.disease_file_counts:
            total_files = sum(st.session_state.disease_file_counts.values())
            total_categories = len(st.session_state.disease_file_counts)
            avg_files = total_files / total_categories if total_categories > 0 else 0

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("📊 Total Images", f"{total_files:,}")
            with col_b:
                st.metric("📁 Categories", f"{total_categories}")

            st.metric("📏 Average per Category", f"{avg_files:.1f}")

            # Top categories
            if st.session_state.disease_file_counts:
                top_categories = sorted(
                    st.session_state.disease_file_counts.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]

                st.subheader("🏆 Top 5 Categories")
                for i, (category, count) in enumerate(top_categories, 1):
                    st.metric(f"{i}. {category}", f"{count} images")


def create_pest_report_page():
    """Page 2: Pest Report with file counts"""
    st.title("🥭 Durian Pest Report")
    st.caption("📊 Pest Statistics")

    # File counting section
    pest_path = "/Users/mac/Documents/PROJECTS/image_retrieval/dataset/images/original_dataset/pest"

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📁 Pest Categories File Count")

        if st.button("🔄 Refresh File Counts", use_container_width=True):
            st.session_state.pest_file_counts = count_files_in_directories(pest_path)

        if "pest_file_counts" not in st.session_state:
            st.session_state.pest_file_counts = count_files_in_directories(pest_path)

        if st.session_state.pest_file_counts:
            # Create DataFrame for better display
            df = pd.DataFrame(
                [
                    {"Category": name, "File Count": count}
                    for name, count in st.session_state.pest_file_counts.items()
                ]
            )

            # Sort by file count descending
            df = df.sort_values("File Count", ascending=False)

            # Display as a beautiful table
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Category": st.column_config.TextColumn(
                        "Pest Category", width="medium"
                    ),
                    "File Count": st.column_config.NumberColumn(
                        "Number of Images", width="small"
                    ),
                },
            )

            # Create visualization
            fig = px.bar(
                df,
                x="File Count",
                y="Category",
                orientation="h",
                title="Distribution of Images Across Pest Categories",
                color="File Count",
                color_continuous_scale="viridis",
            )
            fig.update_layout(
                height=600,
                showlegend=False,
                xaxis_title="Number of Images",
                yaxis_title="Pest Category",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No pest categories found or directory doesn't exist.")

    with col2:
        st.subheader("📈 Summary Statistics")

        if st.session_state.pest_file_counts:
            total_files = sum(st.session_state.pest_file_counts.values())
            total_categories = len(st.session_state.pest_file_counts)
            avg_files = total_files / total_categories if total_categories > 0 else 0

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("📊 Total Images", f"{total_files:,}")
            with col_b:
                st.metric("📁 Categories", f"{total_categories}")

            st.metric("📏 Average per Category", f"{avg_files:.1f}")

            # Top categories
            if st.session_state.pest_file_counts:
                top_categories = sorted(
                    st.session_state.pest_file_counts.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]

                st.subheader("🏆 Top 5 Categories")
                for i, (category, count) in enumerate(top_categories, 1):
                    st.metric(f"{i}. {category}", f"{count} images")
