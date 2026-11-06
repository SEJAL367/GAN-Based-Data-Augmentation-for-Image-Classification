
import streamlit as st
import os
import glob
import pandas as pd
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="GAN Data Augmentation Dashboard",
    page_icon="🧠",
    layout="wide"
)

# Project title
st.title("GAN-Based Data Augmentation for Image Classification")
st.markdown("### Interactive Experiment Dashboard")
st.write(
    "Explore synthetic MNIST images generated using GANs "
    "and compare image classification performance."
)

st.divider()

# Project overview
st.header("1. Project Overview")

col1, col2, col3 = st.columns(3)

col1.metric("GAN Models Trained", "10")
col2.metric("GAN Epochs", "10")
col3.metric("Classifier Epochs", "30")

st.write("""
This project uses Generative Adversarial Networks (GANs)
to generate synthetic handwritten digit images.
The generated images are combined with real MNIST images
to study their effect on image classification accuracy.
""")

st.divider()

# Synthetic image gallery
st.header("2. Generated Synthetic Images")

output_path = "outputs/trial1"

if os.path.exists(output_path):
    gan_folders = sorted(glob.glob(os.path.join(output_path, "gan*")))

    if gan_folders:
        selected_gan = st.selectbox(
            "Select GAN Model",
            gan_folders,
            format_func=lambda x: os.path.basename(x)
        )

        image_files = sorted(glob.glob(os.path.join(selected_gan, "*.jpg")))

        if image_files:
            cols = st.columns(5)

            for i, image_path in enumerate(image_files):
                with cols[i % 5]:
                    image = Image.open(image_path)
                    st.image(
                        image,
                        caption=os.path.basename(image_path),
                        use_container_width=True
                    )
        else:
            st.info("No generated images found in this folder.")
    else:
        st.warning("No GAN output folders found.")
else:
    st.warning("Output folder not found.")

st.divider()

# GAN loss graphs
st.header("3. GAN Training Loss")

loss_files = glob.glob("outputs/trial1/gan*/loss*.png")

if loss_files:
    selected_loss = st.selectbox(
        "Select GAN Loss Graph",
        sorted(loss_files)
    )
    st.image(selected_loss, caption="GAN Training Loss")
else:
    st.info("GAN loss graphs not found.")

st.divider()

# Classifier results
st.header("4. Classifier Performance")

accuracy_file = "classifier_results/trial1/accuracies"

if os.path.exists(accuracy_file):
    with open(accuracy_file, "r") as file:
        lines = file.readlines()

    # Ignore the first incomplete/previous run
    results = []

    for line in lines[1:]:
        try:
            values = line.strip().split()
            if len(values) >= 3:
                results.append({
                    "Real Images": int(values[0]),
                    "Synthetic Images": int(values[1]),
                    "Accuracy": float(values[2])
                })
        except ValueError:
            continue

    if results:
        df = pd.DataFrame(results)

        st.dataframe(df, use_container_width=True)

        st.subheader("Accuracy Comparison")
        chart_df = df.copy()
        chart_df["Experiment"] = (
            chart_df["Real Images"].astype(str)
            + " Real + "
            + chart_df["Synthetic Images"].astype(str)
            + " Synthetic"
        )

        st.bar_chart(
            chart_df.set_index("Experiment")["Accuracy"]
        )

        best_result = df.loc[df["Accuracy"].idxmax()]

        st.success(
            f"Best accuracy: {best_result['Accuracy']:.2f}% "
            f"with {int(best_result['Real Images'])} real images "
            f"and {int(best_result['Synthetic Images'])} synthetic images."
        )
    else:
        st.warning("No valid classifier results found.")
else:
    st.warning("Classifier accuracy file not found.")

st.divider()

# Classifier graphs
st.header("5. Classifier Training Graphs")

graph_files = (
    glob.glob("classifier_results/trial1/*accuracy*.png")
    + glob.glob("classifier_results/trial1/*loss*.png")
)

if graph_files:
    selected_graph = st.selectbox(
        "Select Classifier Graph",
        sorted(graph_files)
    )
    st.image(selected_graph, caption=os.path.basename(selected_graph))
else:
    st.info("Classifier graphs not found.")

st.divider()

st.caption("GAN-Based Data Augmentation for Image Classification | MNIST Dataset")
