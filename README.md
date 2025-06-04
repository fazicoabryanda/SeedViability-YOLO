# Seed Analysis Suite

## 📖 Overview

The **Seed Analysis Suite** is a comprehensive web application built with Streamlit for advanced seed analysis. It leverages deep learning (YOLO models) and sophisticated image processing techniques to provide precise assessments of seed count, viability, and purity. The application aims to offer a technological edge for superior crop performance and yield optimization.

**Version:** 1.0
**Year:** 2025 

---

## ✨ Features

* **🏠 Home Page:**
    * Introduction to the application's capabilities.
    * Easy navigation to testing modules.
    * "About" page link (navigates to `pages/About.py`).
* **🔬 Seed Testing Module:**
    * **🌱 Enhanced Seed Counting & Feature Analysis:**
        * Upload an image to detect and count seeds.
        * Individual seed bounding box detection.
        * **Feature Extraction per Seed:**
            * RGB mean values (extracted using Otsu Thresholding within each bounding box).
            * GLCM (Gray Level Co-occurrence Matrix) features: Contrast, Correlation, Energy, Homogeneity at angles 0°, 45°, 90°, 135°.
        * Displays original and processed images with detections.
        * Provides a detailed table of analysis for each seed (ID, Thumbnail, BBox, Segmentation Method, RGB means, GLCM features).
        * **📊 Interactive Feature Dashboard:**
            * Pixel Intensity Histogram (Red, Green, Blue channels).
            * Mean RGB Value Box Plot.
            * Average GLCM Feature Values Bar Chart.
            * Feature Correlation Heatmap (RGB & Average GLCM).
            * 2D PCA (Principal Component Analysis) of Seed Features.
        * Downloadable seed analysis data as a CSV file.
    * **❤️ Seed Viability Detector:**
        * Upload an image to classify seeds as 'viable' or 'non-viable'.
        * Displays original and processed images with classification labels and bounding boxes.
        * Provides counts for viable and non-viable seeds.
        * Calculates and displays the overall viability percentage.
    * **🌿 Seed Purity Analysis:**
        * Upload an image to analyze seed purity and identify contaminants.
        * Classifies items into categories: 'Pure-seed', 'Gulma' (Weed), 'Innert-mater' (Inert Matter), 'BTL' (Other Crop Seed).
        * Displays original and processed images with classification labels and bounding boxes/masks.
        * Provides counts for each category.
        * Calculates and displays the overall purity percentage.
* **📞 Contact Page:**
    * Displays contact information for the developers:
        * Fazico Rakcel Abryanda (Software Developer and AI Engineer)
        * Mira Landep Widiastuti (Researcher – Seed Science and Technology)
* **⚙️ Technical Highlights:**
    * Uses custom-trained YOLO models for detection and classification.
    * Intelligent model loading with Streamlit's caching.
    * Robust image processing pipeline including Otsu thresholding for segmentation.
    * Secure file handling for uploads.

---

## 🛠️ Technologies & Libraries Used

* **Framework:** Streamlit
* **Object Detection/Classification:** Ultralytics YOLO
* **Image Processing:** OpenCV (`cv2`), Pillow (PIL), scikit-image (`skimage`)
* **Data Handling & Numerical Operations:** Pandas, NumPy
* **Plotting & Visualization:** Matplotlib, Seaborn
* **Machine Learning (PCA):** scikit-learn (`sklearn`)
* **UI Components:** `streamlit_option_menu`

---

## 🚀 Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file with the following (or add more specific versions if known):
    ```txt
    streamlit
    opencv-python-headless
    ultralytics
    Pillow
    numpy
    pandas
    matplotlib
    seaborn
    scikit-image
    scikit-learn
    streamlit-option-menu
    # Add any other specific versions if necessary
    ```
    Then install:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ensure Model Files are Present:**
    Place the YOLO model files in the root directory of the project (or update paths in the script if they are located elsewhere):
    * `counting_model2.pt`
    * `viability_model2.pt`
    * `purity_model.pt`

5.  **Ensure Asset Files are Present:**
    Place the following asset files in the root directory (or update paths if necessary):
    * `icon.ico` (for page icon)
    * `logo.png` (for Home page)
    * `Aset1.png` (for Home page)
    * `kirito.jpg` (Fazico's profile picture on Contact page)
    * `Mira-Widiastuti-2.png` (Mira's profile picture on Contact page)

6.  **Create `pages` Directory for "About" page:**
    * Create a directory named `pages` in the root of your project.
    * Inside the `pages` directory, create an `About.py` file (the content for this page is not in the provided script, but the navigation points to it). Example:
        ```python
        # pages/About.py
        import streamlit as st

        st.set_page_config(layout="wide", page_title="About - Seed Analysis Suite")

        st.title("About the Seed Analysis Suite")
        st.write("""
        This application is designed to provide advanced analysis of seeds...
        (Add more details about the project, its purpose, acknowledgments, etc.)
        """)
        ```

7.  **Run the Streamlit application:**
    ```bash
    streamlit run your_main_script_name.py
    ```
    (Replace `your_main_script_name.py` with the actual name of your Python file, e.g., `app.py`).

---

## 💻 Usage

1.  **Navigate the Application:** Use the main menu in the sidebar ("Home", "Seed Testing", "Contact").
2.  **Seed Testing:**
    * Select the desired test type ("Counting", "Viability Test", "Purity Test") from the "Seed Testing Options" submenu in the sidebar.
    * **Upload an Image:** Click the "Choose an image..." or "Select the seed image..." button to upload a `.png`, `.jpg`, or `.jpeg` file.
    * **View Results:**
        * The application will process the image and display the original and processed images.
        * For "Counting", detailed tables and an interactive feature dashboard will be shown.
        * For "Viability" and "Purity", counts and percentages will be displayed.
    * **Download Data (Counting):** A "Download Seed Data as CSV" button will appear after analysis in the Counting section.
3.  **Explore Home & Contact:** Visit the "Home" page for an overview and the "Contact" page for developer information.

---

## 📁 File Structure (Key Files)
