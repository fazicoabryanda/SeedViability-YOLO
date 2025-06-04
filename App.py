import streamlit as st
from PIL import Image, UnidentifiedImageError
import cv2
from ultralytics import YOLO
import os
import sys
from werkzeug.utils import secure_filename
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from io import BytesIO
import base64
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
import seaborn as sns # Untuk heatmap dan styling
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- (OPSIONAL TAPI DIREKOMENDASIKAN) Fungsi Helper Path untuk Bundling Nanti ---
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, relative_path)

# --- Konfigurasi Aplikasi ---
st.set_page_config(layout="wide", page_title="Seed Analysis Suite", page_icon="icon.ico")

# --- Model Configuration ---
COUNTING_MODEL_FILENAME = 'counting_model2.pt'
VIABILITY_MODEL_FILENAME = 'viability_model2.pt'
PURITY_MODEL_FILENAME = 'purity_model.pt'
TETRAZOLIUM_MODEL_FILENAME = 'tz_model.pt' 

# Jika model ada di direktori yang sama dengan script:
COUNTING_MODEL_PATH = COUNTING_MODEL_FILENAME
VIABILITY_MODEL_PATH = VIABILITY_MODEL_FILENAME
PURITY_MODEL_PATH = PURITY_MODEL_FILENAME
TETRAZOLIUM_MODEL_PATH = TETRAZOLIUM_MODEL_FILENAME

# # Jika model ada di subdirektori 'models' (misalnya, untuk PyInstaller)
# # Pastikan Anda membuat direktori 'models' dan menempatkan file .pt di dalamnya
# # atau sesuaikan path ini jika Anda menggunakan resource_path untuk bundling.
# # COUNTING_MODEL_PATH = resource_path(os.path.join('models', COUNTING_MODEL_FILENAME))
# # VIABILITY_MODEL_PATH = resource_path(os.path.join('models', VIABILITY_MODEL_FILENAME))
# # PURITY_MODEL_PATH = resource_path(os.path.join('models', PURITY_MODEL_FILENAME))


UPLOAD_DIR = "streamlit_uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# --- Fungsi untuk Memuat Model (dengan Cache Streamlit) ---
@st.cache_resource
def load_yolo_model(path_ke_model):
    try:
        if not os.path.exists(path_ke_model):
            st.error(f"Model file not found at: {path_ke_model}")
            return None
        model = YOLO(path_ke_model)
        # Minimal success message, or remove if too verbose during use
        # st.sidebar.success(f"Model '{os.path.basename(path_ke_model)}' ready.")
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model from '{path_ke_model}': {e}")
        return None

# --- Helper Functions for GLCM (dari diskusi sebelumnya) ---
GLCM_PROPERTIES = ['Contrast', 'Correlation', 'Energy', 'Homogeneity']
GLCM_ANGLES_DEG_STR = ['0', '45', '90', '135']
GLCM_ANGLES_RAD = [0, np.pi/4, np.pi/2, 3*np.pi/4]

def get_default_glcm_features():
    """Menghasilkan dictionary fitur GLCM default dengan nilai -1.0."""
    features = {}
    for prop in GLCM_PROPERTIES:
        for angle_str in GLCM_ANGLES_DEG_STR:
            features[f"{prop}_{angle_str}"] = -1.0
    return features

def extract_glcm_features(gray_image_region):
    """
    Mengekstrak fitur GLCM untuk setiap sudut yang ditentukan.
    """
    default_features = get_default_glcm_features()

    if gray_image_region is None or gray_image_region.size == 0 or gray_image_region.ndim != 2:
        return default_features

    if gray_image_region.dtype != np.uint8:
        if np.max(gray_image_region) <= 1.0 and np.min(gray_image_region) >=0:
            gray_image_region = (gray_image_region * 255).astype(np.uint8)
        else:
            gray_image_region = np.clip(gray_image_region, 0, 255).astype(np.uint8)

    if np.max(gray_image_region) == np.min(gray_image_region):
        custom_homogeneous_features = default_features.copy()
        for angle_str in GLCM_ANGLES_DEG_STR:
            custom_homogeneous_features[f"Contrast_{angle_str}"] = 0.0
            custom_homogeneous_features[f"Homogeneity_{angle_str}"] = 1.0
            if gray_image_region.size > 0:
                custom_homogeneous_features[f"Energy_{angle_str}"] = 1.0 / (gray_image_region.size) if gray_image_region.size > 0 else 0.0
            else:
                custom_homogeneous_features[f"Energy_{angle_str}"] = 0.0
        return custom_homogeneous_features

    distances = [1]

    try:
        glcm_matrix = graycomatrix(gray_image_region,
                                   distances=distances,
                                   angles=GLCM_ANGLES_RAD,
                                   levels=256,
                                   symmetric=True,
                                   normed=True)
        extracted_features = {}
        for prop_original_name in ['contrast', 'correlation', 'energy', 'homogeneity']:
            prop_values_for_angles = graycoprops(glcm_matrix, prop_original_name)[0]
            prop_display_name = prop_original_name.capitalize()
            for i, angle_str in enumerate(GLCM_ANGLES_DEG_STR):
                feature_key = f"{prop_display_name}_{angle_str}"
                value = prop_values_for_angles[i]
                extracted_features[feature_key] = round(float(value), 4) if not np.isnan(value) else -1.0
        return extracted_features
    except Exception as e:
        return default_features

# --- Side Bar Menu ---
with st.sidebar:
    selected_page = option_menu(
        menu_title="Main Menu",
        options=["Home", "Seed Testing", "Contact"],
        icons=['house', 'search-heart', 'envelope'],
        menu_icon="cast",
        default_index=0,
    )

    st.sidebar.markdown("---") # Pemisah

    # --- NMS TUNING SLIDERS (GLOBAL) ---
    st.sidebar.subheader("NMS Tuning (Global)")

    conf_threshold_slider = st.sidebar.slider(
        "Confidence Threshold (conf)",
        min_value=0.01, max_value=0.99,
        value=0.25, # Nilai default
        step=0.01,
        help="Deteksi dengan skor di bawah ini akan diabaikan. Naikkan untuk mengurangi deteksi (termasuk yang salah/lemah), turunkan untuk lebih banyak deteksi.",
        key="conf_slider_global"
    )
    iou_threshold_slider = st.sidebar.slider(
        "IoU Threshold (iou) for NMS",
        min_value=0.01, max_value=0.99,
        value=0.45, # Nilai default (YOLOv8 default is 0.7, but lower can be better for overlapping objects)
        step=0.01,
        help="Maksimum tumpang tindih (IoU) yang diizinkan antar deteksi. Turunkan untuk lebih agresif menekan kotak yang tumpang tindih.",
        key="iou_slider_global"
    )
    agnostic_nms_checkbox = st.sidebar.checkbox(
        "Agnostic NMS",
        value=False, # Default
        help="Jika dicentang, NMS akan menekan kotak tumpang tindih tanpa mempedulikan kelasnya.",
        key="agnostic_checkbox_global"
    )
    st.sidebar.markdown("---")



# --- Konten Halaman Berdasarkan Pilihan Sidebar ---

if selected_page == "Home":
    col_spacer_home, col_button_home = st.columns([0.85, 0.15])
    with col_button_home:
        if st.button("About", key="about_button_home_tr", help="Pelajari lebih lanjut tentang aplikasi ini"):
            try:
                st.switch_page("pages/About.py")
            except Exception as e:
                st.error(f"Tidak dapat beralih halaman: {e}. Pastikan file 'pages/About.py' ada.")

    col1_home, col2_home = st.columns([1, 1])
    with col1_home:
        # Cek apakah logo.png ada, jika tidak, jangan tampilkan atau tampilkan placeholder
        if os.path.exists("logo.png"):
            st.columns([1, 1])[0].image("logo.png", width=300)
        else:
            st.warning("logo.png tidak ditemukan.")
        st.markdown("<br><br>" * 2 + "<br>" * 8, unsafe_allow_html=True) # Adjusted spacing
        st.header("Next-Generation Seed Insights: Powered by Advanced AI")
        st.markdown("""
        <div style='text-align: justify; line-height: 1.6;'>
        Harness the capabilities of sophisticated deep learning and state-of-the-art
        image processing. Our system offers unparalleled precision in assessing seed
        count, viability, and purity, providing you with the critical technological
        edge for superior crop performance and yield optimization this year.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Explore Seed Testing →", type="primary", key="get_started_home"):
            st.info("Please select a test type from 'Seed Testing' in the sidebar menu, then upload an image.")
            # Consider automatically switching the page or focusing the sidebar if desired,
            # but for now, an info message is fine.
    with col2_home:
        st.markdown(
            """
            <style>
            .right-aligned-image { display: flex; justify-content: flex-end; align-items: center; height: 100%; padding-top: 2rem; }
            </style>
            <div class="right-aligned-image">
            """, unsafe_allow_html=True)
        # Cek apakah Aset1.png ada
        if os.path.exists("Aset1.png"):
            st.image("Aset1.png", width=600)
        else:
            st.warning("Aset1.png tidak ditemukan.")
        st.markdown("</div>", unsafe_allow_html=True)


elif selected_page == "Seed Testing":
    with st.sidebar: # Sub-menu for seed testing options
        selected_test_type = option_menu(
            menu_title="Seed Testing Options",
            options=["Counting", "Viability Test", "Purity Test","Tetrazolium Test"],
            icons=['123', 'heart-pulse', 'filter-circle',  'droplet-half'],
            menu_icon="seedling", # Bootstrap icon for seedling
            default_index=0,
            key="seed_testing_submenu" # Unique key for this menu
        )

    if selected_test_type == "Counting":
        st.title("🌱 Enhanced Seed Counting & Feature Analysis")
        st.markdown("Upload an image to count seeds. Features (RGB, GLCM) are extracted using Otsu Thresholding within each detected bounding box. Adjust NMS parameters in the sidebar for optimal detection.")
        model_counting = load_yolo_model(COUNTING_MODEL_PATH)
        if model_counting is None: st.warning(f"Counting model ('{os.path.basename(COUNTING_MODEL_PATH)}') could not be loaded. Functionality will be limited.")

        uploaded_file_counting = st.file_uploader("Choose an image for seed counting...", type=["png", "jpg", "jpeg"], key="counting_uploader")

        if uploaded_file_counting is not None and model_counting is not None:
            filename_counting = secure_filename(uploaded_file_counting.name)
            temp_filepath_counting = os.path.join(UPLOAD_DIR, f"counting_{filename_counting}")
            with open(temp_filepath_counting, "wb") as f: f.write(uploaded_file_counting.getbuffer())
            st.subheader("Seed Counting & Detailed Analysis:")
            original_image_cv = cv2.imread(temp_filepath_counting)
            if original_image_cv is None:
                st.error(f"Could not read image file for counting: {filename_counting}"); st.stop()

            original_image_for_pixel_hist = original_image_cv.copy()
            original_image_rgb_pil = Image.fromarray(cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2RGB))
            col1_counting_img, col2_counting_img = st.columns(2)
            with col1_counting_img:
                st.markdown("#### Original Image"); st.image(original_image_rgb_pil, use_container_width=True)

            with st.spinner("Performing seed detection and analysis, please wait..."):
                try:
                    results_counting = model_counting(
                        temp_filepath_counting,
                        device='cpu',
                        imgsz=640,
                        conf=conf_threshold_slider,      # Gunakan nilai dari slider
                        iou=iou_threshold_slider,         # Gunakan nilai dari slider
                        agnostic_nms=agnostic_nms_checkbox # Gunakan nilai dari checkbox
                    )
                    output_image_counting_viz = original_image_cv.copy()
                    predictions_data = []
                    seed_count = 0

                    if results_counting and len(results_counting) > 0 and results_counting[0].boxes is not None:
                        seed_count = len(results_counting[0].boxes)
                        yolo_provides_masks = results_counting[0].masks is not None and \
                                              results_counting[0].masks.xy is not None and \
                                              len(results_counting[0].masks.xy) == seed_count

                        for i in range(seed_count): # Loop untuk setiap deteksi benih
                            seed_id = i + 1; box_data = results_counting[0].boxes[i]
                            bbox_coords = box_data.xyxy[0].cpu().numpy().astype(int)
                            x1, y1, x2, y2 = bbox_coords
                            x1_orig, y1_orig, x2_orig, y2_orig = x1, y1, x2, y2 # Simpan koordinat asli
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(original_image_cv.shape[1], x2), min(original_image_cv.shape[0], y2)
                            if x1 >= x2 or y1 >= y2:
                                predictions_data.append({'No.': seed_id, 'Thumbnail': Image.new('RGB', (32,32), color='lightgray'), 'BBox (x1,y1,x2,y2)': f"({x1_orig},{y1_orig},{x2_orig},{y2_orig})-Invalid", 'Seg. Method': 'Error-BBox', 'Mean_R': -1.0, 'Mean_G': -1.0, 'Mean_B': -1.0, **get_default_glcm_features()}); continue
                            bbox_str = f"({x1_orig},{y1_orig},{x2_orig},{y2_orig})"
                            mean_r, mean_g, mean_b = -1.0, -1.0, -1.0
                            glcm_features_dict = get_default_glcm_features(); thumbnail_pil = None
                            roi_color_original = original_image_cv[y1:y2, x1:x2]
                            if roi_color_original.size == 0:
                                predictions_data.append({'No.': seed_id, 'Thumbnail': Image.new('RGB', (32,32), color='lightgray'), 'BBox (x1,y1,x2,y2)': bbox_str, 'Seg. Method': 'Error-ROI', 'Mean_R': mean_r, 'Mean_G': mean_g, 'Mean_B': mean_b, **glcm_features_dict}); continue
                            foreground_rgb_for_extraction = None; foreground_gray_for_glcm = None
                            used_segmentation_method = "Otsu"; otsu_contours_for_viz = None
                            roi_gray_for_otsu = cv2.cvtColor(roi_color_original, cv2.COLOR_BGR2GRAY)
                            if roi_gray_for_otsu.size > 0:
                                _, otsu_mask = cv2.threshold(roi_gray_for_otsu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                if otsu_mask.shape == roi_color_original.shape[:2] and (cv2.countNonZero(otsu_mask) > 0 or cv2.countNonZero(roi_gray_for_otsu) == 0) : # Cek jika mask valid atau ROI memang hitam semua
                                    foreground_rgb_for_extraction = cv2.bitwise_and(roi_color_original, roi_color_original, mask=otsu_mask)
                                    foreground_gray_for_glcm = cv2.bitwise_and(roi_gray_for_otsu, roi_gray_for_otsu, mask=otsu_mask)
                                    contours, _ = cv2.findContours(otsu_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    otsu_contours_for_viz = [cnt + (x1, y1) for cnt in contours] # Offset contour ke posisi global di gambar
                                else:
                                    foreground_rgb_for_extraction = roi_color_original; foreground_gray_for_glcm = cv2.cvtColor(roi_color_original, cv2.COLOR_BGR2GRAY)
                                    used_segmentation_method = "BBox (Otsu Fail)"
                            else:
                                foreground_rgb_for_extraction = roi_color_original; foreground_gray_for_glcm = cv2.cvtColor(roi_color_original, cv2.COLOR_BGR2GRAY)
                                used_segmentation_method = "BBox (ROI Gray Empty)"
                            if foreground_rgb_for_extraction is not None and foreground_rgb_for_extraction.size > 0:
                                thumbnail_pil = Image.fromarray(cv2.cvtColor(foreground_rgb_for_extraction, cv2.COLOR_BGR2RGB))
                            else:
                                thumbnail_pil = Image.new('RGB', (32,32), color='lightgray')
                                if foreground_rgb_for_extraction is None: foreground_rgb_for_extraction = np.zeros((1,1,3), dtype=np.uint8) # fallback
                                if foreground_gray_for_glcm is None: foreground_gray_for_glcm = np.zeros((1,1), dtype=np.uint8) # fallback

                            cv2.rectangle(output_image_counting_viz, (x1_orig, y1_orig), (x2_orig, y2_orig), (0, 0, 255), 2) # BBox Merah
                            # if otsu_contours_for_viz and used_segmentation_method == "Otsu":
                            #     cv2.drawContours(output_image_counting_viz, otsu_contours_for_viz, -1, (255, 0, 0), 1) # Contour Otsu Biru
                            if yolo_provides_masks and i < len(results_counting[0].masks.xy): # Jika model menghasilkan mask
                                mask_polygon_points = results_counting[0].masks.xy[i].astype(np.int32)
                                cv2.polylines(output_image_counting_viz, [mask_polygon_points], isClosed=True, color=(0, 255, 0), thickness=1) # Mask Hijau

                            if foreground_rgb_for_extraction is not None and foreground_rgb_for_extraction.size > 0:
                                active_pixel_mask_for_rgb = None # Mask untuk mean RGB (hanya piksel foreground)
                                if used_segmentation_method == "Otsu": # Hanya hitung mean dari piksel yang terdeteksi Otsu
                                    temp_gray_for_rgb_mask = cv2.cvtColor(foreground_rgb_for_extraction, cv2.COLOR_BGR2GRAY)
                                    _, active_pixel_mask_for_rgb = cv2.threshold(temp_gray_for_rgb_mask, 1, 255, cv2.THRESH_BINARY) # Piksel > 0 dianggap foreground

                                if active_pixel_mask_for_rgb is not None and cv2.countNonZero(active_pixel_mask_for_rgb) > 0 :
                                    mean_bgr_values = cv2.mean(foreground_rgb_for_extraction, mask=active_pixel_mask_for_rgb)
                                elif used_segmentation_method.startswith("BBox"): # Jika Otsu gagal, hitung mean dari seluruh BBox
                                     mean_bgr_values = cv2.mean(foreground_rgb_for_extraction)
                                else: mean_bgr_values = (-1.0, -1.0, -1.0, 0) # Fallback jika tidak ada data
                                mean_b, mean_g, mean_r = mean_bgr_values[0], mean_bgr_values[1], mean_bgr_values[2]

                            if foreground_gray_for_glcm is not None and foreground_gray_for_glcm.size > 0:
                                glcm_features_dict = extract_glcm_features(foreground_gray_for_glcm)
                            predictions_data.append({'No.': seed_id, 'Thumbnail': thumbnail_pil, 'BBox (x1,y1,x2,y2)': bbox_str, 'Seg. Method': used_segmentation_method, 'Mean_R': round(mean_r, 2), 'Mean_G': round(mean_g, 2), 'Mean_B': round(mean_b, 2), **glcm_features_dict})

                    with col2_counting_img:
                        result_image_pil_counting_viz = Image.fromarray(cv2.cvtColor(output_image_counting_viz, cv2.COLOR_BGR2RGB))
                        st.markdown("#### Processed Image (Detections)"); st.image(result_image_pil_counting_viz, use_container_width=True)

                    st.subheader("Overall Counting Result:"); st.metric(label="🧮 Total Seeds Detected", value=seed_count)

                    if predictions_data:
                        df_prediksi = pd.DataFrame(predictions_data)
                        st.subheader("📑 Detailed Seed Analysis Table:")
                        column_config_dynamic = {
                            "No.": st.column_config.NumberColumn("ID", format="%d", width="small"),
                            "Thumbnail": st.column_config.ImageColumn("Seed Img", width="medium"),
                            "BBox (x1,y1,x2,y2)": st.column_config.TextColumn("BBox"),
                            "Seg. Method": st.column_config.TextColumn("Seg.", help="Method: Otsu, or BBox (if Otsu failed/not applicable)"),
                            "Mean_R": st.column_config.NumberColumn("R", format="%.2f"), "Mean_G": st.column_config.NumberColumn("G", format="%.2f"), "Mean_B": st.column_config.NumberColumn("B", format="%.2f")}

                        glcm_cols_ordered = []
                        for prop in GLCM_PROPERTIES:
                            for angle_str in GLCM_ANGLES_DEG_STR:
                                col_name = f"{prop}_{angle_str}"
                                glcm_cols_ordered.append(col_name)
                                if col_name in df_prediksi.columns:
                                    column_config_dynamic[col_name] = st.column_config.NumberColumn(f"{prop[:4]}{angle_str}°", help=f"{prop} at {angle_str}°", format="%.3f" )
                        desired_display_order = ['No.', 'Thumbnail', 'BBox (x1,y1,x2,y2)', 'Seg. Method', 'Mean_R', 'Mean_G', 'Mean_B'] + glcm_cols_ordered
                        display_cols = [col for col in desired_display_order if col in df_prediksi.columns]
                        st.dataframe(df_prediksi[display_cols], column_config=column_config_dynamic, use_container_width=True, hide_index=True, height=min(400, len(df_prediksi)*35 + 70))

                        st.markdown("---")
                        st.subheader("📊 Interactive Feature Dashboard")

                        all_red_pixels_hist_list = []
                        all_green_pixels_hist_list = []
                        all_blue_pixels_hist_list = []

                        if 'BBox (x1,y1,x2,y2)' in df_prediksi.columns and 'Seg. Method' in df_prediksi.columns:
                            for _, row_seed in df_prediksi.iterrows():
                                if "Error" not in row_seed['Seg. Method']:
                                    try:
                                        bbox_str_for_hist = row_seed['BBox (x1,y1,x2,y2)']
                                        coords_for_hist = [int(c.strip()) for c in bbox_str_for_hist.strip('()').split(',')]
                                        x1h, y1h, x2h, y2h = coords_for_hist
                                        x1h, y1h = max(0, x1h), max(0, y1h)
                                        x2h = min(original_image_for_pixel_hist.shape[1], x2h)
                                        y2h = min(original_image_for_pixel_hist.shape[0], y2h)

                                        if x1h < x2h and y1h < y2h:
                                            roi_for_hist = original_image_for_pixel_hist[y1h:y2h, x1h:x2h]
                                            if roi_for_hist.size > 0:
                                                gray_roi_for_hist = cv2.cvtColor(roi_for_hist, cv2.COLOR_BGR2GRAY)
                                                _, otsu_mask_for_hist = cv2.threshold(gray_roi_for_hist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                                                all_blue_pixels_hist_list.extend(roi_for_hist[:,:,0][otsu_mask_for_hist == 255])
                                                all_green_pixels_hist_list.extend(roi_for_hist[:,:,1][otsu_mask_for_hist == 255])
                                                all_red_pixels_hist_list.extend(roi_for_hist[:,:,2][otsu_mask_for_hist == 255])
                                    except Exception:
                                        pass

                        all_red_pixels_np = np.array(all_red_pixels_hist_list)
                        all_green_pixels_np = np.array(all_green_pixels_hist_list)
                        all_blue_pixels_np = np.array(all_blue_pixels_hist_list)

                        df_valid_features = df_prediksi.copy()
                        rgb_glcm_cols_for_dash = ['Mean_R', 'Mean_G', 'Mean_B'] + glcm_cols_ordered
                        for col in rgb_glcm_cols_for_dash:
                            if col in df_valid_features.columns:
                                df_valid_features[col] = df_valid_features[col].replace(-1.0, np.nan) # -1.0 adalah nilai default/error
                        df_valid_features.dropna(subset=['Mean_R', 'Mean_G', 'Mean_B'], inplace=True)
                        df_valid_features = df_valid_features[
                            (df_valid_features['Mean_R'] >= 0) & (df_valid_features['Mean_R'] <= 255) &
                            (df_valid_features['Mean_G'] >= 0) & (df_valid_features['Mean_G'] <= 255) &
                            (df_valid_features['Mean_B'] >= 0) & (df_valid_features['Mean_B'] <= 255)
                        ]

                        if not df_prediksi.empty:
                            st.markdown("#### Row 1: RGB Analysis")
                            col1_hist, col2_boxplot, col3_controls = st.columns([5, 4, 2])

                            with col3_controls:
                                st.markdown("###### Channels")
                                global_show_r = st.checkbox("Red", True, key="global_show_r_dash")
                                global_show_g = st.checkbox("Green", True, key="global_show_g_dash")
                                global_show_b = st.checkbox("Blue", True, key="global_show_b_dash")

                            with col1_hist:
                                st.markdown("###### Pixel Intensity Histogram (All Seeds)")
                                fig_pixel_hist, ax_pixel_hist = plt.subplots(figsize=(6,4))
                                plotted_pixel_hist = False
                                if global_show_r and all_red_pixels_np.size > 0:
                                    ax_pixel_hist.hist(all_red_pixels_np, bins=50, color='red', alpha=0.6, label='Red', density=False)
                                    plotted_pixel_hist = True
                                if global_show_g and all_green_pixels_np.size > 0:
                                    ax_pixel_hist.hist(all_green_pixels_np, bins=50, color='green', alpha=0.6, label='Green', density=False)
                                    plotted_pixel_hist = True
                                if global_show_b and all_blue_pixels_np.size > 0:
                                    ax_pixel_hist.hist(all_blue_pixels_np, bins=50, color='blue', alpha=0.6, label='Blue', density=False)
                                    plotted_pixel_hist = True

                                if plotted_pixel_hist:
                                    ax_pixel_hist.set_title('Pixel Intensity Distribution', fontsize=10)
                                    ax_pixel_hist.set_xlabel('Pixel Intensity', fontsize=8)
                                    ax_pixel_hist.set_ylabel('Pixel Frequency', fontsize=8)
                                    ax_pixel_hist.legend(fontsize=8); ax_pixel_hist.grid(axis='y', linestyle='--', alpha=0.7)
                                    ax_pixel_hist.tick_params(axis='both', which='major', labelsize=7)
                                else:
                                    ax_pixel_hist.text(0.5, 0.5, "Select channel(s) or no pixel data", ha='center', va='center', fontsize=8)
                                st.pyplot(fig_pixel_hist); plt.close(fig_pixel_hist)

                            with col2_boxplot:
                                st.markdown("###### Mean RGB Value Box Plot")
                                boxplot_data_list = []
                                boxplot_labels_list = []
                                boxplot_colors_list = []
                                if not df_valid_features.empty:
                                    if global_show_r and 'Mean_R' in df_valid_features and not df_valid_features['Mean_R'].dropna().empty:
                                        boxplot_data_list.append(df_valid_features['Mean_R'].dropna())
                                        boxplot_labels_list.append('Red'); boxplot_colors_list.append((1,0,0,0.5))
                                    if global_show_g and 'Mean_G' in df_valid_features and not df_valid_features['Mean_G'].dropna().empty:
                                        boxplot_data_list.append(df_valid_features['Mean_G'].dropna())
                                        boxplot_labels_list.append('Green'); boxplot_colors_list.append((0,1,0,0.5))
                                    if global_show_b and 'Mean_B' in df_valid_features and not df_valid_features['Mean_B'].dropna().empty:
                                        boxplot_data_list.append(df_valid_features['Mean_B'].dropna())
                                        boxplot_labels_list.append('Blue'); boxplot_colors_list.append((0,0,1,0.5))

                                if boxplot_data_list:
                                    fig_boxplot_rgb, ax_boxplot_rgb = plt.subplots(figsize=(5,4))
                                    bp = ax_boxplot_rgb.boxplot(boxplot_data_list, patch_artist=True, labels=boxplot_labels_list)
                                    for patch, color_val in zip(bp['boxes'], boxplot_colors_list): patch.set_facecolor(color_val)
                                    ax_boxplot_rgb.set_title('Mean RGB Value Spread', fontsize=10)
                                    ax_boxplot_rgb.set_ylabel('Mean Pixel Intensity', fontsize=8)
                                    ax_boxplot_rgb.tick_params(axis='both', which='major', labelsize=7)
                                    st.pyplot(fig_boxplot_rgb); plt.close(fig_boxplot_rgb)
                                else:
                                    st.info("Select channel(s) or no valid mean RGB data for box plot.")
                            st.markdown("---")

                            if not df_valid_features.empty:
                                st.markdown("#### Row 2: GLCM Feature Averages")
                                valid_glcm_cols_for_mean = [col for col in glcm_cols_ordered if col in df_valid_features.columns]
                                if valid_glcm_cols_for_mean:
                                    df_glcm_means = df_valid_features[valid_glcm_cols_for_mean].mean().dropna()
                                else: df_glcm_means = pd.Series(dtype='float64')
                                if not df_glcm_means.empty:
                                    fig_barchart_glcm, ax_barchart_glcm = plt.subplots(figsize=(10, 5))
                                    df_glcm_means.sort_values(ascending=False).plot(kind='bar', ax=ax_barchart_glcm, color=sns.color_palette("YlGnBu", len(df_glcm_means)))
                                    ax_barchart_glcm.set_title('Average GLCM Feature Values', fontsize=10)
                                    ax_barchart_glcm.set_ylabel('Mean Value', fontsize=8)
                                    plt.setp(ax_barchart_glcm.get_xticklabels(), rotation=45, ha="right", fontsize=7)
                                    ax_barchart_glcm.tick_params(axis='y', labelsize=7)
                                    plt.tight_layout(); st.pyplot(fig_barchart_glcm); plt.close(fig_barchart_glcm)
                                else: st.info("No GLCM data for bar chart.")
                                st.markdown("---")

                                st.markdown("#### Row 3: Feature Relationships & Dimensionality Reduction")
                                col1_row3, col2_row3 = st.columns(2)
                                with col1_row3:
                                    st.markdown("###### Feature Correlation Heatmap")
                                    df_for_heatmap = df_valid_features[['Mean_R', 'Mean_G', 'Mean_B']].copy()
                                    for prop_basename in GLCM_PROPERTIES:
                                        angle_cols_for_prop = [f"{prop_basename}_{angle}" for angle in GLCM_ANGLES_DEG_STR if f"{prop_basename}_{angle}" in df_valid_features.columns]
                                        if angle_cols_for_prop:
                                            df_for_heatmap[f'Avg_{prop_basename}'] = df_valid_features[angle_cols_for_prop].mean(axis=1)
                                    df_for_heatmap.dropna(inplace=True)
                                    if not df_for_heatmap.empty and len(df_for_heatmap) > 1 and len(df_for_heatmap.columns) > 1:
                                        corr_matrix = df_for_heatmap.corr()
                                        fig_heatmap, ax_heatmap = plt.subplots(figsize=(7,5))
                                        sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=.5, ax=ax_heatmap, annot_kws={"size": 7})
                                        ax_heatmap.set_title('RGB & Avg. GLCM Correlation', fontsize=10)
                                        ax_heatmap.tick_params(axis='both', which='major', labelsize=7)
                                        plt.tight_layout(); st.pyplot(fig_heatmap); plt.close(fig_heatmap)
                                    else: st.info("Not enough data for correlation heatmap.")
                                with col2_row3:
                                    st.markdown("###### 2D PCA of Features")
                                    features_for_pca = ['Mean_R', 'Mean_G', 'Mean_B'] + [col for col in glcm_cols_ordered if col in df_valid_features.columns]
                                    pca_data = df_valid_features[features_for_pca].dropna()
                                    if not pca_data.empty and len(pca_data) >= 2 :
                                        scaler = StandardScaler(); scaled_data = scaler.fit_transform(pca_data)
                                        pca = PCA(n_components=2, random_state=42); principal_components = pca.fit_transform(scaled_data)
                                        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
                                        fig_pca, ax_pca = plt.subplots(figsize=(7,5))
                                        sns.scatterplot(x='PC1', y='PC2', data=pca_df, ax=ax_pca, alpha=0.7, s=30)
                                        ax_pca.set_title('2D PCA of Seed Features', fontsize=10)
                                        ax_pca.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=8)
                                        ax_pca.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=8)
                                        ax_pca.grid(True, linestyle='--', alpha=0.6); ax_pca.tick_params(axis='both', which='major', labelsize=7)
                                        st.pyplot(fig_pca); plt.close(fig_pca)
                                    else: st.info("Not enough valid data points for PCA plot (min 2 required).")
                            else:
                                st.info("No valid feature data (after filtering for dashboard) available to generate GLCM, Heatmap, or PCA plots.")
                        else:
                            st.info("No feature data available to generate dashboard.")

                        st.markdown("---")
                        desired_csv_order = ['No.', 'BBox (x1,y1,x2,y2)', 'Seg. Method', 'Mean_R', 'Mean_G', 'Mean_B'] + glcm_cols_ordered
                        csv_cols = [col for col in desired_csv_order if col in df_prediksi.columns]
                        df_for_csv = df_prediksi[csv_cols]
                        csv_data = df_for_csv.to_csv(index=False).encode('utf-8')
                        st.download_button(label="Download Seed Data as CSV", data=csv_data, file_name=f'seed_analysis_counting_{filename_counting.split(".")[0]}.csv', mime='text/csv')

                    elif seed_count > 0 : st.warning("Seeds were detected, but detailed data extraction failed or yielded no valid entries for the table.")
                    else: st.info("No seeds detected in the image.")
                except Exception as e:
                    st.error(f"An error occurred during seed counting and analysis: {e}"); import traceback; st.error(f"Traceback: {traceback.format_exc()}")
        elif uploaded_file_counting is None and model_counting is not None: st.info("Please upload an image to start seed counting and analysis.")
        elif model_counting is None and uploaded_file_counting is not None: st.warning(f"Cannot process image because the counting model ('{os.path.basename(COUNTING_MODEL_PATH)}') is not loaded. Please check model path and file.")


    elif selected_test_type == "Viability Test":
        st.title("🔬 Seed Viability Detector")
        st.markdown("Upload a seed image to detect its viability. Adjust NMS parameters in the sidebar for optimal detection.")
        model_viability = load_yolo_model(VIABILITY_MODEL_PATH)
        if model_viability is None:
            st.warning(f"Viability model ('{VIABILITY_MODEL_FILENAME}') could not be loaded. Viability testing will be limited.")

        uploaded_file_viability = st.file_uploader("Select the seed image (viability)...", type=["png", "jpg", "jpeg"], key="viability_uploader")

        if uploaded_file_viability is not None and model_viability is not None:
            filename_viability = secure_filename(uploaded_file_viability.name)
            temp_filepath_viability = os.path.join(UPLOAD_DIR, f"viability_{filename_viability}")
            with open(temp_filepath_viability, "wb") as f:
                f.write(uploaded_file_viability.getbuffer())

            st.subheader("Viability Image Analysis:")
            col1_viability, col2_viability = st.columns(2)
            with col1_viability:
                st.markdown("#### Original Image")
                try:
                    original_image_pil_viability = Image.open(temp_filepath_viability)
                    st.image(original_image_pil_viability, use_container_width=True)
                except UnidentifiedImageError:
                    st.error("Cannot identify image file. It might be corrupted or not a valid image format.")
                except Exception as e:
                    st.error(f"Failed to display original image: {e}")

            with st.spinner("Analyzing image viability, please wait..."):
                try:
                    results_viability = model_viability(
                        temp_filepath_viability,
                        device='cpu',
                        imgsz=640,
                        conf=conf_threshold_slider,      # Gunakan nilai dari slider
                        iou=iou_threshold_slider,         # Gunakan nilai dari slider
                        agnostic_nms=agnostic_nms_checkbox # Gunakan nilai dari checkbox
                    )

                    img_cv_viability = cv2.imread(temp_filepath_viability)
                    if img_cv_viability is None:
                        st.error(f"Cannot read image file: {filename_viability}")
                    else:
                        output_image_viability = img_cv_viability.copy()
                        viable_count = 0
                        non_viable_count = 0
                        class_names_viability = getattr(model_viability, 'names', {0: 'non-viable', 1: 'viable'})
                        VIABLE_CLASS_NAME_EXPECTED = class_names_viability.get(1, 'viable')
                        NON_VIABLE_CLASS_NAME_EXPECTED = class_names_viability.get(0, 'non-viable')
                        color_viable = (0, 255, 0)
                        color_non_viable = (0, 0, 255)

                        if results_viability and len(results_viability) > 0 and results_viability[0].boxes is not None:
                            for detection_box in results_viability[0].boxes:
                                if detection_box.cls is None or len(detection_box.cls) == 0: continue
                                label_id = int(detection_box.cls[0])
                                class_name = class_names_viability.get(label_id, f"ID_{label_id}")
                                if detection_box.xyxy is None or len(detection_box.xyxy) == 0: continue
                                box = detection_box.xyxy[0].cpu().numpy().astype(int)
                                current_color, text_label = None, ""
                                if class_name == VIABLE_CLASS_NAME_EXPECTED:
                                    viable_count += 1
                                    current_color, text_label = color_viable, "Viable"
                                elif class_name == NON_VIABLE_CLASS_NAME_EXPECTED:
                                    non_viable_count += 1
                                    current_color, text_label = color_non_viable, "Non-Viable"

                                if current_color:
                                    cv2.rectangle(output_image_viability, (box[0], box[1]), (box[2], box[3]), current_color, 2)
                                    cv2.putText(output_image_viability, text_label, (box[0], box[1] - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, current_color, 2)

                        font_scale, thickness = 0.8, 2
                        cv2.putText(output_image_viability, f'{VIABLE_CLASS_NAME_EXPECTED} (Green)', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_viable, thickness, cv2.LINE_AA)
                        cv2.putText(output_image_viability, f'{NON_VIABLE_CLASS_NAME_EXPECTED} (Red)', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_non_viable, thickness, cv2.LINE_AA)
                        result_image_pil_viability = Image.fromarray(cv2.cvtColor(output_image_viability, cv2.COLOR_BGR2RGB))

                        with col2_viability:
                            st.markdown("#### Viability Detection Result Image")
                            st.image(result_image_pil_viability, use_container_width=True)

                        st.subheader("Viability Calculation Results:")
                        count_col1_v, count_col2_v, count_col3_v, count_col4_v = st.columns(4)
                        with count_col1_v: st.metric(label=f"✅ {VIABLE_CLASS_NAME_EXPECTED}", value=viable_count)
                        with count_col2_v: st.metric(label=f"❌ {NON_VIABLE_CLASS_NAME_EXPECTED}", value=non_viable_count)
                        total_seeds_v = viable_count + non_viable_count
                        with count_col3_v: st.metric(label="∑ Total Detected", value=total_seeds_v)
                        with count_col4_v:
                            viability_percentage = (viable_count / total_seeds_v) * 100 if total_seeds_v > 0 else 0
                            st.metric(label="🌿 Viability Percentage", value=f"{viability_percentage:.2f}%")
                except Exception as e:
                    st.error(f"An error occurred while processing the viability image: {e}")
        elif uploaded_file_viability is None and model_viability is not None:
            st.info("Please upload an image file to start viability detection.")
        elif model_viability is None and uploaded_file_viability is not None:
            st.warning("Cannot process image because the viability model is not loaded.")


    elif selected_test_type == "Purity Test":
        st.title("🌿 Seed Purity Analysis")
        st.markdown("Upload seed images for purity analysis and contaminant identification. Adjust NMS parameters in the sidebar for optimal detection.")
        model_purity = load_yolo_model(PURITY_MODEL_PATH)
        if model_purity is None:
            st.warning(f"Purity model ('{PURITY_MODEL_FILENAME}') could not be loaded. Purity testing will be limited.")

        uploaded_file_purity = st.file_uploader("Select the seed image (purity)...", type=["png", "jpg", "jpeg"], key="purity_uploader")

        if uploaded_file_purity is not None and model_purity is not None:
            filename_purity = secure_filename(uploaded_file_purity.name)
            temp_filepath_purity = os.path.join(UPLOAD_DIR, f"purity_{filename_purity}")
            with open(temp_filepath_purity, "wb") as f:
                f.write(uploaded_file_purity.getbuffer())

            st.subheader("Purity Image Analysis:")
            col1_purity, col2_purity = st.columns(2)
            with col1_purity:
                st.markdown("#### Original Image")
                try:
                    original_image_pil_purity = Image.open(temp_filepath_purity)
                    st.image(original_image_pil_purity, use_container_width=True)
                except UnidentifiedImageError:
                    st.error("Cannot identify image file. It might be corrupted or not a valid image format.")
                except Exception as e:
                    st.error(f"Failed to display original image for purity:{e}")

            with st.spinner("Analyzing image purity, please wait..."):
                try:
                    results_purity = model_purity(
                        temp_filepath_purity,
                        device='cpu',
                        imgsz=640,
                        conf=conf_threshold_slider,      # Gunakan nilai dari slider
                        iou=iou_threshold_slider,         # Gunakan nilai dari slider
                        agnostic_nms=agnostic_nms_checkbox # Gunakan nilai dari checkbox
                    )
                    img_cv_purity = cv2.imread(temp_filepath_purity)
                    if img_cv_purity is None:
                        st.error(f"Unable to read image file for purity: {filename_purity}")
                    else:
                        output_image_purity = img_cv_purity.copy()
                        # Urutan kelas yang diharapkan dan warnanya
                        purity_classes_defined = ['Pure-seed', 'BTL', 'Gulma', 'Innert-mater']
                        purity_class_colors_bgr = {
                            'Pure-seed': (255, 0, 0),      # Biru untuk Pure-seed
                            'BTL': (0, 0, 255),            # Merah untuk BTL
                            'Gulma': (0, 255, 0),          # Hijau untuk Gulma
                            'Innert-mater': (0, 165, 255)  # Oranye untuk Innert-mater (BGR)
                        }
                        default_color_bgr = (128, 128, 128) # Abu-abu untuk kelas tak dikenal

                        purity_counts = {name: 0 for name in purity_classes_defined}
                        model_class_names_purity = getattr(model_purity, 'names', {})

                        if not model_class_names_purity: # Fallback jika model.names tidak ada
                             # Asumsikan urutan kelas di model.names (jika ada) sesuai dengan purity_classes_defined
                             # Jika tidak, Anda mungkin perlu mapping manual atau memastikan model.names konsisten
                             st.warning("Model class names not found directly, using predefined order. Ensure model output classes match: 0:Pure-seed, 1:BTL, 2:Gulma, 3:Innert-mater for correct coloring/counting if model.names is missing.")
                             model_class_names_purity = {i: name for i, name in enumerate(purity_classes_defined)}


                        if results_purity and len(results_purity) > 0 and results_purity[0].boxes is not None:
                            has_masks = results_purity[0].masks is not None and results_purity[0].masks.xy is not None and len(results_purity[0].masks.xy) > 0

                            for i_det, detection_box in enumerate(results_purity[0].boxes):
                                if detection_box.cls is None or len(detection_box.cls) == 0: continue
                                cls_id = int(detection_box.cls[0])
                                class_name_from_model = model_class_names_purity.get(cls_id, f"Unknown_ID_{cls_id}")

                                if class_name_from_model in purity_counts:
                                    purity_counts[class_name_from_model] += 1

                                current_color = purity_class_colors_bgr.get(class_name_from_model, default_color_bgr)
                                box_coords = detection_box.xyxy[0].cpu().numpy().astype(int)
                                x1, y1, x2, y2 = box_coords

                                if has_masks and i_det < len(results_purity[0].masks.xy):
                                    try:
                                        polygon_points = results_purity[0].masks.xy[i_det].astype(np.int32)
                                        cv2.polylines(output_image_purity, [polygon_points], isClosed=True, color=current_color, thickness=2)
                                    except Exception:
                                        cv2.rectangle(output_image_purity, (x1, y1), (x2, y2), current_color, 2)
                                else:
                                    cv2.rectangle(output_image_purity, (x1, y1), (x2, y2), current_color, 2)

                                label_text = f"{class_name_from_model}"
                                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                cv2.rectangle(output_image_purity, (x1, y1 - th - 10), (x1 + tw, y1-5), current_color, -1) # Latar belakang label
                                cv2.putText(output_image_purity, label_text, (x1, y1 - 7),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1) # Teks label putih

                        legend_y_start = 25
                        for cls_n in purity_classes_defined: # Tampilkan legenda sesuai urutan yang diinginkan
                            color_v = purity_class_colors_bgr.get(cls_n, default_color_bgr)
                            cv2.putText(output_image_purity, cls_n, (10, legend_y_start),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_v, 2, cv2.LINE_AA)
                            legend_y_start += 25
                        result_image_pil_purity = Image.fromarray(cv2.cvtColor(output_image_purity, cv2.COLOR_BGR2RGB))

                        with col2_purity:
                            st.markdown("#### Image of Purity Analysis Results")
                            st.image(result_image_pil_purity, use_container_width=True)

                        st.subheader("Purity Calculation Results:")
                        # Tampilkan metrik sesuai urutan purity_classes_defined
                        cols_counts_purity = st.columns(len(purity_classes_defined) + 1) # +1 untuk persentase
                        for idx, class_n_iter in enumerate(purity_classes_defined):
                            with cols_counts_purity[idx]:
                                st.metric(label=f"{class_n_iter}", value=purity_counts.get(class_n_iter, 0))

                        total_detected_items = sum(purity_counts.values())
                        pure_seed_count = purity_counts.get('Pure-seed', 0)
                        purity_percentage = (pure_seed_count / total_detected_items) * 100 if total_detected_items > 0 else 0.0

                        with cols_counts_purity[len(purity_classes_defined)]:
                             st.metric(label="🌾 Kemurnian (%)", value=f"{purity_percentage:.2f}%")
                        st.caption(f"Total items detected (relevant for purity): {total_detected_items}")
                except Exception as e:
                    st.error(f"An error occurred while processing the purity image: {e}")
        elif uploaded_file_purity is None and model_purity is not None:
            st.info("Please upload an image file to start purity analysis.")
        elif model_purity is None and uploaded_file_purity is not None:
            st.warning("Cannot process image because the purity model is not loaded.")


# Di dalam blok `elif selected_page == "Seed Testing":`
# Setelah blok `elif selected_test_type == "Purity Test":`

    # ... (kode untuk Counting, Viability, Purity tetap sama) ...

    elif selected_test_type == "Tetrazolium Test":
        st.title("🧪 Tetrazolium Test (TZ) - Viability Classification")
        st.markdown("""
        Unggah gambar benih (atau sampel yang sudah disiapkan) setelah pewarnaan Tetrazolium. 
        Model akan mengklasifikasikan viabilitasnya berdasarkan keseluruhan gambar.
        Pengaturan 'Confidence Threshold' dari sidebar dapat digunakan sebagai ambang batas minimum 
        untuk menerima hasil klasifikasi. Pengaturan 'IoU Threshold' dan 'Agnostic NMS' 
        tidak berlaku untuk tugas klasifikasi gambar ini.
        """)
        
        # --- Load Model Klasifikasi Tetrazolium ---
        # Pastikan TETRAZOLIUM_MODEL_PATH sudah didefinisikan dengan benar
        model_tetrazolium_cls = load_yolo_model(TETRAZOLIUM_MODEL_PATH) 
        
        if model_tetrazolium_cls is None:
            st.error(f"Model klasifikasi Tetrazolium ('{os.path.basename(TETRAZOLIUM_MODEL_PATH)}') tidak dapat dimuat. Pastikan file model ada dan path sudah benar.")
            st.stop() # Hentikan eksekusi halaman ini jika model tidak ada

        uploaded_file_tz_cls = st.file_uploader("Pilih gambar hasil uji Tetrazolium untuk klasifikasi...", type=["png", "jpg", "jpeg"], key="tz_cls_uploader")

        if uploaded_file_tz_cls is not None:
            filename_tz_cls = secure_filename(uploaded_file_tz_cls.name)
            # Buat path file sementara di dalam UPLOAD_DIR
            if not os.path.exists(UPLOAD_DIR):
                os.makedirs(UPLOAD_DIR)
            temp_filepath_tz_cls = os.path.join(UPLOAD_DIR, f"tz_cls_{filename_tz_cls}")

            # Simpan file yang diunggah untuk diproses model
            try:
                with open(temp_filepath_tz_cls, "wb") as f:
                    f.write(uploaded_file_tz_cls.getbuffer())
            except Exception as e:
                st.error(f"Gagal menyimpan file yang diunggah: {e}")
                st.stop()

            st.markdown("---")
            st.subheader("Gambar untuk Diklasifikasi:")
            
            col_img_display_tz, col_result_display_tz = st.columns([2, 3])

            with col_img_display_tz:
                try:
                    image_to_display = Image.open(temp_filepath_tz_cls)
                    st.image(image_to_display, caption="Uploaded Tetrazolium Image", use_container_width=True)
                except UnidentifiedImageError:
                    st.error("Tidak dapat mengidentifikasi file gambar. Mungkin rusak atau bukan format gambar yang valid.")
                    st.stop()
                except Exception as e:
                    st.error(f"Gagal menampilkan gambar TZ: {e}")
                    st.stop()

            # Tombol untuk memicu klasifikasi
            if st.button("Classify TZ Viability", key="classify_tz_button_main_page", type="primary", use_container_width=True):
                with col_result_display_tz:
                    with st.spinner("Mengklasifikasikan gambar..."):
                        try:
                            # Panggil model klasifikasi Ultralytics
                            results_tz_classification_list = model_tetrazolium_cls(
                                temp_filepath_tz_cls,
                                conf=conf_threshold_slider 
                            ) 

                            if results_tz_classification_list:
                                result = results_tz_classification_list[0] 
                                
                                # --- PERBAIKAN AKSES ATRIBUT ---
                                # Akses top1 dan top1conf dari objek result.probs
                                if hasattr(result, 'probs') and result.probs is not None and \
                                   hasattr(result.probs, 'top1') and result.probs.top1 is not None and \
                                   hasattr(result.probs, 'top1conf') and result.probs.top1conf is not None and \
                                   hasattr(result, 'names') and result.names is not None:
                                    
                                    predicted_class_index = result.probs.top1 # Ambil dari result.probs
                                    predicted_confidence = float(result.probs.top1conf) # Ambil dari result.probs
                                    
                                    class_names_from_model = result.names # result.names adalah dictionary
                                    if isinstance(class_names_from_model, dict):
                                        predicted_class_name = class_names_from_model.get(predicted_class_index, f"Unknown Class ID: {predicted_class_index}")
                                    else:
                                        st.warning("Format `result.names` tidak terduga (bukan dictionary). Menggunakan ID kelas.")
                                        predicted_class_name = f"Class ID: {predicted_class_index}"
                                    # --- AKHIR PERBAIKAN AKSES ATRIBUT ---

                                    st.markdown("---")
                                    st.markdown("#### Hasil Klasifikasi:")

                                    if predicted_confidence >= conf_threshold_slider:
                                        if "viable" in predicted_class_name.lower():
                                            st.success(f"**Kelas Prediksi: {predicted_class_name}**")
                                    
                                        elif "nonviable" in predicted_class_name.lower() or "non-viable" in predicted_class_name.lower():
                                            st.error(f"**Kelas Prediksi: {predicted_class_name}**")
                                        else: 
                                            st.info(f"**Kelas Prediksi: {predicted_class_name}** (Periksa nama kelas di model)")
                                        
                                        st.metric(label="Skor Kepercayaan", value=f"{predicted_confidence:.2%}")
                                    else:
                                        st.warning(f"Skor kepercayaan klasifikasi ({predicted_confidence:.2%}) di bawah ambang batas ({conf_threshold_slider:.2%}). Hasil prediksi mentah: '{predicted_class_name}'.")

                                    st.markdown("---")
                                    st.markdown("##### Probabilitas Detail per Kelas:")
                                    
                                    try:
                                        # result.probs.data adalah tensor probabilitas
                                        probs_tensor = result.probs.data 
                                        if hasattr(probs_tensor, 'cpu'): 
                                            probs_data_list = probs_tensor.cpu().numpy().tolist()
                                        else: 
                                            probs_data_list = probs_tensor.tolist()

                                        if isinstance(class_names_from_model, dict):
                                            class_names_for_df = [class_names_from_model.get(i, f"ID_{i}") for i in range(len(probs_data_list))]
                                        else:
                                            class_names_for_df = [f"Class_ID_{i}" for i in range(len(probs_data_list))]
                                            # Peringatan sudah diberikan di atas jika result.names bukan dict

                                        probs_df = pd.DataFrame({
                                            "Nama Kelas": class_names_for_df,
                                            "Probabilitas": [f"{p:.2%}" for p in probs_data_list]
                                        })
                                        st.dataframe(probs_df.sort_values(by="Probabilitas", ascending=False), use_container_width=True, hide_index=True)
                                    except Exception as e_probs:
                                        st.warning(f"Tidak dapat menampilkan probabilitas detail: {e_probs}")
                                        # st.write("Objek probs mentah:", result.probs)

                                else:
                                    st.warning("Format output model klasifikasi tidak sesuai harapan atau atribut penting (probs, names, top1, top1conf) tidak ditemukan/None.")
                                    st.markdown("---")
                                    st.markdown("##### Output Mentah Model (untuk Debugging):")
                                    st.text(f"Tipe objek result: {type(result)}")
                                    st.text(f"Representasi string result:\n{str(result)}")
                                    st.write("Atribut objek result:", dir(result)) 
                                    if hasattr(result, 'probs'):
                                        st.text(f"Tipe objek result.probs: {type(result.probs)}")
                                        st.text(f"Representasi string result.probs:\n{str(result.probs)}")
                                        st.write("Atribut objek result.probs:", dir(result.probs))


                            else:
                                st.error("Model tidak mengembalikan hasil apapun untuk klasifikasi.")
                        
                        except Exception as e:
                            st.error(f"Terjadi error saat klasifikasi gambar Tetrazolium: {e}")
                            import traceback
                            st.error(f"Traceback Lengkap: {traceback.format_exc()}")
            
            # (Opsional) Hapus file sementara setelah selesai
            # try:
            #     if os.path.exists(temp_filepath_tz_cls):
            #         os.remove(temp_filepath_tz_cls)
            # except Exception as e_remove:
            #     st.warning(f"Tidak dapat menghapus file sementara {temp_filepath_tz_cls}: {e_remove}")

        elif uploaded_file_tz_cls is None:
            st.info("Silakan unggah gambar hasil uji Tetrazolium untuk memulai klasifikasi.")


elif selected_page == "Contact":
    st.title("📞 Contact Us")
    # Fungsi untuk konversi gambar ke base64
    def get_base64(img_path):
        try:
            with open(img_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        except FileNotFoundError:
            st.warning(f"Image file not found: {img_path}")
            return None # Return None jika file tidak ditemukan

    # Load foto profil
    fazico_img_path = "kirito.jpg"
    mira_img_path = "Mira-Widiastuti-2.png"

    fazico_base64 = get_base64(fazico_img_path)
    mira_base64 = get_base64(mira_img_path)

    # Layout 2 kolom
    col1_contact, col2_contact = st.columns(2)

# Di dalam blok `elif selected_page == "Contact":`

    # ... (fungsi get_base64 dan pemuatan foto profil tetap sama) ...

    with col1_contact: # Fazico
        if fazico_base64:
            st.markdown(f"""<div style="display: flex; justify-content: center;"><img src="data:image/png;base64,{fazico_base64}" alt="Fazico" style="width: 180px; height: 180px; border-radius: 50%; object-fit: cover; box-shadow: 0px 4px 10px rgba(0,0,0,0.1);"></div>""", unsafe_allow_html=True)
        else:
            st.markdown("<p style='text-align:center;'>Fazico's Photo Not Found</p>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        ## Fazico Rakcel Abryanda
        ##### Software Developer and AI Engineer
        **Agricultural Engineering and Biosystems**
        **Brawijaya University, Class of 2021**

        Experienced in developing AI-driven solutions for agricultural applications.
        Passionate about leveraging technology to enhance agricultural productivity and sustainability.

        ---
        """, unsafe_allow_html=True) # Pisahkan markdown teks dari HTML ikon

        # --- PERBAIKAN UNTUK IKON FAZICO ---
        st.markdown(f"""
        <p>📧 Email: <a href="mailto:fazicochiko@gmail.com">fazicochiko@gmail.com</a></p>
        <p><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="16" height="16" style="vertical-align:middle; margin-right:5px;"> LinkedIn: <a href="https://www.linkedin.com/in/fazico-rakcel-abryanda-130970233" target="_blank">Fazico Rakcel Abryanda</a></p>
        <p><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="16" height="16" style="vertical-align:middle; margin-right:5px;"> GitHub: <a href="https://github.com/fazicoabryanda" target="_blank">fazicoabryanda</a></p>
        """, unsafe_allow_html=True)

    with col2_contact: # Mira
        if mira_base64:
            st.markdown(f"""<div style="display: flex; justify-content: center;"><img src="data:image/png;base64,{mira_base64}" alt="Mira" style="width: 180px; height: 180px; border-radius: 50%; object-fit: cover; box-shadow: 0px 4px 10px rgba(0,0,0,0.1);"></div>""", unsafe_allow_html=True)
        else:
            st.markdown("<p style='text-align:center;'>Mira's Photo Not Found</p>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        ## Mira Landep Widiastuti
        ##### Researcher – Seed Science and Technology
        **National Research and Innovation Agency (BRIN)**

        Dedicated researcher focused on advancing seed science and technology.
        Expertise in seed quality assessment and innovative agricultural practices.

        ---
        """, unsafe_allow_html=True) # Pisahkan markdown teks dari HTML ikon

        # --- PERBAIKAN UNTUK IKON MIRA ---
        # Untuk Google Scholar, kita bisa gunakan emoji atau tag <img> jika ada URL ikon yang bagus.
        # Tag <svg> langsung di markdown bisa tricky.
        # Saya akan menggunakan emoji untuk Google Scholar sebagai contoh, dan <img> untuk yang lain.
        
        # URL ikon Google Scholar yang lebih stabil (jika ada, atau gunakan emoji)
        google_scholar_icon_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/Google_Scholar_logo.svg/1024px-Google_Scholar_logo.svg.png" # Contoh URL ikon
        
        st.markdown(f"""
        <p>📧 Email: (Not Publicly Available)</p>
        <p><img src="{google_scholar_icon_url}" width="16" height="16" style="vertical-align:middle; margin-right:5px;"> Google Scholar: <a href="https://scholar.google.co.id/citations?user=MB4ADTMAAAAJ" target="_blank">Mira Widiastuti</a></p>
        <p><img src="https://www.researchgate.net/images/favicon.png" width="16" height="16" style="vertical-align:middle; margin-right:5px;"> ResearchGate: <a href="https://www.researchgate.net/profile/Mira_Widiastuti" target="_blank">Mira Widiastuti</a></p>
        <p><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="16" height="16" style="vertical-align:middle; margin-right:5px;"> LinkedIn: <a href="https://id.linkedin.com/in/mira-l-widiastuti-52400a42" target="_blank">Mira L. Widiastuti</a></p>
        """, unsafe_allow_html=True)

    # Footer
    st.sidebar.info("Seed Analysis Suite v1")
    st.divider()
    st.caption("Seed Analysis Suite ©2025")
