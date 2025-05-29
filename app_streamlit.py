import streamlit as st
from PIL import Image, UnidentifiedImageError # Added UnidentifiedImageError
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
COUNTING_MODEL_FILENAME = 'counting_model.pt'
VIABILITY_MODEL_FILENAME = 'viability_model.pt'
PURITY_MODEL_FILENAME = 'purity_model.pt'

COUNTING_MODEL_PATH = COUNTING_MODEL_FILENAME
VIABILITY_MODEL_PATH = VIABILITY_MODEL_FILENAME
PURITY_MODEL_PATH = PURITY_MODEL_FILENAME

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


# --- Side Bar Menu ---
with st.sidebar:
    selected_page = option_menu(
        menu_title="Main Menu",
        options=["Home", "Seed Testing", "Contact"],
        icons=['house', 'search-heart', 'envelope'],
        menu_icon="cast",
        default_index=0,
    )

# --- Konten Halaman Berdasarkan Pilihan Sidebar ---

if selected_page == "Home":
    # ... (Home page content remains the same) ...
    st.columns([1, 1])[0].image("logo.png", width=300) # Ganti "logo.png"
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([3, 0.5, 3])
    with col1:
        st.markdown("<br><br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
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
             st.info("Please select a test type from 'Seed Testing' in the sidebar menu.")
    with col3:
        st.markdown(
            """
            <style>
            .right-aligned-image { display: flex; justify-content: flex-end; align-items: center; height: 100%; padding-top: 2rem; }
            </style>
            <div class="right-aligned-image">
            """, unsafe_allow_html=True)
        st.image("Aset1.png", width=600) # Ganti "Aset1.png"
        st.markdown("</div>", unsafe_allow_html=True)


elif selected_page == "Seed Testing":
    with st.sidebar:
        selected_test_type = option_menu(
            menu_title="Seed Testing Options",
            options=["Counting", "Viability Test", "Purity Test"],
            icons=['123', 'heart-pulse', 'filter-circle'],
            menu_icon="seedling",
            default_index=0,
            key="seed_testing_submenu"
        )
    
    if selected_test_type == "Counting":
        st.title("🌱 Enhanced Seed Counting & Analysis")
        st.markdown("Upload an image to count seeds and extract individual seed data (RGB, BBox).")

        model_counting = load_yolo_model(COUNTING_MODEL_PATH)
        if model_counting is None:
            st.warning(f"Counting model ('{COUNTING_MODEL_FILENAME}') could not be loaded. Functionality will be limited.")



        uploaded_file_counting = st.file_uploader("Choose an image for seed counting...", type=["png", "jpg", "jpeg"], key="counting_uploader")

        if uploaded_file_counting is not None and model_counting is not None:
            filename_counting = secure_filename(uploaded_file_counting.name)
            temp_filepath_counting = os.path.join(UPLOAD_DIR, f"counting_{filename_counting}")

            with open(temp_filepath_counting, "wb") as f:
                f.write(uploaded_file_counting.getbuffer())

            st.subheader("Seed Counting & Detailed Analysis:")
            
            # Load the original image once for processing
            original_image_cv = cv2.imread(temp_filepath_counting)
            if original_image_cv is None:
                st.error(f"Could not read image file for counting: {filename_counting}")
                st.stop()
            
            original_image_rgb_pil = Image.fromarray(cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2RGB))

            col1_counting, col2_counting = st.columns(2)
            with col1_counting:
                st.markdown("#### Original Image")
                st.image(original_image_rgb_pil, use_container_width=True)

            with st.spinner("Performing seed detection and analysis, please wait..."):
                try:
                    results_counting = model_counting(temp_filepath_counting, device='cpu', imgsz=640) 
                    
                    output_image_counting_viz = original_image_cv.copy() # For drawing boxes/masks for visualization
                    predictions_data = []
                    seed_count = 0

                    if results_counting and len(results_counting) > 0 and results_counting[0].boxes is not None:
                        seed_count = len(results_counting[0].boxes) # Total detections
                        
                        # Determine if masks are available for segmentation-based cropping
                        has_masks = results_counting[0].masks is not None and results_counting[0].masks.xy is not None and len(results_counting[0].masks.xy) > 0

                        for i in range(seed_count):
                            seed_id = i + 1
                            box_data = results_counting[0].boxes[i]
                            bbox_coords = box_data.xyxy[0].cpu().numpy().astype(int)
                            x1, y1, x2, y2 = bbox_coords
                            bbox_str = f"({x1},{y1},{x2},{y2})"
                            
                            # --- Cropping and RGB Extraction ---
                            mean_r, mean_g, mean_b = 0, 0, 0
                            thumbnail_pil = None

                            if has_masks and i < len(results_counting[0].masks.xy): # Ensure index is valid for masks
                                mask_polygon_points = results_counting[0].masks.xy[i].astype(np.int32)
                                
                                # Create binary mask for the current seed (full image size)
                                single_seed_binary_mask_full = np.zeros(original_image_cv.shape[:2], dtype=np.uint8)
                                cv2.fillPoly(single_seed_binary_mask_full, [mask_polygon_points], 255)
                                
                                # Crop the original image region and the binary mask to the bbox
                                cropped_original_seed_region = original_image_cv[y1:y2, x1:x2]
                                cropped_binary_mask_for_mean = single_seed_binary_mask_full[y1:y2, x1:x2]
                                
                                if cropped_original_seed_region.size > 0 and cropped_binary_mask_for_mean.size > 0 and cv2.countNonZero(cropped_binary_mask_for_mean) > 0:
                                    mean_bgr_values = cv2.mean(cropped_original_seed_region, mask=cropped_binary_mask_for_mean)
                                    mean_b, mean_g, mean_r = mean_bgr_values[0], mean_bgr_values[1], mean_bgr_values[2]

                                    # Create thumbnail from masked region
                                    masked_thumbnail_region = cv2.bitwise_and(cropped_original_seed_region, cropped_original_seed_region, mask=cropped_binary_mask_for_mean)
                                    thumbnail_pil = Image.fromarray(cv2.cvtColor(masked_thumbnail_region, cv2.COLOR_BGR2RGB))
                                else: # Fallback if mask is empty or crop is invalid
                                    mean_r, mean_g, mean_b = -1, -1, -1 # Indicate error or empty mask

                                # Draw mask on visualization image
                                cv2.polylines(output_image_counting_viz, [mask_polygon_points], isClosed=True, color=(0, 255, 0), thickness=2)
                            
                            else: # Fallback to bounding box if no masks or mask error
                                cropped_seed_bbox = original_image_cv[y1:y2, x1:x2]
                                if cropped_seed_bbox.size > 0:
                                    mean_bgr_values = cv2.mean(cropped_seed_bbox)
                                    mean_b, mean_g, mean_r = mean_bgr_values[0], mean_bgr_values[1], mean_bgr_values[2]
                                    thumbnail_pil = Image.fromarray(cv2.cvtColor(cropped_seed_bbox, cv2.COLOR_BGR2RGB))
                                else:
                                    mean_r, mean_g, mean_b = -1,-1,-1
                                
                                # Draw bounding box on visualization image
                                cv2.rectangle(output_image_counting_viz, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red Bbox as fallback

                            # This is the line that might be incorrect in your current code
                            # Ensure it looks like this:
                            predictions_data.append({
                                'No.': seed_id,
                                'Thumbnail': thumbnail_pil if thumbnail_pil else Image.new('RGB', (32,32), color='gray'), # CORRECT
                                'BBox (x1,y1,x2,y2)': bbox_str,
                                'Mean R': round(mean_r, 2),
                                'Mean G': round(mean_g, 2),
                                'Mean B': round(mean_b, 2)
                            })
                    
                    result_image_pil_counting_viz = Image.fromarray(cv2.cvtColor(output_image_counting_viz, cv2.COLOR_BGR2RGB))
                    with col2_counting:
                        st.markdown("#### Processed Image (Detections)")
                        st.image(result_image_pil_counting_viz, use_container_width=True)

                    # --- Display Metrics and Table ---
                    st.subheader("Overall Counting Result:")
                    st.metric(label="🧮 Total Seeds Detected", value=seed_count)

                    if predictions_data:
                        df_prediksi = pd.DataFrame(predictions_data)
                        st.subheader("Detailed Seed Analysis Table:")
                        st.dataframe(df_prediksi,
                                     column_config={
                                         "No.": st.column_config.NumberColumn("ID", format="%d", width="small"),
                                         "Thumbnail": st.column_config.ImageColumn("Seed Image", width="medium"),
                                         "BBox (x1,y1,x2,y2)": st.column_config.TextColumn("Bounding Box"),
                                         "Mean R": st.column_config.NumberColumn("Avg. Red", format="%.2f"),
                                         "Mean G": st.column_config.NumberColumn("Avg. Green", format="%.2f"),
                                         "Mean B": st.column_config.NumberColumn("Avg. Blue", format="%.2f"),
                                     }, use_container_width=True, height=400)

                        # Prepare CSV for download (dropping Thumbnail PIL objects)
                        df_for_csv = df_prediksi.drop(columns=['Thumbnail'])
                        csv_data = df_for_csv.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Seed Data as CSV",
                            data=csv_data,
                            file_name='seed_analysis_rgb.csv',
                            mime='text/csv',
                        )
                    elif seed_count > 0 : # Detections happened but data extraction failed for all
                        st.warning("Seeds were detected, but detailed data extraction failed or yielded no valid entries for the table.")
                    else:
                        st.info("No seeds detected in the image.")
                        
                except Exception as e:
                    st.error(f"An error occurred during seed counting and analysis: {e}")
                    import traceback
                    st.error(f"Traceback: {traceback.format_exc()}")

        elif uploaded_file_counting is None and model_counting is not None:
            st.info("Please upload an image to start seed counting and analysis.")
        elif model_counting is None and uploaded_file_counting is not None:
            st.warning("Cannot process image because the counting model is not loaded.")


    elif selected_test_type == "Viability Test":
        # ... (Viability Test content remains the same as your last version) ...
        st.title("🔬 Seed Viability Detector")
        st.markdown("Upload a seed image to detect its viability.")

        # --- Load Viability Model ---
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
                except Exception as e:
                    st.error(f"Failed to display original image: {e}")

            with st.spinner("Analyzing image viability, please wait..."):
                try:
                    results_viability = model_viability(temp_filepath_viability, device='cpu', imgsz=640)

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
        # ... (Purity Test content remains the same as your last version) ...
        st.title("🌿 Seed Purity Analysis")
        st.markdown("Upload seed images for purity analysis and contaminant identification..")
        
        model_purity = load_yolo_model(PURITY_MODEL_PATH)
        if model_purity is None:
            st.warning(f"Purity model ('{PURITY_MODEL_FILENAME}') could not be loaded. Purity testing will be limited.")
        else:
            purity_classes_defined = ['BTL', 'Gulma', 'Innert-mater', 'Pure-seed'] 
            model_actual_classes = getattr(model_purity, 'names', {}).values()


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
                except Exception as e:
                    st.error(f"Failed to display original image for purity:{e}")

            with st.spinner("Analyzing image purity, please wait..."):
                try:
                    results_purity = model_purity(temp_filepath_purity, device='cpu', imgsz=640) 

                    img_cv_purity = cv2.imread(temp_filepath_purity)
                    if img_cv_purity is None:
                        st.error(f"Unable to read image file for purity: {filename_purity}")
                    else:
                        output_image_purity = img_cv_purity.copy()
                        purity_classes_defined = ['BTL', 'Gulma', 'Innert-mater', 'Pure-seed']
                        purity_counts = {name: 0 for name in purity_classes_defined}
                        
                        purity_class_colors_bgr = {
                            'BTL': (0, 0, 255), 'Gulma': (0, 255, 0),
                            'Innert-mater': (255, 165, 0), 'Pure-seed': (255, 0, 0) }
                        default_color_bgr = (128, 128, 128)

                        model_class_names_purity = getattr(model_purity, 'names', {})
                        if not model_class_names_purity:
                             model_class_names_purity = {i: name for i, name in enumerate(purity_classes_defined)}


                        if results_purity and len(results_purity) > 0 and results_purity[0].boxes is not None:
                            has_masks = results_purity[0].masks is not None and results_purity[0].masks.xy 
                            
                            for i in range(len(results_purity[0].boxes)):
                                box_data = results_purity[0].boxes[i]
                                if box_data.cls is None or len(box_data.cls) == 0: continue
                                
                                cls_id = int(box_data.cls[0])
                                class_name_from_model = model_class_names_purity.get(cls_id, f"ID_{cls_id}")

                                if class_name_from_model in purity_counts:
                                    purity_counts[class_name_from_model] += 1
                                
                                current_color = purity_class_colors_bgr.get(class_name_from_model, default_color_bgr)
                                box_coords = box_data.xyxy[0].cpu().numpy().astype(int)
                                x1, y1, x2, y2 = box_coords

                                if has_masks and i < len(results_purity[0].masks.xy):
                                    try:
                                        polygon_points = results_purity[0].masks.xy[i].astype(np.int32)
                                        cv2.polylines(output_image_purity, [polygon_points], isClosed=True, color=current_color, thickness=2)
                                    except Exception: 
                                        cv2.rectangle(output_image_purity, (x1, y1), (x2, y2), current_color, 2)
                                else: 
                                    cv2.rectangle(output_image_purity, (x1, y1), (x2, y2), current_color, 2)
                                
                                label_text = f"{class_name_from_model}"
                                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                cv2.rectangle(output_image_purity, (x1, y1 - th - 10), (x1 + tw, y1-5), current_color, -1)
                                cv2.putText(output_image_purity, label_text, (x1, y1 - 7), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                        legend_y_start = 25
                        for cls_n, color_v in purity_class_colors_bgr.items():
                            cv2.putText(output_image_purity, cls_n, (10, legend_y_start), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_v, 2, cv2.LINE_AA)
                            legend_y_start += 25

                        result_image_pil_purity = Image.fromarray(cv2.cvtColor(output_image_purity, cv2.COLOR_BGR2RGB))

                        with col2_purity:
                            st.markdown("#### Image of Purity Analysis Results")
                            st.image(result_image_pil_purity, use_container_width=True)

                        st.subheader("Purity Calculation Results:")
                        cols_counts_purity = st.columns(len(purity_classes_defined) + 1) 
                        for idx, class_n_iter in enumerate(purity_classes_defined):
                            with cols_counts_purity[idx]:
                                st.metric(label=f"{class_n_iter}", value=purity_counts[class_n_iter])
                        
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


elif selected_page == "Contact":
        # Fungsi untuk konversi gambar ke base64
        def get_base64(img):
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()

        # Load foto profil
        fazico_img = Image.open("kirito.jpg")  # Ganti sesuai filemu
        mira_img = Image.open("Mira-Widiastuti-2.png")      # Ganti sesuai file Mira

        fazico_base64 = get_base64(fazico_img)
        mira_base64 = get_base64(mira_img)

        # Layout 2 kolom
        col1, col2 = st.columns(2)

        # -------------------- FAZICO --------------------
        with col1:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: center;">
                    <img src="data:image/png;base64,{fazico_base64}"
                        alt="Fazico"
                        style="width: 180px; height: 180px; border-radius: 50%; object-fit: cover; box-shadow: 0px 4px 10px rgba(0,0,0,0.1);">
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            ### Fazico Rakcel Abryanda
            🎓 Undergraduate Student  
            **Agricultural and Biosystems Engineering**  
            **Brawijaya University, Class of 2021**

            Passionate about artificial intelligence, agriculture, technology, data science, and computer vision. Let's connect!

            **📧 Email:** [fazicochiko@gmail.com](mailto:fazicochiko@gmail.com)  
            **📱 WhatsApp:** [0895-2604-3044](https://wa.me/6289526043044)  
            **💼 LinkedIn:** [Fazico Rakcel Abryanda](https://www.linkedin.com/in/fazico-rakcel-abryanda-130970233)  
            **💻 GitHub:** [fazicoabryanda](https://github.com/fazicoabryanda)
            """)

        # -------------------- MIRA --------------------
        with col2:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: center;">
                    <img src="data:image/png;base64,{mira_base64}"
                        alt="Mira"
                        style="width: 180px; height: 180px; border-radius: 50%; object-fit: cover; box-shadow: 0px 4px 10px rgba(0,0,0,0.1);">
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            ### Mira Landep Widiastuti  
            🧪 Researcher – Seed Science and Technology  
            **National Research and Innovation Agency (BRIN)**

            Mira specializes in rice seed physiology, seed processing technology, and digital image analysis for seed classification. She has contributed to many scientific papers in seed quality improvement.

            **📧 Email:** N/A  
            **📚 Google Scholar:** [Mira Widiastuti](https://scholar.google.co.id/citations?user=MB4ADTMAAAAJ)  
            **📖 ResearchGate:** [Mira Widiastuti](https://www.researchgate.net/profile/Mira_Widiastuti)  
            **💼 LinkedIn:** [Mira L. Widiastuti](https://id.linkedin.com/in/mira-l-widiastuti-52400a42)
            """)

        # Footer
        st.divider()
        st.caption("©2025")


st.sidebar.markdown("---")
st.sidebar.info("Seed Analysis Suite v1")