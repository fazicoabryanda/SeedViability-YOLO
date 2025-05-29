import streamlit as st
from PIL import Image # Untuk bekerja dengan gambar di Streamlit
import cv2 # OpenCV tetap digunakan untuk pemrosesan gambar
from ultralytics import YOLO # Model YOLO Anda
import os
import sys
from werkzeug.utils import secure_filename # Bisa tetap digunakan untuk sanitasi nama file

# --- (OPSIONAL TAPI DIREKOMENDASIKAN) Fungsi Helper Path untuk Bundling Nanti ---
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, relative_path)

# --- Konfigurasi Aplikasi ---
st.set_page_config(layout="wide", page_title="Seed Viability Detector", page_icon="🌱")

MODEL_FILENAME = 'bobotviabilitas1.pt'
MODEL_PATH = MODEL_FILENAME # Asumsi model di direktori yang sama untuk dev Streamlit
# MODEL_PATH = resource_path(MODEL_FILENAME) # Untuk persiapan bundling

UPLOAD_DIR = "streamlit_uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# --- Fungsi untuk Memuat Model (dengan Cache Streamlit) ---
@st.cache_resource
def load_yolo_model(path_ke_model):
    try:
        model = YOLO(path_ke_model)
        # st.success(f"Model '{os.path.basename(path_ke_model)}' berhasil dimuat.") # Bisa di-uncomment jika ingin notifikasi
        return model
    except Exception as e:
        st.error(f"Error saat memuat model YOLO: {e}")
        return None

model_yolo = load_yolo_model(MODEL_PATH)

# --- Antarmuka Pengguna (UI) Streamlit ---
st.title("🌱 Seed Viability Detector")
st.markdown("Unggah gambar benih untuk mendeteksi viabilitasnya.")

uploaded_file = st.file_uploader("Pilih gambar benih...", type=["png", "jpg", "jpeg"])

if model_yolo is None:
    st.error("Model YOLO tidak dapat dimuat. Pastikan file model '{}' ada di direktori yang benar.".format(MODEL_FILENAME))
    st.stop() # Hentikan eksekusi jika model tidak ada

if uploaded_file is not None:
    filename = secure_filename(uploaded_file.name)
    temp_filepath = os.path.join(UPLOAD_DIR, filename)

    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.subheader("Analisis Gambar:")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Gambar Asli")
        try:
            original_image_pil = Image.open(temp_filepath)
            # Menggunakan use_container_width yang baru
            st.image(original_image_pil, use_container_width=True) 
        except Exception as e:
            st.error(f"Gagal menampilkan gambar asli: {e}")

    with st.spinner("Menganalisis gambar, mohon tunggu... Ini mungkin memakan waktu beberapa saat."):
        try:
            results = model_yolo(temp_filepath, device='cpu', imgsz=640)
            
            img_cv = cv2.imread(temp_filepath)
            if img_cv is None:
                st.error(f"Tidak dapat membaca file gambar yang diunggah dengan OpenCV: {filename}")
            else:
                viable_count = 0
                non_viable_count = 0
                color_viable = (0, 255, 0)
                color_non_viable = (0, 0, 255)

                if results and len(results) > 0 and results[0].boxes is not None:
                    for detection_box in results[0].boxes:
                        if detection_box.cls is None or len(detection_box.cls) == 0:
                            continue
                        label = int(detection_box.cls[0])
                        
                        if detection_box.xyxy is None or len(detection_box.xyxy) == 0:
                            continue
                        box = detection_box.xyxy[0].cpu().numpy().astype(int)
                        
                        current_color = None
                        if label == 1: 
                            viable_count += 1
                            current_color = color_viable
                        elif label == 0: 
                            non_viable_count += 1
                            current_color = color_non_viable
                        
                        if current_color:
                            cv2.rectangle(img_cv, (box[0], box[1]), (box[2], box[3]), current_color, 2)
                
                font_scale = 0.8
                thickness = 2
                cv2.putText(img_cv, 'Viable (Hijau)', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_viable, thickness, cv2.LINE_AA)
                cv2.putText(img_cv, 'Non-Viable (Merah)', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_non_viable, thickness, cv2.LINE_AA)

                result_image_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

                with col2:
                    st.markdown("#### Gambar Hasil Deteksi")
                    # Menggunakan use_container_width yang baru
                    st.image(result_image_pil, use_container_width=True) 

                st.subheader("Hasil Perhitungan:")
                count_col1, count_col2, count_col3 = st.columns(3)
                with count_col1:
                    st.metric(label="✅ Benih Viable", value=viable_count)
                with count_col2:
                    st.metric(label="❌ Benih Non-Viable", value=non_viable_count)
                with count_col3:
                    total_seeds = viable_count + non_viable_count
                    st.metric(label="∑ Total Benih Terdeteksi", value=total_seeds)
                
                # (Opsional) Hapus file sementara setelah selesai untuk menghemat ruang
                # try:
                #     os.remove(temp_filepath)
                # except Exception as e:
                #     st.warning(f"Tidak dapat menghapus file sementara {temp_filepath}: {e}")

        except Exception as e:
            st.error(f"Terjadi error saat pemrosesan gambar: {e}")
            # import traceback # Uncomment untuk debugging lebih detail
            # st.error(f"Traceback Lengkap: {traceback.format_exc()}")

elif uploaded_file is None:
    st.info("Silakan unggah file gambar untuk memulai deteksi.")

st.markdown("---")
st.markdown("Dibuat dengan Streamlit & YOLO.")