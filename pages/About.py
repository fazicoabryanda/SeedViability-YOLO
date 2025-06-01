# pages/About.py
import streamlit as st
from PIL import Image
import base64
from io import BytesIO
import os

# Set page config - should be first Streamlit command
st.set_page_config(
    page_title="About Seed Analysis Suite",
    page_icon="icon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS that complements your config.toml theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    /* Base styles that work with your theme */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    .header-text {
        font-size: 2.5rem !important;
        font-weight: 600 !important;
        color: var(--primary-color) !important;
        margin-bottom: 0.5rem !important;
    }
    
    .subheader-text {
        font-size: 1.1rem !important;
        font-weight: 400 !important;
        color: var(--text-color) !important;
        opacity: 0.8;
        margin-bottom: 1.5rem !important;
    }
    
    /* Cards that complement your dark green theme */
    .feature-card {
        background: rgba(0, 152, 119, 0.1);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid var(--primary-color);
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .tech-card {
        background: rgba(0, 152, 119, 0.1);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--primary-color), transparent);
        margin: 2rem 0;
        border: none;
    }
    
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
    }
    
    .contact-card {
        background: rgba(0, 152, 119, 0.1);
        border-radius: 10px;
        padding: 1.5rem;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .version-badge {
        background-color: var(--primary-color);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    /* Links styling */
    a {
        color: var(--primary-color) !important;
        text-decoration: none !important;
        font-weight: 600 !important;
    }
    
    a:hover {
        text-decoration: underline !important;
    }
    
    /* List styling */
    ul, ol {
        padding-left: 1.5rem !important;
    }
    
    li {
        margin-bottom: 0.5rem !important;
        color: var(--text-color) !important;
    }
</style>
""", unsafe_allow_html=True)

def get_image_base64(image_path: str, desired_width: int = None):
    """Convert image file to base64 string for HTML embedding."""
    try:
        img = Image.open(image_path)
        
        if desired_width:
            if img.width == 0:
                return None, f"Image width is 0 for '{image_path}'."
            aspect_ratio = img.height / img.width
            new_height = int(aspect_ratio * desired_width)
            img = img.resize((desired_width, new_height))

        output_buffer = BytesIO()
        image_format = Image.MIME.get(img.format) if img.format else 'PNG'
        if img.format == "JPEG":
            image_format = "JPEG"
        elif img.format == "PNG":
            image_format = "PNG"
            
        img.save(output_buffer, format=image_format)
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode()
        
        mime_type = f"image/{image_format.lower()}"
        if image_format.upper() == "ICO":
            mime_type = "image/x-icon"

        return f"data:{mime_type};base64,{base64_str}", None
    except FileNotFoundError:
        return None, f"Image file not found at: {os.path.abspath(image_path)}"
    except Exception as e:
        return None, f"Error processing image '{image_path}': {e}"

# Header Section with Logo
col1, col2 = st.columns([1, 2])

with col1:
    image_file_name = "logo.png"
    image_display_width = 300
    
    logo_base64, error_message = get_image_base64(image_file_name, desired_width=image_display_width)
    
    if error_message:
        st.error(error_message)
    elif logo_base64:
        st.markdown(
            f"""
            <div class="logo-container">
                <img src="{logo_base64}" alt="App Logo" style="max-width: 100%; height: auto; filter: drop-shadow(0 0 8px rgba(0, 152, 119, 0.3));">
            </div>
            """,
            unsafe_allow_html=True,
        )

with col2:
    st.markdown('<p class="header-text">About Seed Analysis Suite</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader-text">Harnessing AI for precision agriculture</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the **Seed Analysis Suite** - a cutting-edge web application designed to revolutionize seed quality assessment 
    through advanced computer vision and machine learning technologies.
    """)

# Features Section
st.markdown("""
<div class="divider"></div>
""", unsafe_allow_html=True)

st.markdown('<p class="header-text">Key Features</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader-text">Powerful tools for seed analysis</p>', unsafe_allow_html=True)

features = [
    {
        "icon": "🔢",
        "title": "Seed Counting",
        "desc": "Automatically count seeds in an image and extract crucial features like average color (RGB) and texture (GLCM) from each detected seed."
    },
    {
        "icon": "🌿",
        "title": "Viability Test",
        "desc": "Predict seed viability (ability to germinate) based on image analysis, classifying seeds as viable or non-viable."
    },
    {
        "icon": "🧪",
        "title": "Purity Test",
        "desc": "Identify and classify components within a seed sample, such as pure seeds, other crop seeds, weed seeds, and inert matter."
    }
]

for feature in features:
    with st.container():
        st.markdown(f"""
        <div class="feature-card">
            <h3>{feature['icon']} {feature['title']}</h3>
            <p>{feature['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

# Technology Stack Section
st.markdown("""
<div class="divider"></div>
""", unsafe_allow_html=True)

st.markdown('<p class="header-text">Technology Stack</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader-text">Built with modern tools</p>', unsafe_allow_html=True)

tech_col1, tech_col2 = st.columns(2)

with tech_col1:
    st.markdown("""
    <div class="tech-card">
        <h4>Core Technologies</h4>
        <ul>
            <li>Python</li>
            <li>Streamlit</li>
            <li>YOLO (You Only Look Once)</li>
            <li>OpenCV</li>
            <li>Pillow (PIL)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with tech_col2:
    st.markdown("""
    <div class="tech-card">
        <h4>Supporting Libraries</h4>
        <ul>
            <li>NumPy & Pandas</li>
            <li>Scikit-image</li>
            <li>Matplotlib & Seaborn</li>
            <li>Scikit-learn</li>
            <li>TensorFlow/PyTorch</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# How to Use Section - Fixed Container Issue
st.markdown("""
<div class="divider"></div>
""", unsafe_allow_html=True)

# Correct "How to Use" section in your Streamlit code
with st.container():
    st.markdown('<p class="header-text">How to Use</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader-text">Get started in minutes</p>', unsafe_allow_html=True)
    
    steps = [
        "Select the desired analysis type from the Main Menu in the left sidebar",
        "Choose the specific test sub-menu (Counting, Viability Test, or Purity Test)",
        "Upload your seed image using the file uploader button",
        "View the processed results including visualizations and data tables",
        "Download seed feature data in CSV format for further analysis"
    ]
    
    # Build the HTML content properly
    steps_html = """
    <div class="tech-card">
        <ol>
    """
    
    for step in steps:
        steps_html += f"<li>{step}</li>"
    
    steps_html += """
        </ol>
    </div>
    """
    
    st.markdown(steps_html, unsafe_allow_html=True)

# Developers & Contact Section
st.markdown("""
<div class="divider"></div>
""", unsafe_allow_html=True)

st.markdown('<p class="header-text">Development Team</p>', unsafe_allow_html=True)

with st.container():
    st.markdown("""
    <div class="contact-card">
        <div style="display: flex; flex-direction: column; gap: 1rem;">
            <div>
                <h4>Fazico Rakcel Abryanda</h4>
                <h6>Software Developer & AI Engineer</h6>
                <p>Department of Agricultural Engineering and Biosystems, Brawijaya University</p>
                <p>Email: <a href="mailto:fazicochiko@gmail.com">fazicochiko@gmail.com</a></p>
            </div>
            <div>
                <h4>Mira Landep Widiastuti</h4>
                <h6>Researcher</h6>
                <p>National Research and Innovation Agency (BRIN), Indonesia</p>
                <p>Google Scholar: <a href="https://scholar.google.co.id/citations?user=MB4ADTMAAAAJ" target="_blank">Mira Widiastuti</a></p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Version & Footer
st.markdown("""
<div class="divider"></div>
""", unsafe_allow_html=True)

APP_VERSION = "1.0.0"
LAST_UPDATED = "May 31, 2025"

col1, col2, col3 = st.columns([1,1,1])
with col2:
    st.markdown(f"""
    <div style="text-align: center;">
        <div class="version-badge">Version {APP_VERSION}</div>
        <p style="margin-top: 0.5rem; color: var(--text-color); opacity: 0.8; font-size: 0.8rem;">Last updated: {LAST_UPDATED}</p>
    </div>
    """, unsafe_allow_html=True)
