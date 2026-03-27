import streamlit as st
import requests
import base64
from PIL import Image
from sentence_transformers import SentenceTransformer

# Page Styling
st.set_page_config(page_title="Aperion FlashCost Agent", layout="wide")
st.title("🧥 FlashCost: Design to Manufacturing")
st.markdown("---")

# Load Embedder (Cached for speed)
#@st.cache_resource
#def load_model():
#    return SentenceTransformer('clip-ViT-B-32')

#model = load_model()



# Service URLs
CNN_URL = "http://localhost:8000/predict"
VECTOR_URL = "http://localhost:8001/search"

# UI Layout
uploaded_file = st.file_uploader("Upload Sketch", type=["png", "jpg", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Target Design")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        
        # Prep for CNN
        b64_img = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
        clean_b64 = b64_img.replace('+', '-').replace('/', '_').rstrip('=')

    with col2:
        st.subheader("Aperion Intelligence")
        
        # 1. Cost Prediction
        with st.spinner("Analyzing Cost..."):
            resp = requests.post(CNN_URL, json={"instances": [{"image_bytes": clean_b64}]})
            if resp.status_code == 200:
                cost = resp.json()['predictions'][0][0]
                st.metric("Estimated Cost", f"${cost:.2f}")
            else:
                st.error("CNN Service Error")

        # 2. Historical Search
        # In the search logic, change the embedding to:
        with st.spinner("Finding Historical Evidence..."):
            emb = [0.1] * 1280  # Dummy vector
            vec_resp = requests.post(VECTOR_URL, json={"embedding": emb, "top_k": 3})

            if vec_resp.status_code == 200:
                st.write("### Similar Historical References")
                for match in vec_resp.json().get('matches', []):
                    st.success(f"ID: {match['id']} | Distance: {match['distance']:.4f}")
