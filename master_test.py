import requests
import base64
from sentence_transformers import SentenceTransformer
from PIL import Image

# Configuration
IMAGE_PATH = "test_00015.png"
CNN_URL = "http://localhost:8000/predict"
VECTOR_URL = "http://localhost:8001/search"

# Load a local model that outputs 1280-dim or compatible vectors
# 'clip-ViT-B-32' is a standard choice for apparel/vision
#model = SentenceTransformer('clip-ViT-B-32')

# Comment out the CLIP model loading
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('clip-ViT-B-32')

# In the run_nova_pipeline function, change step 3 to:
print("🧠 Using dummy vector for hand-off (bypass download)...")
real_embedding = [0.1] * 1280

def run_nova_pipeline(image_path):
    print(f"--- Processing: {image_path} ---")

    # 1. Image Preparation for CNN
    with open(image_path, "rb") as f:
        b64_img = base64.b64encode(f.read()).decode("utf-8")
        clean_b64 = b64_img.replace('+', '-').replace('/', '_').rstrip('=')

    # 2. Get Cost from CNN
    cnn_resp = requests.post(CNN_URL, json={"instances": [{"image_bytes": clean_b64}]})
    print(f"💰 Predicted Manufacturing Cost: ${cnn_resp.json()['predictions'][0][0]:.2f}")

    # 3. DUMMY Vectorization (Bypassing CLIP for Hand-off)
    print("🧠 Using dummy vector for hand-off (bypass download)...")
    real_embedding = [0.1] * 1280 

    # 4. Search with Dummy Data
    vec_resp = requests.post(VECTOR_URL, json={"embedding": real_embedding, "top_k": 3})
    matches = vec_resp.json().get('matches', [])

    print(f"\n🔍 Top {len(matches)} Historical Matches (Evidence):")
    for match in matches:
        print(f"   - Item ID: {match['id']} (Distance: {match['distance']:.4f})")


if __name__ == "__main__":
    run_nova_pipeline(IMAGE_PATH)
