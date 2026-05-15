import streamlit as st
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from deepface import DeepFace
from PIL import Image
import tempfile
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="The Baybee Finder",
    page_icon="💛",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0a;
    color: #f0ece4;
}

.main { background-color: #0a0a0a; }

.title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(135deg, #ffc94d, #ff85c2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.subtitle {
    text-align: center;
    color: rgba(240,236,228,0.45);
    font-size: 14px;
    margin-bottom: 2rem;
    letter-spacing: 1px;
}

.result-found {
    background: rgba(0,255,136,0.08);
    border: 1px solid rgba(0,255,136,0.3);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin: 1rem 0;
}

.result-not-found {
    background: rgba(255,68,68,0.08);
    border: 1px solid rgba(255,68,68,0.3);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin: 1rem 0;
}

.confidence-box {
    background: rgba(255,201,77,0.08);
    border: 1px solid rgba(255,201,77,0.2);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    margin: 0.5rem 0;
}

.stButton > button {
    background: linear-gradient(135deg, #ffc94d, #ff85c2);
    color: #0a0a0a;
    border: none;
    border-radius: 12px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 15px;
    padding: 0.6rem 2rem;
    width: 100%;
}

.stFileUploader {
    border: 1.5px dashed rgba(255,201,77,0.35);
    border-radius: 16px;
    padding: 1rem;
    background: rgba(255,201,77,0.03);
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_embeddings():
    with open('baybee_embeddings.pkl', 'rb') as f:
        return pickle.load(f)


def find_baybee(test_image_path, embeddings_data, threshold=0.19):
    stored_embeddings = [e['embedding'] for e in embeddings_data['embeddings']]
    model_name = embeddings_data['model_name']
    detector = embeddings_data['detector']

    try:
        test_faces = DeepFace.represent(
            img_path=test_image_path,
            model_name=model_name,
            detector_backend=detector,
            enforce_detection=True,
            align=True
        )
    except:
        return {'found': False, 'faces_detected': 0, 'all_faces': [], 'best_confidence': 0}

    results = []
    for face in test_faces:
        test_emb = np.array(face['embedding'])
        distances = []
        for stored_emb in stored_embeddings:
            stored = np.array(stored_emb)
            cos_sim = np.dot(test_emb, stored) / (np.linalg.norm(test_emb) * np.linalg.norm(stored))
            distances.append(1 - cos_sim)

        min_distance = min(distances)
        avg_distance = np.mean(sorted(distances)[:5])  # average of 5 closest matches
        confidence = round((1 - avg_distance) * 100, 1)
        is_baybee = avg_distance < threshold

        results.append({
            'is_baybee': is_baybee,
            'confidence': confidence,
            'distance': round(avg_distance, 4),
            'facial_area': face['facial_area']
        })

    baybee_found = any(r['is_baybee'] for r in results)
    best_match = max(results, key=lambda x: x['confidence'])

    return {
        'found': baybee_found,
        'faces_detected': len(results),
        'best_confidence': best_match['confidence'],
        'all_faces': results
    }


def draw_result(image_path, result):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, figsize=(10, 8))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    ax.imshow(img_rgb)

    for face in result['all_faces']:
        area = face['facial_area']
        x, y, w, h = area['x'], area['y'], area['w'], area['h']
        color = '#00FF88' if face['is_baybee'] else '#FF4444'
        label = f"Baybee ✓ {face['confidence']}%" if face['is_baybee'] else f"Not Baybee"

        rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 10, label, color=color, fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#0a0a0a', alpha=0.8))

    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig


# ── UI ──────────────────────────────────────────────────────

st.markdown('<p class="title">The Baybee Finder</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload any photo. The data never lies.</p>', unsafe_allow_html=True)

embeddings_data = load_embeddings()

uploaded_file = st.file_uploader(
    "Drop a photo here",
    type=['jpg', 'jpeg', 'png'],
    label_visibility='collapsed'
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.image(uploaded_file, use_column_width=True, caption="Uploaded photo")

    with st.spinner(" Scanning for Baybee..."):
        result = find_baybee(tmp_path, embeddings_data, threshold=0.19)

    if result['faces_detected'] == 0:
        st.markdown('<div class="result-not-found"><h3> No faces detected</h3><p>Try a clearer photo with visible faces.</p></div>', unsafe_allow_html=True)
    elif result['found']:
        st.markdown(f'<div class="result-found"><h2> Baybee is here!</h2><p>She was found in this photo.</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="confidence-box"><p style="color:rgba(240,236,228,0.5);font-size:12px;letter-spacing:2px;text-transform:uppercase;">Confidence</p><p style="font-size:2rem;font-weight:500;color:#ffc94d;">{result["best_confidence"]}%</p></div>', unsafe_allow_html=True)
        fig = draw_result(tmp_path, result)
        st.pyplot(fig)
    else:
        st.markdown(f'<div class="result-not-found"><h2> Baybee is not in this photo</h2><p>Best match: {result["best_confidence"]}% — below threshold.</p></div>', unsafe_allow_html=True)
        fig = draw_result(tmp_path, result)
        st.pyplot(fig)

    st.markdown(f'<p style="color:rgba(240,236,228,0.3);font-size:12px;text-align:center;">Faces detected: {result["faces_detected"]} — Model: FaceNet512</p>', unsafe_allow_html=True)
    os.unlink(tmp_path)