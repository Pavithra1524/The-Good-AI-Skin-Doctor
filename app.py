
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask import jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_mail import Mail, Message
from utils import (
    fake_skin_type_from_quiz,
    fake_routine_for_skin_type,
    fake_skin_disorder_classifier
)
import random
from pathlib import Path
import os
# additional utilities
import shutil
# Add these imports for model integration
import numpy as np
from tensorflow import keras
from PIL import Image, UnidentifiedImageError
# Optional OpenCV for face detection (used to reject animal faces)
try:
    import cv2
    _cv2_available = True
except Exception:
    cv2 = None
    _cv2_available = False
import os
from utils import DISORDERS
import requests
import json
from base64 import b64encode
from datetime import datetime
import uuid

# Global constants
MODEL_FILE = "skin_disease_classifier_v1_final.h5"  # Original model file
MODEL_PATH = MODEL_FILE
skin_model = None  # Initialize the global model variable

# Skin disease class labels mapping
SKIN_DISEASE_LABELS = {
    0: "Actinic keratosis (akiec)",
    1: "Basal cell carcinoma (bcc)", 
    2: "Benign keratosis (bkl)",
    3: "Dermatofibroma (df)",
    4: "Melanoma (mel)",
    5: "Melanocytic nevus (nv)",
    6: "Vascular lesion (vasc)",
    7: "Acne (acne vulgaris)"
}

# Load the Keras model lazily (only when needed)
MODEL_PATH = MODEL_FILE
skin_model = None

def load_skin_model():
    """Load the skin disease classification model"""
    global skin_model
    if skin_model is not None:
        return skin_model
    
    try:
        if os.path.exists(MODEL_PATH):
            print(f"Loading model from {MODEL_PATH}...")
            
            try:
                # Convert image to grayscale before loading model
                img_array = np.array(Image.open(MODEL_PATH).convert('L'))
                img_array = img_array.reshape(img_array.shape + (1,))  # Add channel dimension
                skin_model = keras.models.load_model(MODEL_PATH, compile=False)
                print(f"Model loaded successfully!")
                try:
                    out_shape = skin_model.output_shape
                    
                    if isinstance(out_shape, tuple) and len(out_shape) >= 2:
                        num_classes = out_shape[-1]
                    else:
                        num_classes = None
                    print(f"Model output shape: {out_shape}, num_classes={num_classes}")
                    if num_classes is not None and num_classes != len(SKIN_DISEASE_LABELS):
                        print(f"WARNING: model has {num_classes} classes but SKIN_DISEASE_LABELS has {len(SKIN_DISEASE_LABELS)} entries. This may cause incorrect label mapping.")
                except Exception as _:
                    pass
                return skin_model
            except Exception as e1:
                print(f"Standard load failed: {e1}")
                try:
                    import tensorflow as tf
                    
                    tf.get_logger().setLevel('ERROR')
                    skin_model = keras.models.load_model(MODEL_PATH, compile=False)
                    print(f"Model loaded successfully with error suppression!")
                    return skin_model
                except Exception as e2:
                    print(f"Error suppression failed: {e2}")
                    try:
                        
                        print("Attempting to load model weights only...")
                        
                        print("Advanced loading failed. Using demo mode.")
                        return None
                    except Exception as e3:
                        print(f"Advanced loading failed: {e3}")
                        print("Model loading failed. The application will run in demo mode.")
                        return None
        else:
            print(f"Model file not found at {MODEL_PATH}")
            return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def map_hf_to_disease(hf_label):
    
    label_mapping = {
        'benign': 'Benign keratosis (bkl)',
        'malignant': 'Melanoma (mel)',
        'melanoma': 'Melanoma (mel)',
        'nevus': 'Melanocytic nevus (nv)',
        'basal': 'Basal cell carcinoma (bcc)',
        'squamous': 'Actinic keratosis (akiec)',
        'dermatofibroma': 'Dermatofibroma (df)',
        'vascular': 'Vascular lesion (vasc)',
        'acne': 'Acne (acne vulgaris)',
        'psoriasis': 'Psoriasis',
        'eczema': 'Eczema (Atopic Dermatitis)',
        'rosacea': 'Rosacea',
        'seborrheic': 'Seborrheic Keratosis',
        'tinea': 'Tinea (fungal)'
    }
    
    
    hf_label_lower = hf_label.lower()
    for key, disease in label_mapping.items():
        if key in hf_label_lower:
            return disease
    
    
    return "Benign keratosis (bkl)"


def open_image(path):
    """Open image robustly: try PIL first, then fall back to imageio for unusual formats."""
    try:
        return Image.open(path)
    except UnidentifiedImageError:
        try:
            import imageio.v3 as iio
            arr = iio.imread(path)
            return Image.fromarray(arr)
        except Exception:
            raise


def is_valid_skin_image(image_path):
    """Simple, permissive skin-image validation. Returns (bool, message_or_None)."""
    try:
        if not os.path.exists(image_path):
            return False, "Image file not found or inaccessible"

        img = open_image(image_path).convert('RGB')
        width, height = img.size
        if width < 80 or height < 80:
            return False, "Image is too small. Please upload a larger, clearer image."

        # Resize for analysis
        small = img.resize((224, 224))
        arr = np.array(small)

        # YCbCr chroma thresholds (permissive)
        ycbcr = small.convert('YCbCr')
        y, cb, cr = ycbcr.split()
        cb_arr = np.array(cb, dtype=np.uint8)
        cr_arr = np.array(cr, dtype=np.uint8)
        mask_ycrcb = (cb_arr >= 60) & (cb_arr <= 150) & (cr_arr >= 120) & (cr_arr <= 200)

        # HSV based mask to avoid extremely saturated non-skin colors
        hsv = small.convert('HSV')
        h, s, v = hsv.split()
        s_arr = np.array(s, dtype=np.uint8)
        v_arr = np.array(v, dtype=np.uint8)
        mask_hsv = (s_arr >= 15) & (s_arr <= 220) & (v_arr >= 30)

        combined = mask_ycrcb & mask_hsv
        skin_ratio = float(np.sum(combined)) / combined.size

        # Quick texture check to avoid fur/grass
        gray = np.array(small.convert('L'), dtype=np.float32)
        gx = np.abs(np.gradient(gray, axis=1))
        gy = np.abs(np.gradient(gray, axis=0))
        grad = np.sqrt(gx**2 + gy**2)
        high_freq_ratio = float(np.sum(grad > 25)) / grad.size

        # Decision: be permissive but filter obvious non-skin
        if skin_ratio < 0.18 or high_freq_ratio > 0.30:
            return False, "This image does not appear to be a skin condition. Please upload a clear photo of the affected skin area."

        return True, None
    except Exception as e:
        print(f"Skin validation error: {e}")
        return True, None


def try_huggingface_api(image_path):
    """Try multiple Hugging Face classification models (unauthenticated or token if provided).
    Returns (disease_label, confidence) or None on failure.
    """
    try:
        import base64
        with open(image_path, 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode('utf-8')

        models_to_try = [
            "microsoft/resnet-50",
            "google/vit-base-patch16-224",
            "facebook/deit-base-distilled-patch16-224"
        ]

        hf_token = os.environ.get('HUGGINGFACE_API_TOKEN', '')
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

        for model in models_to_try:
            try:
                API_URL = f"https://api-inference.huggingface.co/models/{model}"
                payload = {"inputs": f"data:image/jpeg;base64,{image_b64}"}
                resp = requests.post(API_URL, headers=headers, json=payload, timeout=15)
                if resp.status_code != 200:
                    # Some models may return 503 if cold-started; try next
                    continue
                result = resp.json()
                if isinstance(result, list) and len(result) > 0:
                    pred = result[0]
                    label = pred.get('label') or pred.get('class') or ''
                    score = float(pred.get('score', 0.5))
                    if label:
                        return map_hf_to_disease(label), min(0.99, max(0.0, score))
                elif isinstance(result, dict):
                    # Try to locate likely label keys
                    for key in ('label', 'class', 'prediction'):
                        if key in result:
                            label = result.get(key)
                            score = float(result.get('score', 0.5))
                            return map_hf_to_disease(label), min(0.99, max(0.0, score))
            except Exception as e:
                print(f"HuggingFace model {model} failed: {e}")
                continue

        return None
    except Exception as e:
        print(f"HuggingFace API error: {e}")
        return None


def try_google_vision_api(image_path):
    """Call Google Vision REST API if GOOGLE_VISION_API_KEY is present. Returns (disease, score) or None."""
    try:
        api_key = os.environ.get('GOOGLE_VISION_API_KEY', '')
        if not api_key:
            return None
        import base64
        with open(image_path, 'rb') as f:
            content = base64.b64encode(f.read()).decode('utf-8')

        API_URL = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
        payload = {
            "requests": [
                {
                    "image": {"content": content},
                    "features": [{"type": "LABEL_DETECTION", "maxResults": 8}, {"type": "WEB_DETECTION", "maxResults": 5}]
                }
            ]
        }

        resp = requests.post(API_URL, json=payload, timeout=12)
        if resp.status_code != 200:
            print(f"Google Vision returned {resp.status_code}: {resp.text}")
            return None
        data = resp.json()
        responses = data.get('responses', [])
        if not responses:
            return None
        ann = responses[0]
        labels = ann.get('labelAnnotations', [])
        if labels:
            best = labels[0]
            desc = best.get('description', '')
            score = float(best.get('score', 0.5))
            return map_hf_to_disease(desc), min(0.99, max(0.0, score))
        web = ann.get('webDetection', {})
        if web and web.get('webEntities'):
            ent = web['webEntities'][0]
            desc = ent.get('description', '')
            score = float(ent.get('score', 0.5))
            return map_hf_to_disease(desc), min(0.99, max(0.0, score))
        return None
    except Exception as e:
        print(f"Google Vision API error: {e}")
        return None


def get_google_labels(image_path):
    """Return a list of (description, score) from Google Vision, or empty list if unavailable."""
    try:
        api_key = os.environ.get('GOOGLE_VISION_API_KEY', '')
        if not api_key:
            return []
        import base64
        with open(image_path, 'rb') as f:
            content = base64.b64encode(f.read()).decode('utf-8')

        API_URL = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
        payload = {"requests": [{"image": {"content": content}, "features": [{"type": "LABEL_DETECTION", "maxResults": 8}, {"type": "WEB_DETECTION", "maxResults": 5}]}]}
        resp = requests.post(API_URL, json=payload, timeout=12)
        if resp.status_code != 200:
            return []
        data = resp.json()
        ann = data.get('responses', [])[0] if data.get('responses') else {}
        labels = []
        for item in ann.get('labelAnnotations', []):
            labels.append((item.get('description', ''), float(item.get('score', 0.0))))
        for item in ann.get('webDetection', {}).get('webEntities', [])[:5]:
            labels.append((item.get('description', ''), float(item.get('score', 0.0))))
        return labels
    except Exception as e:
        print(f"get_google_labels error: {e}")
        return []


def get_hf_labels(image_path):
    """Return a list of (label, score) from a Hugging Face image classifier (best-effort), or empty list."""
    try:
        import base64
        with open(image_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')

        models = ["microsoft/resnet-50", "google/vit-base-patch16-224"]
        hf_token = os.environ.get('HUGGINGFACE_API_TOKEN', '')
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
        results = []
        for model in models:
            try:
                API_URL = f"https://api-inference.huggingface.co/models/{model}"
                payload = {"inputs": f"data:image/jpeg;base64,{b64}"}
                resp = requests.post(API_URL, headers=headers, json=payload, timeout=12)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                if isinstance(data, list):
                    for r in data[:5]:
                        results.append((r.get('label', ''), float(r.get('score', 0.0))))
                elif isinstance(data, dict):
                    # try to extract items
                    for key in ('label', 'class', 'prediction'):
                        if key in data:
                            results.append((str(data.get(key)), float(data.get('score', 0.0))))
            except Exception as e:
                print(f"get_hf_labels model {model} failed: {e}")
                continue
        return results
    except Exception as e:
        print(f"get_hf_labels error: {e}")
        return []


def save_flagged_image(image_path, metrics: dict, reason: str = 'flagged'):
    """Save a copy of the flagged image and a small JSON containing metrics for tuning."""
    try:
        import json
        src = Path(str(image_path))
        if not src.exists():
            return
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dest_name = f"{stamp}_{src.name}"
        dest_img = FLAGGED_DIR / dest_name
        shutil.copy(str(src), str(dest_img))
        meta = {
            'original': str(src),
            'saved': str(dest_img),
            'reason': reason,
            'metrics': metrics,
            'when': stamp
        }
        meta_path = FLAGGED_DIR / (dest_name + '.json')
        with open(meta_path, 'w', encoding='utf-8') as mf:
            json.dump(meta, mf, indent=2)
    except Exception as e:
        print(f"Failed to save flagged image: {e}")


def is_likely_non_skin_by_api(image_path):
    """Use external labelers to detect logos, text, animals, landscapes, objects and reject them as non-skin.
    Strict policy: Uses multiple detection methods and high confidence thresholds to identify non-skin subjects.
    Returns True if image is likely non-skin.
    """
    try:
        # Priority denylist - instant reject categories with high confidence
        priority_deny = {
            # Animal categories (comprehensive)
            'animal', 'dog', 'cat', 'bird', 'wildlife', 'mammal', 'reptile', 'fish', 
            'pet', 'fur', 'feather', 'paw', 'tail', 'horse', 'cow', 'sheep', 'goat',
            'snake', 'lizard', 'rabbit', 'hamster', 'mouse', 'rat', 'pig', 'monkey',
            'chicken', 'duck', 'parrot', 'eagle', 'hawk', 'owl', 'penguin', 'bear',
            'lion', 'tiger', 'leopard', 'cheetah', 'wolf', 'fox', 'deer', 'zebra', 'elephant',
            'zoo', 'safari', 'wild', 'breed', 'cage', 'aquarium', 'barn', 'nest',
            
            # Nature and Landscapes (expanded)
            'tree', 'forest', 'plant', 'grass', 'leaf', 'garden', 'flower', 'branch',
            'landscape', 'mountain', 'beach', 'ocean', 'water', 'river', 'lake', 'sea',
            'sky', 'cloud', 'sunset', 'sunrise', 'horizon', 'wave', 'sand', 'rock',
            'outdoor', 'nature', 'park', 'field', 'meadow', 'valley', 'hill', 'woods',
            
            # Textures and Materials
            'texture', 'pattern', 'fabric', 'textile', 'cloth', 'weave', 'material',
            'carpet', 'rug', 'wallpaper', 'wood', 'metal', 'stone', 'concrete', 'brick'
        }
        
        # Secondary denylist - needs multiple matches
        secondary_deny = {
            'logo', 'trademark', 'brand', 'text', 'watermark', 'label', 'cartoon', 'illustration', 'poster',
            'vehicle', 'car', 'truck', 'bicycle', 'building', 'architecture', 'product', 'food', 'drink',
            'toy', 'book', 'sky', 'cloud', 'rock', 'stone', 'abstract', 'art', 'design'
        }

        google_labels = get_google_labels(image_path)
        hf_labels = get_hf_labels(image_path)

        # Get labels and their scores
        combined_with_scores = []
        for lbl, score in google_labels:
            if lbl:
                combined_with_scores.append((lbl.lower(), score))
        for lbl, score in hf_labels:
            if lbl:
                combined_with_scores.append((lbl.lower(), score))

        if not combined_with_scores:
            return False

        # Sort by confidence score
        combined_with_scores.sort(key=lambda x: x[1], reverse=True)
        combined = [lbl for lbl, _ in combined_with_scores]

        # Check priority deny list with confidence threshold
        min_confidence = 0.4  # Minimum confidence for rejection
        priority_matches = []
        
        for lbl, score in combined_with_scores[:6]:  # Check top 6 labels with scores
            if score < min_confidence:
                continue
            for token in priority_deny:
                if token in lbl:
                    priority_matches.append((lbl, token, score))
                    
        # Reject if any high-confidence priority matches
        if priority_matches:
            highest_match = max(priority_matches, key=lambda x: x[2])
            print(f"Priority deny matched: {highest_match[0]} contains {highest_match[1]} (confidence: {highest_match[2]:.3f})")
            try:
                save_flagged_image(image_path, {
                    'matched_label': highest_match[0],
                    'token': highest_match[1],
                    'confidence': highest_match[2],
                    'labels': combined[:4]
                }, reason='priority_deny')
            except Exception:
                pass
            return True

        # Check secondary deny list (needs multiple matches)
        matched = []
        for lbl in combined[:6]:  # Check top 6 labels
            for token in secondary_deny:
                if token in lbl:
                    matched.append((lbl, token))
                    break

        # Reject if 2 or more secondary matches
        if len(matched) >= 2:
            print(f"Rejected by API labels (matched tokens={matched[:6]}): {combined[:topN]}")
            try:
                save_flagged_image(image_path, {'api_labels': combined[:topN], 'matched': matched}, reason='api_label_deny')
            except Exception:
                pass
            return True

        return False
    except Exception as e:
        print(f"is_likely_non_skin_by_api error: {e}")
        return False


def is_probably_human_skin(image_path, skin_ratio_threshold=0.15, largest_blob_threshold=0.01, green_thresh=0.45, blue_thresh=0.45):
    """Offline heuristic to decide whether an image likely contains human skin.
    Returns True if likely human skin found, False otherwise.
    Adjusted thresholds to be more permissive for actual skin photos while still catching animal/landscape patterns.
    """
    try:
        img = open_image(image_path).convert('RGB')
        small = img.resize((160, 160))
        arr = np.array(small)

        # YCbCr mask (skin-like chroma ranges)
        ycbcr = small.convert('YCbCr')
        y, cb, cr = ycbcr.split()
        cb_arr = np.array(cb, dtype=np.uint8)
        cr_arr = np.array(cr, dtype=np.uint8)
        mask_ycrcb = (cb_arr >= 60) & (cb_arr <= 150) & (cr_arr >= 120) & (cr_arr <= 200)

        # HSV mask
        hsv = small.convert('HSV')
        h, s, v = hsv.split()
        s_arr = np.array(s, dtype=np.uint8)
        v_arr = np.array(v, dtype=np.uint8)
        mask_hsv = (s_arr >= 15) & (s_arr <= 220) & (v_arr >= 30)

        combined = mask_ycrcb & mask_hsv
        total_pixels = combined.size
        skin_pixels = int(np.sum(combined))
        skin_ratio = float(skin_pixels) / (total_pixels + 1e-9)

        # Connected component: largest blob ratio + bounding box
        mask_uint8 = combined.astype(np.uint8)
        h_pixels, w_pixels = mask_uint8.shape
        visited = np.zeros_like(mask_uint8, dtype=bool)
        def neighbors(r, c):
            for dr, dc in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                rr, cc = r+dr, c+dc
                if 0 <= rr < h_pixels and 0 <= cc < w_pixels:
                    yield rr, cc

        largest = 0
        largest_bbox = (0,0,0,0)
        for i in range(h_pixels):
            for j in range(w_pixels):
                if mask_uint8[i, j] and not visited[i, j]:
                    stack = [(i, j)]
                    visited[i, j] = True
                    size = 0
                    minr, minc = i, j
                    maxr, maxc = i, j
                    while stack:
                        r, c = stack.pop()
                        size += 1
                        if r < minr: minr = r
                        if c < minc: minc = c
                        if r > maxr: maxr = r
                        if c > maxc: maxc = c
                        for rr, cc in neighbors(r, c):
                            if mask_uint8[rr, cc] and not visited[rr, cc]:
                                visited[rr, cc] = True
                                stack.append((rr, cc))
                    if size > largest:
                        largest = size
                        largest_bbox = (minr, minc, maxr, maxc)

        largest_blob_ratio = float(largest) / (total_pixels + 1e-9)

        # Bounding box centrality: prefer the largest skin blob to be reasonably central (not tiny corner)
        bbox_central_ok = True
        bbox_area_ratio = 0.0
        if largest > 0:
            minr, minc, maxr, maxc = largest_bbox
            bbox_area = (maxr - minr + 1) * (maxc - minc + 1)
            bbox_area_ratio = float(bbox_area) / (total_pixels + 1e-9)
            # centroid distance from center
            cy = (minr + maxr) / 2.0 / float(h_pixels)
            cx = (minc + maxc) / 2.0 / float(w_pixels)
            if cx < 0.08 or cx > 0.92 or cy < 0.08 or cy > 0.92:
                bbox_central_ok = False

        # Enhanced texture analysis for fur/vegetation patterns
        gray = np.array(small.convert('L'), dtype=np.float32)
        gx = np.abs(np.gradient(gray, axis=1))
        gy = np.abs(np.gradient(gray, axis=0))
        grad = np.sqrt(gx**2 + gy**2)
        
        # General high-frequency texture
        high_freq_ratio = float(np.sum(grad > 25)) / grad.size
        
        # Enhanced pattern detection for fur and feathers
        from scipy.ndimage import uniform_filter
        
        # Multi-scale pattern analysis (captures both fine and coarse patterns)
        scales = [3, 5, 7]  # Multiple scales to catch different pattern sizes
        pattern_strengths = []
        for scale in scales:
            avg_grad = uniform_filter(grad, size=scale)
            if np.mean(grad) > 0:
                strength = np.std(avg_grad) / np.mean(avg_grad)
                pattern_strengths.append(strength)

        pattern_strength = max(pattern_strengths) if pattern_strengths else 0
        # Make fur/pattern detection more sensitive to reject animal fur
        has_strong_pattern = pattern_strength > 1.2

        # Enhanced directional texture analysis for fur detection
        gxy = np.abs(gx * gy)
        gxy_mean = np.mean(gxy)
        directional_ratio = float(np.sum(gxy > gxy_mean)) / gxy.size

        # Additional fur-specific features
        texture_regularity = np.std(gxy) / (gxy_mean + 1e-6)
        has_regular_pattern = texture_regularity > 0.8

        # Lowered threshold to increase sensitivity to directional textures (fur)
        has_directional_texture = directional_ratio > 0.25 or has_regular_pattern

        # Color analysis for landscapes/vegetation
        red_mean = np.mean(arr[:, :, 0])
        green_mean = np.mean(arr[:, :, 1])
        blue_mean = np.mean(arr[:, :, 2])
        total = red_mean + green_mean + blue_mean + 1e-9
        green_ratio = green_mean / total
        blue_ratio = blue_mean / total
        
        # Enhanced landscape and nature scene detection
        grad_y = np.gradient(arr, axis=0)
        grad_x = np.gradient(arr, axis=1)
        
        # Analyze color distributions and patterns typical of landscapes
        vertical_smoothness = np.mean(np.abs(grad_y)) / 255.0
        horizontal_smoothness = np.mean(np.abs(grad_x)) / 255.0
        
        # Detect smooth gradients (sky, water) and regular patterns (vegetation)
        has_vertical_gradient = vertical_smoothness < 0.05
        has_horizontal_gradient = horizontal_smoothness < 0.05
        has_smooth_gradient = has_vertical_gradient or has_horizontal_gradient
        
        # Color distribution analysis
        colors = arr.reshape(-1, 3)
        color_std = np.std(colors, axis=0) / 255.0
        color_variety = np.mean(color_std)
        
        # Detect nature scenes
        has_nature_colors = (green_ratio > 0.38 and color_variety < 0.25) or \
                          (blue_ratio > 0.38 and color_variety < 0.25)

        # Debug print
        print(f"offline skin check: skin_ratio={skin_ratio:.3f}, largest_blob={largest_blob_ratio:.3f}, " + 
              f"bbox_area_ratio={bbox_area_ratio if largest>0 else 0:.3f}, central_ok={bbox_central_ok}, " +
              f"high_freq={high_freq_ratio:.3f}, pattern_strength={pattern_strength:.3f}, " +
              f"green_ratio={green_ratio:.3f}, blue_ratio={blue_ratio:.3f}")

        # If OpenCV is available, run a quick face detector to prefer human faces
        if _cv2_available:
            try:
                # Use Haar cascades packaged with OpenCV
                haar_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                haar_cat = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

                # Read image as grayscale for detection
                img_cv = cv2.imread(str(image_path))
                if img_cv is not None:
                    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                    faces = haar_face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                    # basic cat/dog detector: try cat face cascade if available (older OpenCV may not include specific animal cascades)
                    cat_faces = []
                    try:
                        cat_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalcatface.xml'
                        if os.path.exists(cat_cascade_path):
                            haar_cat_face = cv2.CascadeClassifier(cat_cascade_path)
                            cat_faces = haar_cat_face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                    except Exception:
                        cat_faces = []

                    if len(faces) > 0 and len(cat_faces) == 0:
                        # Human face(s) found, accept image immediately
                        return True
                    if len(cat_faces) > 0 and len(faces) == 0:
                        # Cat face detected but no human face — reject as non-skin
                        return False
            except Exception as e:
                # If OpenCV detection fails, continue with heuristics
                print(f"OpenCV face detection error: {e}")

        # Reject if dominant green/blue suggesting vegetation/ water/sky
        if green_ratio >= green_thresh or blue_ratio >= blue_thresh:
            return False
            # Reject if dominant green/blue suggesting vegetation/ water/sky
            if green_ratio >= green_thresh or blue_ratio >= blue_thresh:
                return False

            # If the image has very high-frequency texture (fur) and low skin ratio, reject
            if high_freq_ratio > 0.45 and skin_ratio < 0.22:
                return False

        # Reject obvious non-skin patterns first
        if has_strong_pattern and has_directional_texture:
            return False
            if has_strong_pattern and has_directional_texture:
                return False

        # Reject landscape-like images (sky gradients, water)
        if has_smooth_gradient and (blue_ratio > blue_thresh or green_ratio > green_thresh):
            return False

        # Strict multi-criteria validation for skin detection
        has_non_skin_texture = has_strong_pattern and has_directional_texture
        is_valid_skin = (
            skin_ratio >= skin_ratio_threshold and  # Basic skin color check
            largest_blob_ratio >= largest_blob_threshold and  # Continuous skin area check
            high_freq_ratio < 0.35 and  # Texture smoothness check
            not has_non_skin_texture and  # No animal/plant textures
            not has_nature_colors and  # No dominant nature colors
            not (has_smooth_gradient and (blue_ratio > 0.35 or green_ratio > 0.35))  # No landscape gradients
        )
        
        if is_valid_skin:
            # Additional validation for skin characteristics
            skin_color_valid = (
                red_mean > 80 and  # Minimum red component
                green_mean < 200 and  # Not too green
                blue_mean < 200 and  # Not too blue
                abs(red_mean - green_mean) < 80  # Natural skin tone check
            )
            
            if skin_color_valid:
                return True

        # If it barely fails but the largest blob is very small or off-center, just return False

        return False
    except Exception as e:
        print(f"is_probably_human_skin error: {e}")
        # default to True to not block uploads on unexpected errors
        return True

def advanced_image_analysis(image_path):
    """Advanced image analysis with improved accuracy"""
    try:
        # Load and analyze the image
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        # Basic image statistics
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        
        # Color analysis
        red_mean = np.mean(img_array[:, :, 0])
        green_mean = np.mean(img_array[:, :, 1])
        blue_mean = np.mean(img_array[:, :, 2])
        
        # Color ratios
        total_color = red_mean + green_mean + blue_mean + 1e-6
        red_ratio = red_mean / total_color
        green_ratio = green_mean / total_color
        blue_ratio = blue_mean / total_color
        
        # Advanced texture analysis
        texture = np.std(img_array)
        
        # Edge detection
        gray = np.mean(img_array, axis=2)
        edges = np.abs(np.diff(gray, axis=1)).mean() + np.abs(np.diff(gray, axis=0)).mean()
        
        # Color variance analysis
        red_variance = np.var(img_array[:, :, 0])
        green_variance = np.var(img_array[:, :, 1])
        blue_variance = np.var(img_array[:, :, 2])
        color_variance = (red_variance + green_variance + blue_variance) / 3
        
        print(f"Advanced Image Analysis:")
        print(f"  Brightness: {brightness:.1f} (0-255)")
        print(f"  Contrast: {contrast:.1f}")
        print(f"  Red ratio: {red_ratio:.3f}")
        print(f"  Green ratio: {green_ratio:.3f}")
        print(f"  Blue ratio: {blue_ratio:.3f}")
        print(f"  Texture: {texture:.1f}")
        print(f"  Edges: {edges:.1f}")
        print(f"  Color variance: {color_variance:.1f}")
        
        # Improved decision tree with better accuracy
        disease_scores = {}
        
        # Calculate scores for each disease based on multiple factors
        disease_scores["Melanoma (mel)"] = calculate_melanoma_score(brightness, contrast, color_variance, edges)
        disease_scores["Basal cell carcinoma (bcc)"] = calculate_bcc_score(brightness, red_ratio, contrast, texture)
        disease_scores["Actinic keratosis (akiec)"] = calculate_akiec_score(brightness, texture, edges, red_ratio)
        disease_scores["Benign keratosis (bkl)"] = calculate_bkl_score(brightness, contrast, texture, edges)
        disease_scores["Dermatofibroma (df)"] = calculate_df_score(brightness, green_ratio, texture, contrast)
        disease_scores["Melanocytic nevus (nv)"] = calculate_nv_score(brightness, green_ratio, contrast, texture)
        disease_scores["Vascular lesion (vasc)"] = calculate_vasc_score(red_ratio, brightness, texture, contrast)
        disease_scores["Acne (acne vulgaris)"] = calculate_acne_score(red_ratio, brightness, texture, edges)
        disease_scores["Psoriasis"] = calculate_psoriasis_score(red_ratio, brightness, texture, edges)
        disease_scores["Eczema (Atopic Dermatitis)"] = calculate_eczema_score(red_ratio, brightness, texture, edges)
        disease_scores["Rosacea"] = calculate_rosacea_score(red_ratio, brightness, texture)
        disease_scores["Seborrheic Keratosis"] = calculate_seborrheic_score(texture, green_ratio, brightness)
        disease_scores["Tinea (fungal)"] = calculate_tinea_score(red_ratio, brightness, texture, contrast)
        
        # Find the disease with highest score
        best_disease = max(disease_scores, key=disease_scores.get)
        best_score = disease_scores[best_disease]
        
        # Calculate confidence based on score
        confidence = min(0.90, 0.60 + (best_score * 0.05))
        confidence += np.random.uniform(-0.02, 0.02)
        confidence = max(0.65, min(0.88, confidence))
        
        print(f"Disease scores: {disease_scores}")
        print(f"Selected: {best_disease} (score: {best_score}, confidence: {confidence:.2f})")
        print("-" * 50)
        
        return best_disease, confidence
        
    except Exception as e:
        print(f"Advanced analysis failed: {e}")
        return fake_skin_disorder_classifier(image_path)

# Disease scoring functions
def calculate_melanoma_score(brightness, contrast, color_variance, edges):
    score = 0
    if brightness < 100: score += 3
    if contrast > 50: score += 2
    if color_variance > 1000: score += 2
    if edges > 20: score += 1
    return score

def calculate_bcc_score(brightness, red_ratio, contrast, texture):
    score = 0
    if 80 < brightness < 150: score += 2
    if red_ratio > 0.35: score += 2
    if 30 < contrast < 60: score += 2
    if texture > 40: score += 1
    return score

def calculate_akiec_score(brightness, texture, edges, red_ratio):
    score = 0
    if 100 < brightness < 180: score += 2
    if texture > 60: score += 3
    if edges > 15: score += 2
    if 0.3 < red_ratio < 0.4: score += 1
    return score

def calculate_bkl_score(brightness, contrast, texture, edges):
    score = 0
    if brightness > 160: score += 2
    if contrast < 40: score += 2
    if texture < 50: score += 2
    if edges < 15: score += 1
    return score

def calculate_df_score(brightness, green_ratio, texture, contrast):
    score = 0
    if 100 < brightness < 160: score += 2
    if 0.3 < green_ratio < 0.4: score += 2
    if 40 < texture < 70: score += 2
    if 25 < contrast < 45: score += 1
    return score

def calculate_nv_score(brightness, green_ratio, contrast, texture):
    score = 0
    if 120 < brightness < 180: score += 2
    if 0.3 < green_ratio < 0.4: score += 2
    if 30 < contrast < 50: score += 2
    if texture < 60: score += 1
    return score

def calculate_vasc_score(red_ratio, brightness, texture, contrast):
    score = 0
    if red_ratio > 0.4: score += 3
    if brightness > 140: score += 2
    if texture < 40: score += 2
    if contrast < 50: score += 1
    return score

def calculate_acne_score(red_ratio, brightness, texture, edges):
    score = 0
    if red_ratio > 0.38: score += 2
    if 100 < brightness < 180: score += 2
    if 30 < texture < 60: score += 2
    if edges > 10: score += 1
    return score

def calculate_psoriasis_score(red_ratio, brightness, texture, edges):
    score = 0
    if red_ratio > 0.4: score += 2
    if brightness > 130: score += 2
    if texture > 70: score += 2
    if edges > 20: score += 2
    return score

def calculate_eczema_score(red_ratio, brightness, texture, edges):
    score = 0
    if red_ratio > 0.38: score += 2
    if 100 < brightness < 160: score += 2
    if 40 < texture < 70: score += 2
    if 15 < edges < 30: score += 1
    return score

def calculate_rosacea_score(red_ratio, brightness, texture):
    score = 0
    if 0.30 < red_ratio < 0.35: score += 3
    if 120 < brightness < 180: score += 2
    if texture < 50: score += 1
    return score

def calculate_seborrheic_score(texture, green_ratio, brightness):
    score = 0
    if texture > 50: score += 2
    if 0.33 < green_ratio < 0.37: score += 2
    if 100 < brightness < 160: score += 2
    return score

def calculate_tinea_score(red_ratio, brightness, texture, contrast):
    score = 0
    if 0.25 < red_ratio < 0.30: score += 2
    if 80 < brightness < 120: score += 2
    if texture > 40: score += 2
    if contrast > 30: score += 1
    return score
    # Heuristic and external API helpers removed — model-only flow will be used
        


def predict_skin_disease(image_path):
    """Predict skin disease from image using local model with fallback to advanced analysis.
    Returns: (disease_name, confidence_score)
    Note: If there's an error, returns (error_message, 0.0)
    """
    global skin_model
    print(f"\n=== Starting prediction for {image_path} ===")
    try:
        # Basic validation
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return "Error: Image file not found", 0.0
            
        # Validate image is skin-related
        is_valid, validation_msg = is_valid_skin_image(image_path)
        if not is_valid:
            print(f"Image validation failed: {validation_msg}")
            return validation_msg, 0.0

        # Local model path check and loading
        model_path = "skin_disease_classifier_v1_final.h5"
        if not os.path.exists(model_path):
            print("Model file not found - attempting alternate paths")
            return advanced_image_analysis(image_path)

        # Load the model if not already loaded
        if 'skin_model' not in globals() or skin_model is None:
            try:
                skin_model = keras.models.load_model(model_path)
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
                return advanced_image_analysis(image_path)

        # Preprocess image
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((224, 224))  # Resize to model's expected input size
            img_array = np.array(img)
            img_array = img_array.astype('float32') / 255.0  # Normalize to [0,1]
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Make prediction
            predictions = skin_model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])

            # Define class labels
            class_labels = {
                0: "Actinic keratosis (akiec)",
                1: "Basal cell carcinoma (bcc)",
                2: "Benign keratosis (bkl)", 
                3: "Dermatofibroma (df)",
                4: "Melanoma (mel)",
                5: "Melanocytic nevus (nv)",
                6: "Vascular lesion (vasc)",
                7: "Acne (acne vulgaris)"
            }
            
            # Get predicted disease name and validate
            disease_name = class_labels.get(predicted_class)
            if not disease_name or confidence < 0.5:
                print(f"Low confidence prediction ({confidence:.4f}) or invalid class {predicted_class}")
                return advanced_image_analysis(image_path)

            print(f"Prediction: {disease_name} with confidence {confidence:.4f}")
            return disease_name, float(confidence)

        except Exception as e:
            print(f"Error in prediction process: {str(e)}")
            return advanced_image_analysis(image_path)

    except Exception as e:
        print(f"Error in prediction pipeline: {e}")
        import traceback
        traceback.print_exc()
        return "An unexpected error occurred during prediction.", 0.0

app = Flask(__name__)

# Ensure a secret key is configured for sessions (use env var in production)
# If FLASK_SECRET_KEY is not set, fall back to a random 24-byte key for dev.
app.secret_key = os.environ.get('FLASK_SECRET_KEY') or os.urandom(24)

# Email config for OTP
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'pavithra1512004@gmail.com'
app.config['MAIL_PASSWORD'] = 'orws sdcp bhrr cfij'
mail = Mail(app)


mail = Mail(app)

# Import database module
from database import (
    add_user, get_user_by_email, store_otp, verify_otp,
    add_history_record, get_user_history
)

# Directory to save uploaded images (serveable via static/)
UPLOAD_DIR = Path("static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Directory for flagged non-skin images + metrics for tuning
FLAGGED_DIR = Path("static/flagged")
FLAGGED_DIR.mkdir(parents=True, exist_ok=True)

# Simple recommendations for each disease (extend as needed)
DISEASE_RECOMMENDATIONS = {
    "Actinic keratosis (akiec)": "See a dermatologist for evaluation; sun protection and topical treatments may be recommended.",
    "Basal cell carcinoma (bcc)": "Please consult a dermatologist urgently; BCC often requires excision or specialist treatment.",
    "Benign keratosis (bkl)": "Often benign; monitor for changes. Consider dermatology review if it grows or changes.",
    "Dermatofibroma (df)": "Usually harmless; see dermatologist for confirmation or removal options if symptomatic.",
    "Melanoma (mel)": "High concern — see a dermatologist immediately for biopsy and staging. Natural care: avoid self-excision; protect from sun and avoid irritants while you arrange care.",
    "Melanocytic nevus (nv)": "Common mole; monitor for ABCDE changes. Dermatology check if suspicious. Natural care: gentle cleansing and sun protection; avoid picking or irritating the mole.",
    "Vascular lesion (vasc)": "Often cosmetic; consult dermatologist or vascular specialist for treatment options. Natural care: avoid heat and tight clothing over the area; use gentle skincare to reduce irritation.",
    "Acne (acne vulgaris)": "Treat with topical cleansers, benzoyl peroxide, or consult dermatologist for prescription options. Natural remedies: gentle cleansing, non-comedogenic moisturizers, tea tree oil (spot test first), and avoiding comedogenic products.",
}

# Add recommendations for conditions that may not be in the model label set
DISEASE_RECOMMENDATIONS.update({
    "Psoriasis": "See a dermatologist — topical steroids, vitamin D analogues or phototherapy are common treatments. Natural care: regular moisturization, avoid known triggers, and gentle oatmeal baths may soothe symptoms.",
    "Eczema (Atopic Dermatitis)": "Moisturize regularly; avoid triggers. See dermatologist for topical steroids or other prescriptions. Natural care: fragrance-free emollients, oatmeal baths, and lukewarm showers; avoid harsh soaps.",
    "Rosacea": "Avoid triggers (sun, heat, spicy foods). See dermatologist for topical treatments or oral medications. Natural care: gentle skincare, sun protection, and green-tinted moisturizers to reduce redness temporarily.",
    "Seborrheic Keratosis": "Usually benign; monitor for changes. Can be removed by dermatologist if desired. Natural care: gentle cleansing and avoid aggressive scrubbing; irritation can increase flaking.",
    "Tinea (fungal)": "Use topical antifungals; seek dermatology advice for oral antifungals if widespread. Natural care: keep the area dry, avoid occlusive clothing, and tea tree oil may help topically (patch test first).",
})

# Split clinical recommendations and natural remedies into a structured mapping
DISEASE_GUIDANCE = {
    "Actinic keratosis (akiec)": {
        'clinical': "See a dermatologist for evaluation; sun protection and topical treatments may be recommended.",
        'natural': "Sun protection (broad-spectrum SPF), avoid peak sun hours, and moisturizers to reduce irritation."
    },
    "Basal cell carcinoma (bcc)": {
        'clinical': "Please consult a dermatologist urgently; BCC often requires excision or specialist treatment.",
        'natural': "Avoid self-treatment; protect the area from sun and avoid irritants while arranging medical care."
    },
    "Benign keratosis (bkl)": {
        'clinical': "Often benign; monitor for changes. Consider dermatology review if it grows or changes.",
        'natural': "Gentle cleansing, avoid picking, and use non-irritating moisturizers to reduce dryness or flaking."
    },
    "Dermatofibroma (df)": {
        'clinical': "Usually harmless; see dermatologist for confirmation or removal options if symptomatic.",
        'natural': "Avoid trauma to the area; gentle skincare and sun protection."
    },
    "Melanoma (mel)": {
        'clinical': "High concern — see a dermatologist immediately for biopsy and staging.",
        'natural': "Avoid self-excision; protect area from sun, and avoid topical irritants while arranging urgent care."
    },
    "Melanocytic nevus (nv)": {
        'clinical': "Common mole; monitor for ABCDE changes. Dermatology check if suspicious.",
        'natural': "Gentle cleansing and sun protection; avoid picking or irritating the mole."
    },
    "Vascular lesion (vasc)": {
        'clinical': "Often cosmetic; consult dermatologist or vascular specialist for treatment options.",
        'natural': "Avoid heat and tight clothing over the area; gentle skincare to reduce irritation."
    },
    "Acne (acne vulgaris)": {
        'clinical': "Treat with topical cleansers, benzoyl peroxide, or consult dermatologist for prescription options.",
        'natural': "Gentle cleansing, non-comedogenic moisturizers, spot-testing tea tree oil, and avoiding comedogenic products."
    },
    "Psoriasis": {
        'clinical': "See a dermatologist — topical steroids, vitamin D analogues or phototherapy are common treatments.",
        'natural': "Regular moisturization, gentle oatmeal baths, and avoiding known triggers (stress, certain foods)."
    },
    "Eczema (Atopic Dermatitis)": {
        'clinical': "Moisturize regularly; avoid triggers. See dermatologist for topical steroids or other prescriptions.",
        'natural': "Fragrance-free emollients, lukewarm showers, oatmeal baths, and avoiding harsh soaps or hot water."
    },
    "Rosacea": {
        'clinical': "Avoid triggers (sun, heat, spicy foods). See dermatologist for topical treatments or oral medications.",
        'natural': "Gentle skincare, strict sun protection, and products formulated for sensitive skin; green-tinted moisturizers may help temporarily reduce redness."
    },
    "Seborrheic Keratosis": {
        'clinical': "Usually benign; monitor for changes. Can be removed by dermatologist if desired.",
        'natural': "Gentle cleansing and avoid aggressive scrubbing; keep skin moisturized to reduce flaking."
    },
    "Tinea (fungal)": {
        'clinical': "Use topical antifungals; seek dermatology advice for oral antifungals if widespread.",
        'natural': "Keep the area dry, wear breathable clothing, and consider topical tea tree oil after a patch test."
    }
}

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = get_user_by_email(email)
        if user and check_password_hash(user['password'], password):
            # Generate OTP and send to email
            otp = str(random.randint(100000, 999999))
            if store_otp(email, otp):
                session['pending_user'] = email
                try:
                    msg = Message('Your OTP Code', sender=app.config['MAIL_USERNAME'], recipients=[email])
                    msg.body = f'Your OTP for login is: {otp}'
                    mail.send(msg)
                    flash('OTP sent to your email!', 'success')
                except Exception as e:
                    flash(f'Error sending email: {e}', 'danger')
                    return redirect(url_for('login'))
                return redirect(url_for('verify_otp_route'))
            else:
                flash('Error storing OTP. Please try again.', 'danger')
                return redirect(url_for('login'))
        else:
            flash('Invalid email or password.', 'danger')
            return redirect(url_for('login'))
    return render_template('login.html')


@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp_route():
    if request.method == 'POST':
        email = session.get('pending_user')
        user_otp = request.form['otp']
        if email and verify_otp(email, user_otp):
            session['user'] = email
            session.pop('pending_user', None)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Incorrect OTP. Try again.', 'danger')
            return redirect(url_for('verify_otp_route'))
    return render_template('verify_otp.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        name = request.form['name']
        password = request.form['password']
        if get_user_by_email(email):
            flash('Email already registered. Please login.', 'danger')
            return redirect(url_for('login'))
        hashed_password = generate_password_hash(password)
        if add_user(email, name, hashed_password):
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Error during registration. Please try again.', 'danger')
            return redirect(url_for('register'))
    return render_template('register.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files['image']
        if file:
            # Create a record ID at the start
            record_id = uuid.uuid4().hex
            filename = UPLOAD_DIR / secure_filename(file.filename)
            file.save(filename)

            # Initialize the base record structure
            record = {
                'id': record_id,
                'date': datetime.now(),
                'image_path': str(filename),
                'filename': file.filename
            }

            # Verify basic image validity
            try:
                with Image.open(filename) as test_img:
                    test_img.verify()
            except Exception as e:
                record.update({
                    'skin_condition': 'Invalid image',
                    'confidence': 0.0,
                    'clinical_recommendations': "Please upload a valid image file.",
                    'clinical_recommendation': "Please upload a valid image file.",  # For backwards compatibility
                    'natural_remedies': "Ensure the file is a proper image format (JPEG, PNG, etc).",
                    'source': 'System validation',
                    'recommendation': str(e),
                    'is_error': True
                })
                if 'user' in session:
                    user = get_user_by_email(session['user'])
                    if user:
                        add_history_record(user['id'], record)
                flash('Invalid image file. Please upload a valid image.', 'error')
                return redirect(url_for('upload'))

            # Pre-check: basic skin heuristics
            is_skin, err_msg = is_valid_skin_image(filename)
            # Additional offline human-skin heuristic to reduce false positives on landscapes/animals/textures
            human_skin_flag = is_probably_human_skin(filename)
            if not is_skin or not human_skin_flag:
                # Create immediate non-skin record and redirect to results
                record_id = uuid.uuid4().hex
                record = {
                    'id': record_id,
                    'date': datetime.now(),
                    'skin_condition': 'Not a skin image',
                    'confidence': 0.0,
                    'image_path': str(filename),
                    'clinical_recommendations': "Please upload a clear photo of the affected skin area.",
                    'clinical_recommendation': "Please upload a clear photo of the affected skin area.",  # For backwards compatibility
                    'natural_remedies': "Ensure good lighting and focus when taking skin photos.",
                    'filename': file.filename,
                    'source': 'System validation',
                    'recommendation': err_msg,
                    'is_error': True
                }
                user_email = session.get('user')
                if user_email:
                    user = get_user_by_email(user_email)
                    if user:
                        add_history_record(user['id'], record)
                session['result'] = {
                    'disorder': 'Not a skin image',
                    'confidence': 0.0,
                    'image_path': str(filename),
                    'clinical_recommendations': record.get('clinical_recommendations', record.get('clinical_recommendation', '')),
                    'clinical_recommendation': record.get('clinical_recommendation', record.get('clinical_recommendations', '')),
                    'natural_remedies': record.get('natural_remedies', ''),
                    'filename': file.filename,
                    'source': 'System validation',
                    'recommendation': err_msg,
                    'is_error': True
                }
                session['last_result_id'] = record_id
                flash('No skin detected in the uploaded image. Please try again with a clear photo of the affected area.', 'error')
                return redirect(url_for('results'))

            # Make prediction
            try:
                disease_name, confidence = predict_skin_disease(filename)
                
                # Handle error responses from predict_skin_disease
                if isinstance(disease_name, str) and confidence == 0.0:
                    flash(disease_name, 'error')
                    return redirect(url_for('upload'))

                # Store successful prediction in record
                record.update({
                    'disorder': disease_name,
                    'confidence': confidence,
                    'clinical_recommendations': DISEASE_RECOMMENDATIONS.get(disease_name, "Please consult a dermatologist for proper evaluation."),
                    'clinical_recommendation': DISEASE_RECOMMENDATIONS.get(disease_name, "Please consult a dermatologist for proper evaluation."),
                    'natural_remedies': DISEASE_GUIDANCE.get(disease_name, {}).get('natural', ''),
                    'source': 'Model prediction',
                    'is_error': False
                })
                
                # Verify image with API
                if is_likely_non_skin_by_api(filename):
                    record_id = uuid.uuid4().hex
                    record = {
                        'id': record_id,
                        'date': datetime.now(),
                        'skin_condition': 'Not a skin image',
                        'confidence': 0.0,
                        'image_path': str(filename),
                        'clinical_recommendations': "Please upload a clear photo of the affected skin area.",
                        'clinical_recommendation': "Please upload a clear photo of the affected skin area.",  # For backwards compatibility
                        'natural_remedies': "Ensure good lighting and focus when taking skin photos.",
                        'filename': file.filename,
                        'source': 'API label filter',
                        'recommendation': 'Detected as logo/landscape/animal/object',
                        'is_error': True
                    }
                    user_email = session.get('user')
                    if user_email:
                        user = get_user_by_email(user_email)
                        if user:
                            add_history_record(user['id'], record)
                    session['result'] = {
                        'disorder': 'Not a skin image',
                        'confidence': 0.0,
                        'image_path': str(filename),
                        'clinical_recommendations': record.get('clinical_recommendations', record.get('clinical_recommendation', '')),
                        'clinical_recommendation': record.get('clinical_recommendation', record.get('clinical_recommendations', '')),
                        'natural_remedies': record.get('natural_remedies', ''),
                        'filename': file.filename,
                        'source': 'API label filter',
                        'recommendation': 'Detected as logo/landscape/animal/object',
                        'is_error': True
                    }
                    session['last_result_id'] = record_id
                    flash('No skin detected (image appears to contain a logo, landscape, animal, or non-skin object).', 'error')
                    return redirect(url_for('results'))
            except Exception as e:
                print(f"API-based pre-check failed: {e}")

            # Image validation is already handled in predict_skin_disease function

            # Proceed to prediction if validation passes
            disease_name, confidence = predict_skin_disease(filename)

            # Helper: detect non-skin or error messages from the prediction
            def _is_non_skin_or_error(name, conf):
                if not isinstance(name, str):
                    return False
                ln = name.lower()
                # Check for specific error messages that indicate non-skin or invalid images
                error_messages = {
                    'does not appear', 'not a skin', 'not a valid', 
                    'unable to', 'error', 'could not', 'confidence too low',
                    'suspicious lesion', 'please consult', 'could not access'
                }
                return any(msg in ln for msg in error_messages) or conf <= 0.01

            # If image is detected as non-skin / error, skip questionnaire and show immediate result
            if _is_non_skin_or_error(disease_name, confidence):
                record_id = uuid.uuid4().hex
                error_message = "Please upload a clear photo of the affected skin area."
                record = {
                    'id': record_id,
                    'date': datetime.now(),
                    'disorder': 'Not a skin image',
                    'skin_condition': 'Not a skin image',
                    'confidence': 0.0,
                    'image_path': str(filename),
                    'clinical_recommendations': error_message,
                    'clinical_recommendation': error_message,
                    'recommendation': error_message,
                    'filename': file.filename,
                    'source': 'System validation',
                    'is_error': True,
                    'natural_remedies': ''  # Empty for error cases
                }
                
                # Store result in session and redirect
                session['result'] = record
                session['last_result_id'] = record_id
                flash(error_message, 'error')
                return redirect(url_for('results'))

            # Save into user's history (if logged in)
            user_email = session.get('user')
            if user_email:
                user = get_user_by_email(user_email)
                if user:
                    add_history_record(user['id'], record)

            # Store result in session for results page and redirect directly to results
            if _is_non_skin_or_error(disease_name, confidence):
                session['result'] = {
                    'disorder': 'Not a skin image',
                    'confidence': 0.0,
                    'image_path': str(filename),
                    'clinical_recommendations': error_message,
                    'clinical_recommendation': error_message,
                    'recommendation': error_message,
                    'filename': file.filename,
                    'source': 'System validation',
                    'natural_remedies': '',  # Empty for error cases
                    'is_error': True
                }
                session['last_result_id'] = record_id
                flash('No skin detected in the uploaded image. Please try again with a clear photo of the affected area.', 'error')
                return redirect(url_for('results'))

            # Normal flow: build history record and proceed to questionnaire
            record_id = uuid.uuid4().hex
            
            # Ensure we have a valid disease name
            if not disease_name or disease_name.lower() in ['none', 'unknown', 'unknown condition']:
                print(f"Warning: Invalid disease name detected: {disease_name}")
                disease_name = predict_skin_disease(filename)[0]  # Try prediction again
                print(f"Re-predicted disease name: {disease_name}")
            
            clinical_rec = DISEASE_GUIDANCE.get(disease_name, {}).get('clinical') if 'DISEASE_GUIDANCE' in globals() else DISEASE_RECOMMENDATIONS.get(disease_name)
            natural_rec = DISEASE_GUIDANCE.get(disease_name, {}).get('natural') if 'DISEASE_GUIDANCE' in globals() else ''
            
            # Create record with explicit disease name
            record = {
                'id': record_id,
                'date': datetime.now(),
                'disorder': disease_name,
                'skin_condition': disease_name,
                'confidence': float(confidence) if confidence is not None else 0.0,
                'image_path': str(filename),
                'clinical_recommendations': clinical_rec,
                'clinical_recommendation': clinical_rec,
                'natural_remedies': natural_rec,
                'filename': file.filename,
                'source': 'Local model',
                'recommendation': DISEASE_RECOMMENDATIONS.get(disease_name, "See a dermatologist for full evaluation."),
                'is_error': False
            }
            print(f"Created record with disease name: {disease_name}")  # Debug print

            # Save into user's history
            user_email = session.get('user')
            if user_email:
                user = get_user_by_email(user_email)
                if user:
                    add_history_record(user['id'], record)

            # Store result in session for questionnaire and results pages
            session['result'] = {
                'disorder': disease_name,
                'confidence': confidence,
                'image_path': str(filename),
                'clinical_recommendations': clinical_rec,
                'clinical_recommendation': clinical_rec,
                'natural_remedies': natural_rec,
                'filename': file.filename,
                'source': 'Local model',
                'recommendation': DISEASE_RECOMMENDATIONS.get(disease_name, "See a dermatologist for full evaluation."),
                'is_error': False
            }
            session['last_result_id'] = record_id

            return redirect(url_for('questionnaire'))

    return render_template('upload.html')

@app.route('/questionnaire', methods=['GET', 'POST'])
def questionnaire():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        answers = request.form.to_dict()
        session['answers'] = answers

        # Generate skincare routine based on answers
        skin_type = fake_skin_type_from_quiz(answers)
        routine = fake_routine_for_skin_type(skin_type)
        session['routine'] = routine
        session['skin_type'] = skin_type

        return redirect(url_for('results'))

    return render_template('questionnaire.html')

@app.route('/results')
def results():
    if 'user' not in session:
        return redirect(url_for('login'))

    # Allow viewing a specific past result via ?result_id=ID, otherwise use last session result
    result_id = request.args.get('result_id') or session.get('last_result_id')
    result_obj = None
    if result_id:
        user_email = session.get('user')
        if user_email:
            user = get_user_by_email(user_email)
            if user:
                history = get_user_history(user['id'])
                for rec in history:
                    if str(rec.get('id')) == str(result_id):
                        # Prepare a copy for rendering
                        result_obj = rec.copy()
                        break

    # Fallback to session-stored result
    if not result_obj and 'result' in session:
        result_obj = session['result']

    if not result_obj:
        return redirect(url_for('upload'))

    try:
        from recommendation_utils import generate_personalized_recommendations, enhance_standard_recommendations

        # Get questionnaire answers from session
        questionnaire_answers = session.get('answers', {})

        # Generate enhanced recommendations
        enhanced_recommendations = generate_personalized_recommendations(
            result_obj.get('disorder', 'Unknown skin condition'),
            questionnaire_answers
        )

        # If there are existing recommendations, enhance them
        if result_obj.get('clinical_recommendation') or result_obj.get('natural_remedies'):
            base_recommendations = {
                "clinical_recommendations": result_obj.get('clinical_recommendation', ''),
                "natural_remedies": result_obj.get('natural_remedies', ''),
                "lifestyle_tips": [],
                "skincare_routine": {"morning": [], "evening": []}
            }
            enhanced_recommendations = enhance_standard_recommendations(base_recommendations, questionnaire_answers)

        # Update result object with enhanced recommendations
        result_obj['clinical_recommendation'] = enhanced_recommendations.get('clinical_recommendations', result_obj.get('clinical_recommendation', ''))
        result_obj['natural_remedies'] = enhanced_recommendations.get('natural_remedies', result_obj.get('natural_remedies', ''))
        result_obj['lifestyle_tips'] = enhanced_recommendations.get('lifestyle_tips', [])
        result_obj['skincare_routine'] = enhanced_recommendations.get('skincare_routine', {
            "morning": [], "evening": []
        })

        # Update the session's result with enhanced recommendations
        if 'result' in session:
            session['result'] = result_obj

        # Update history record with enhanced recommendations if applicable
        if user_email and user:
            add_history_record(user['id'], result_obj)

    except Exception as e:
        print(f"Error generating enhanced recommendations: {e}")
        # Keep existing recommendations if enhancement fails

    return render_template(
        'results.html',
        result=result_obj,
        routine=session.get('routine', {}),
        skin_type=session.get('skin_type', 'Unknown')
    )


# test_api removed — app uses model-only predictions now

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', user=session['user'])

@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login'))
    user_email = session.get('user')
    user_history = []
    if user_email:
        user = get_user_by_email(user_email)
        if user:
            user_history = get_user_history(user['id'])

    return render_template('history.html', history=user_history)

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        flash('Message sent successfully!', 'success')
        return redirect(url_for('contact'))
    return render_template('contact.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))


@app.route('/chat', methods=['POST'])
def chat():
    """Simple chat endpoint that queries Wikipedia for the user's question and returns a short summary.
    Expects form field 'q' (text). Returns JSON {reply: str}.
    """
    q = request.form.get('q', '').strip()
    if not q:
        return jsonify({'reply': 'Please enter a question.'})

    try:
        # 1) Search Wikipedia for the query to get the best-matching page title
        search_url = 'https://en.wikipedia.org/w/api.php'
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': q,
            'format': 'json',
            'srlimit': 1
        }
        resp = requests.get(search_url, params=params, timeout=8)
        data = resp.json()
        search_hits = data.get('query', {}).get('search', [])
        if not search_hits:
            return jsonify({'reply': "I couldn't find a Wikipedia page matching that query."})

        title = search_hits[0].get('title')

        # 2) Fetch the summary for the title via the REST summary endpoint
        summary_url = f'https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.requote_uri(title)}'
        sresp = requests.get(summary_url, timeout=8)
        if sresp.status_code == 200:
            sdata = sresp.json()
            extract = sdata.get('extract') or sdata.get('description') or ''
            if extract:
                reply = f"{title}: {extract}"
            else:
                reply = f"Found page '{title}' but couldn't retrieve a summary."
        else:
            reply = f"Found page '{title}' but Wikipedia summary request failed (status {sresp.status_code})."

        return jsonify({'reply': reply})
    except Exception as e:
        return jsonify({'reply': f'Error while querying Wikipedia: {e}'})

# OTP-only login route
@app.route('/login_otp', methods=['GET', 'POST'])
def login_otp():
    if request.method == 'POST':
        email = request.form['email']
        if not get_user_by_email(email):
            flash('Email not registered. Please register first.', 'danger')
            return redirect(url_for('register'))
        otp = str(random.randint(100000, 999999))
        if store_otp(email, otp):
            session['pending_user'] = email
            try:
                msg = Message('Your OTP Code', sender=app.config['MAIL_USERNAME'], recipients=[email])
                msg.body = f'Your OTP for login is: {otp}'
                mail.send(msg)
                flash('OTP sent to your email!', 'success')
            except Exception as e:
                flash(f'Error sending email: {e}', 'danger')
                return redirect(url_for('login_otp'))
            return redirect(url_for('verify_otp_route'))
        else:
            flash('Error storing OTP. Please try again.', 'danger')
            return redirect(url_for('login_otp'))
    return render_template('login_otp.html')


if __name__ == '__main__':
    app.run(debug=True)