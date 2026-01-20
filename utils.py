from PIL import Image
from pathlib import Path
import random

# Directory for uploaded images
UPLOAD_DIR = Path("instance/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --- STUBS for testing (replace with real models later) ---

SKIN_TYPES = ["Normal", "Oily", "Dry", "Combination", "Sensitive"]

DISORDERS = [
    "Acne (acne vulgaris)",
    "Melanocytic nevus (nv)",
    "Melanoma (mel)",
    "Basal cell carcinoma (bcc)",
    "Actinic keratosis (akiec)",
    "Benign keratosis (bkl)",
    "Vascular lesion (vasc)",
    "Dermatofibroma (df)",
]

# --- Functions --- #

def fake_skin_type_from_quiz(answers: dict) -> str:
    """
    Use quiz answers to determine skin type more accurately based on multiple factors.
    Handles both numeric (0-4) and text-based answers.
    """
    if not answers:
        return "Combination"

    def convert_to_score(value):
        """Convert various answer formats to a 0-4 score"""
        if isinstance(value, (int, float)):
            return int(value)
        if not value:
            return 0
        
        # Handle text-based answers
        value = str(value).lower().strip()
        text_to_score = {
            # Frequency/Amount mappings
            "never": 0, "rarely": 1, "sometimes": 2, "often": 3, "always": 4,
            "none": 0, "little": 1, "medium": 2, "high": 3, "very high": 4,
            "very low": 0, "low": 1, "moderate": 2, "severe": 4,
            # Direct number strings
            "0": 0, "1": 1, "2": 2, "3": 3, "4": 4
        }
        return text_to_score.get(value, 2)  # Default to medium/moderate (2) if unknown

    # Convert answers to scores
    scores = {
        "oily": convert_to_score(answers.get("oily_shine", "0")),  # 0-4
        "dry": convert_to_score(answers.get("dry_patches", "0")),  # 0-4
        "sensitive": convert_to_score(answers.get("sensitivity", "0")),  # 0-4
        "pores": convert_to_score(answers.get("pore_size", "0")),  # 0-4
        "acne": convert_to_score(answers.get("acne_frequency", "0"))  # 0-4
    }

    # Additional questions (if available)
    skin_feel = answers.get("skin_feel", "").lower()  # tight, normal, oily
    breakout_frequency = answers.get("breakout_frequency", "").lower()  # often, sometimes, rarely
    reaction_to_products = answers.get("reaction_to_products", "").lower()  # easily, sometimes, rarely

    # Weighted scoring system
    type_scores = {
        "Oily": 0,
        "Dry": 0,
        "Sensitive": 0,
        "Combination": 0,
        "Normal": 0
    }

    # Core scores from numerical values
    type_scores["Oily"] += scores["oily"] * 1.5 + scores["pores"] * 0.8 + scores["acne"] * 0.5
    type_scores["Dry"] += scores["dry"] * 1.5 + (4 - scores["oily"]) * 0.5
    type_scores["Sensitive"] += scores["sensitive"] * 2.0
    type_scores["Combination"] += (scores["oily"] + scores["dry"]) * 0.7

    # Additional factor adjustments
    if skin_feel == "tight":
        type_scores["Dry"] += 2
    elif skin_feel == "oily":
        type_scores["Oily"] += 2
    elif skin_feel == "normal":
        type_scores["Normal"] += 2

    if breakout_frequency == "often":
        type_scores["Oily"] += 1.5
    elif breakout_frequency == "rarely":
        type_scores["Normal"] += 1.5

    if reaction_to_products == "easily":
        type_scores["Sensitive"] += 2
    elif reaction_to_products == "rarely":
        type_scores["Normal"] += 1

    # Calculate normal score based on balanced indicators
    type_scores["Normal"] += (
        (4 - abs(2 - scores["oily"])) * 0.5 +  # More normal if oiliness is moderate
        (4 - abs(2 - scores["dry"])) * 0.5 +   # More normal if dryness is moderate
        (4 - scores["sensitive"]) * 0.3 +      # More normal if less sensitive
        (4 - scores["pores"]) * 0.3            # More normal if pores are moderate
    )

    # Special case for combination skin
    if abs(scores["oily"] - scores["dry"]) <= 1 and max(scores["oily"], scores["dry"]) >= 2:
        type_scores["Combination"] += 3

    # Return the skin type with highest score
    return max(type_scores.items(), key=lambda x: x[1])[0]


def fake_routine_for_skin_type(skin_type: str) -> dict:
    """
    Return a personalized skincare routine based on skin type.
    """
    routines = {
        "Oily": {
            "morning": [
                "Oil-free foaming cleanser",
                "Alcohol-free toner with BHA",
                "Lightweight oil-free moisturizer",
                "Oil-free sunscreen SPF 50"
            ],
            "evening": [
                "Double cleanse (oil cleanser followed by foam cleanser)",
                "Salicylic acid treatment",
                "Niacinamide serum",
                "Light gel moisturizer"
            ]
        },
        "Dry": {
            "morning": [
                "Cream cleanser",
                "Hydrating toner",
                "Hyaluronic acid serum",
                "Rich moisturizer",
                "Moisturizing sunscreen SPF 50"
            ],
            "evening": [
                "Oil-based cleanser",
                "Hydrating essence",
                "Ceramide serum",
                "Rich night cream",
                "Face oil"
            ]
        },
        "Sensitive": {
            "morning": [
                "Fragrance-free gentle cleanser",
                "Calming toner (alcohol-free)",
                "Centella asiatica serum",
                "Soothing moisturizer",
                "Mineral sunscreen SPF 50"
            ],
            "evening": [
                "Micellar water",
                "Gentle cream cleanser",
                "Calming peptide serum",
                "Barrier repair cream"
            ]
        },
        "Combination": {
            "morning": [
                "Balanced pH cleanser",
                "Hydrating toner (alcohol-free)",
                "Light moisturizer for T-zone",
                "Rich moisturizer for dry areas",
                "Mattifying sunscreen SPF 50"
            ],
            "evening": [
                "Balancing cleanser",
                "BHA for T-zone only",
                "Hydrating serum for dry areas",
                "Zone-specific moisturizer"
            ]
        },
        "Normal": {
            "morning": [
                "Gentle foaming cleanser",
                "Hydrating toner",
                "Antioxidant serum",
                "Light moisturizer",
                "Sunscreen SPF 50"
            ],
            "evening": [
                "Cream cleanser",
                "Retinol serum (2-3 times/week)",
                "Peptide serum",
                "Night moisturizer"
            ]
        }
    }
    
    # Default routine if skin type not found
    default_routine = {
        "morning": [
            "Gentle cleanser",
            "Moisturizer",
            "Sunscreen SPF 50"
        ],
        "evening": [
            "Cleanser",
            "Moisturizer"
        ]
    }
    
    return routines.get(skin_type, default_routine)


def fake_skin_disorder_classifier(image_path: Path) -> tuple[str, float]:
    """
    Dummy image classifier: returns a random skin disorder and confidence.
    """
    # Ensure valid image
    with Image.open(image_path) as im:
        im.convert("RGB")

    prediction = random.choice(DISORDERS)
    confidence = round(random.uniform(0.62, 0.94), 2)
    return prediction, confidence
