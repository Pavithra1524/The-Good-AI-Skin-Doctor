import json

# Comprehensive skincare recommendations database
CONDITION_RECOMMENDATIONS = {
    "Actinic keratosis (akiec)": {
        "clinical_recommendations": "Consult a dermatologist for proper evaluation and treatment. Common treatments include cryotherapy or topical medications.",
        "natural_remedies": "Use natural sunscreens, green tea extracts, and aloe vera. Always maintain strict sun protection.",
        "lifestyle_tips": [
            "Avoid sun exposure between 10 AM and 4 PM",
            "Wear protective clothing and wide-brimmed hats",
            "Use broad-spectrum sunscreen SPF 30+ daily",
            "Perform regular skin self-examinations"
        ]
    },
    "Basal cell carcinoma (bcc)": {
        "clinical_recommendations": "Immediate dermatologist consultation required. May require surgical removal or other specialized treatments.",
        "natural_remedies": "While awaiting treatment, use gentle skincare and sun protection. No natural cure exists for BCC.",
        "lifestyle_tips": [
            "Complete sun protection is essential",
            "Regular skin checks",
            "Maintain a healthy immune system",
            "Avoid tanning beds"
        ]
    },
    "Benign keratosis (bkl)": {
        "clinical_recommendations": "Usually harmless but monitor for changes. Dermatologist can remove if desired.",
        "natural_remedies": "Apple cider vinegar diluted applications, tea tree oil, or aloe vera may help with appearance.",
        "lifestyle_tips": [
            "Gentle exfoliation",
            "Regular moisturizing",
            "Sun protection",
            "Monitor for changes"
        ]
    },
    "Melanoma (mel)": {
        "clinical_recommendations": "URGENT dermatologist consultation required. Early detection and treatment are crucial.",
        "natural_remedies": "No natural remedies - immediate medical attention required.",
        "lifestyle_tips": [
            "Monthly self-skin examinations",
            "Regular dermatologist visits",
            "Complete sun protection",
            "Early morning or late evening outdoor activities"
        ]
    },
    "Melanocytic nevus (nv)": {
        "clinical_recommendations": "Regular monitoring using ABCDE rule. Annual dermatologist check recommended.",
        "natural_remedies": "Use natural sunscreens and gentle skincare products. Monitor for changes.",
        "lifestyle_tips": [
            "Regular self-checks using ABCDE rule",
            "Photo documentation of moles",
            "Sun protection",
            "Healthy diet rich in antioxidants"
        ]
    },
    "Vascular lesion (vasc)": {
        "clinical_recommendations": "Dermatologist evaluation recommended. Laser treatment may be an option.",
        "natural_remedies": "Cool compresses, witch hazel, horse chestnut extract may help appearance.",
        "lifestyle_tips": [
            "Avoid hot showers/baths",
            "Exercise regularly",
            "Elevate affected areas when possible",
            "Avoid tight clothing"
        ]
    },
    "Acne (acne vulgaris)": {
        "clinical_recommendations": "Start with over-the-counter treatments containing benzoyl peroxide or salicylic acid.",
        "natural_remedies": "Tea tree oil, green tea extracts, aloe vera, honey masks",
        "lifestyle_tips": [
            "Gentle cleansing twice daily",
            "Avoid touching face frequently",
            "Change pillowcase regularly",
            "Maintain healthy diet"
        ]
    }
}

SKIN_TYPE_ROUTINES = {
    "oily": {
        "morning": [
            "Gentle foaming cleanser",
            "Oil-free toner",
            "Light moisturizer",
            "Oil-free sunscreen"
        ],
        "evening": [
            "Oil-control cleanser",
            "Salicylic acid treatment",
            "Light gel moisturizer"
        ]
    },
    "dry": {
        "morning": [
            "Creamy gentle cleanser",
            "Hydrating toner",
            "Rich moisturizer",
            "Moisturizing sunscreen"
        ],
        "evening": [
            "Gentle cream cleanser",
            "Hydrating serum",
            "Rich night cream"
        ]
    },
    "combination": {
        "morning": [
            "Balanced pH cleanser",
            "Alcohol-free toner",
            "Zone-appropriate moisturizer",
            "Light sunscreen"
        ],
        "evening": [
            "Gentle cleanser",
            "Treatment for specific zones",
            "Balanced moisturizer"
        ]
    },
    "sensitive": {
        "morning": [
            "Fragrance-free gentle cleanser",
            "Calming moisturizer",
            "Mineral sunscreen"
        ],
        "evening": [
            "Gentle cleanser",
            "Calming serum",
            "Gentle moisturizer"
        ]
    }
}

def determine_skin_type(questionnaire_data):
    """Determine skin type based on questionnaire answers"""
    tzone = questionnaire_data.get('tzone', 'medium').lower()
    sensitivity = questionnaire_data.get('sensitivity', 'low').lower()
    
    if tzone == 'high' and sensitivity in ['low', 'medium']:
        return 'oily'
    elif tzone == 'low' and sensitivity == 'high':
        return 'dry'
    elif tzone == 'medium' and sensitivity == 'high':
        return 'sensitive'
    elif tzone == 'varied':
        return 'combination'
    return 'combination'  # default type

def generate_skincare_recommendations(skin_condition, questionnaire_data):
    """
    Generate personalized skincare recommendations based on skin condition and questionnaire data
    using our local database.
    """
    # Determine skin type
    skin_type = determine_skin_type(questionnaire_data)
    
    # Get base condition recommendations
    condition_recs = CONDITION_RECOMMENDATIONS.get(skin_condition, {
        "clinical_recommendations": "Please consult a dermatologist for proper evaluation.",
        "natural_remedies": "Use gentle, fragrance-free products while awaiting professional advice.",
        "lifestyle_tips": [
            "Maintain good hygiene",
            "Protect skin from sun exposure",
            "Stay hydrated",
            "Get adequate sleep"
        ]
    })
    
    # Get routine based on skin type
    routine = SKIN_TYPE_ROUTINES.get(skin_type, SKIN_TYPE_ROUTINES['combination'])
    
    # Return combined recommendations
    return {
        "skin_type": skin_type,
        "routine": {
            "AM": routine['morning'],
            "PM": routine['evening']
        },
        "natural_remedies": condition_recs['natural_remedies'],
        "avoid": [
            "Harsh soaps and cleansers",
            "Excessive sun exposure",
            "Hot water when washing",
            "Touching face frequently"
        ],
        "lifestyle_tips": condition_recs['lifestyle_tips']
    }

def generate_personalized_recommendations(condition: str, questionnaire_answers: dict) -> dict:
    """
    Generate personalized recommendations based on:
    1. Detected skin condition
    2. Questionnaire answers about skin type, concerns, and current routine
    """
    # Determine skin type
    skin_type = determine_skin_type(questionnaire_answers)
    
    # Get base recommendations for the condition
    condition_recs = CONDITION_RECOMMENDATIONS.get(condition, {
        "clinical_recommendations": "Please consult a dermatologist for proper evaluation.",
        "natural_remedies": "Use gentle, fragrance-free products while awaiting professional advice.",
        "lifestyle_tips": [
            "Maintain good hygiene",
            "Protect skin from sun exposure",
            "Stay hydrated",
            "Get adequate sleep"
        ]
    })
    
    # Get skincare routine based on skin type
    routine = SKIN_TYPE_ROUTINES.get(skin_type, SKIN_TYPE_ROUTINES['combination'])
    
    # Add additional tips based on questionnaire answers
    additional_tips = []
    if questionnaire_answers.get('sun_exposure', '').lower() == 'high':
        additional_tips.extend([
            "Use SPF 50+ sunscreen",
            "Reapply sunscreen every 2 hours when outdoors",
            "Seek shade during peak sun hours"
        ])
    
    if questionnaire_answers.get('stress', '').lower() == 'high':
        additional_tips.extend([
            "Practice stress management techniques",
            "Consider meditation or yoga",
            "Get adequate sleep"
        ])
    
    if questionnaire_answers.get('allergies', '').lower() == 'yes':
        additional_tips.append("Use hypoallergenic products")
        
    lifestyle_tips = condition_recs['lifestyle_tips'] + additional_tips
    
    return {
        "clinical_recommendations": condition_recs['clinical_recommendations'],
        "natural_remedies": condition_recs['natural_remedies'],
        "lifestyle_tips": lifestyle_tips,
        "skincare_routine": {
            "morning": routine['morning'],
            "evening": routine['evening']
        }
    }

def enhance_standard_recommendations(base_recommendations: dict, questionnaire_answers: dict) -> dict:
    """
    Enhance the standard recommendations by incorporating questionnaire answers
    to make them more personalized.
    """
    enhanced = base_recommendations.copy()
    
    # Add skin-type specific modifications
    skin_type = determine_skin_type(questionnaire_answers)
    routine = SKIN_TYPE_ROUTINES.get(skin_type, SKIN_TYPE_ROUTINES['combination'])
    
    # Update skincare routine
    enhanced['skincare_routine'] = {
        "morning": routine['morning'],
        "evening": routine['evening']
    }
    
    # Add sensitivity considerations
    if questionnaire_answers.get('sensitivity', '').lower() == 'high':
        enhanced['natural_remedies'] = "Focus on gentle, fragrance-free products. " + enhanced.get('natural_remedies', '')
        if 'lifestyle_tips' not in enhanced:
            enhanced['lifestyle_tips'] = []
        enhanced['lifestyle_tips'].extend([
            "Avoid harsh skincare ingredients",
            "Patch test new products",
            "Use lukewarm water instead of hot"
        ])
    
    return enhanced