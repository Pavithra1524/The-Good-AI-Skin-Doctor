from tensorflow import keras
import numpy as np
from PIL import Image
import sys

MODEL_PATH = 'skin_disease_classifier_v1_final.h5'

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python tools/diagnose_model.py <image_path>')
        sys.exit(1)
    img_path = sys.argv[1]
    try:
        model = keras.models.load_model(MODEL_PATH, compile=False)
        print('Model loaded')
        try:
            print('Model output shape:', model.output_shape)
        except Exception:
            pass
        img = Image.open(img_path).convert('RGB').resize((224,224))
        arr = np.array(img, dtype=np.float32)/255.0
        preds = model.predict(np.expand_dims(arr,0))[0]
        print('Raw preds:', preds)
        top = np.argsort(preds)[::-1][:5]
        for i in top:
            print(i, preds[i])
    except Exception as e:
        print('Error loading model or predicting:', e)
