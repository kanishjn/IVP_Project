# predict_one.py
import joblib
import numpy as np
from preprocess_and_extract import load_fits, background_subtract, segment_object, extract_features_from_image

# Load trained model
model_data = joblib.load("svm_pipeline_adaptive.joblib")
pipeline = model_data["pipeline"]
le = model_data["label_encoder"]

# Path to one test image
fname = "sdss_data/images/GALAXY_1.fits"

# Preprocess + extract features
img = load_fits(fname)
img_bs = background_subtract(img, boxsize=40)
mask = segment_object(img_bs)
feats = extract_features_from_image(img_bs, mask).reshape(1, -1)

# Predict
pred = pipeline.predict(feats)
label = le.inverse_transform(pred)[0]
print(f"Predicted class for {fname}: {label}")
