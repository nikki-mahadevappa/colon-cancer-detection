# --------- IMPORTANT: Force CPU (fixes slowness on Windows) ----------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

# ---------------- Load Model ONCE ----------------
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model("model/colon_cancer_cnn.h5")

model = load_cnn_model()

# ---------------- Page Config ----------------
st.set_page_config(page_title="Colon Cancer Detection System")

st.title("Colon Cancer Detection System")
st.write("Real-time colon cancer detection using Deep Learning and clinical indicators")

# ---------------- Form (prevents rerun lag) ----------------
with st.form("prediction_form"):
    uploaded_image = st.file_uploader(
        "Upload Colon Histopathology Image",
        type=["jpg", "png", "tif"]
    )

    st.subheader("Clinical Inputs")
    age = st.number_input("Age", 0, 120, 45)
    gene1 = st.number_input("Genetic Marker 1 (0 = Absent, 1 = Present)", 0, 1, 0)
    gene2 = st.number_input("Genetic Marker 2 (0 = Absent, 1 = Present)", 0, 1, 0)

    submitted = st.form_submit_button("Predict")

# ---------------- Prediction Logic ----------------
if submitted:
    if uploaded_image is None:
        st.warning("Please upload an image.")
    else:
        # -------- Image Processing --------
        image_bytes = uploaded_image.read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), 1)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        image_resized = cv2.resize(image, (224, 224))
        image_normalized = image_resized / 255.0
        image_input = np.expand_dims(image_normalized, axis=0)

        # -------- CNN Prediction --------
        prediction = model.predict(image_input, verbose=0)[0][0]

        if prediction > 0.5:
            cancer_status = "Cancer Detected"
        else:
            cancer_status = "No Cancer Detected"

        # -------- Age + Genetic Risk Logic --------
        genetic_score = gene1 + gene2

        age_risk = 0
        if age >= 60:
            age_risk = 2
        elif age >= 45:
            age_risk = 1

        total_risk_score = genetic_score + age_risk

        if total_risk_score <= 1:
            risk_level = "Low Risk"
        elif total_risk_score == 2:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"

        # -------- Output --------
        st.success(f"Cancer Status: {cancer_status}")
        st.info(f"Risk Level: {risk_level}")
        st.write(f"Model Confidence Score: {prediction:.2f}")

        st.caption(
            "Note: Image-based detection is independent of age. "
            "Age and genetic indicators are used only for risk estimation."
        )