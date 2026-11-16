import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Insurance Cost Estimator",
    layout="wide"
)

MODEL_PATH = "insurance_model.pkl"
FEATURE_PATH = "model_features.pkl"

REF_AVG_NON_SMOKER = 8400
REF_AVG_SMOKER = 32000

# Set to True only if your model was trained on np.log1p(charges)
USE_LOG_TARGET = True


@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_PATH)
    return model, list(feature_cols)


def bmi_category(bmi):
    if bmi < 18.5:
        return "underweight"
    elif bmi < 25:
        return "normal"
    elif bmi < 30:
        return "overweight"
    else:
        return "obese"


def build_input_row(age, sex, bmi, children, smoker, region, feature_cols):
    sex = sex.lower()
    smoker = smoker.lower()
    region = region.lower()
    category = bmi_category(bmi)

    # Keys are lowercase; they get matched to the model's columns ignoring case
    values = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": 1 if sex == "male" else 0,
        "smoker_yes": 1 if smoker == "yes" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
        "bmi_category_obese": 1 if category == "obese" else 0,
        "bmi_category_overweight": 1 if category == "overweight" else 0,
        "bmi_category_underweight": 1 if category == "underweight" else 0,
    }

    # Build the row in the exact column names/order the model expects
    row = {col: values.get(col.lower(), 0) for col in feature_cols}
    return pd.DataFrame([row], columns=feature_cols)


model, feature_cols = load_artifacts()

st.title("🩺 Insurance Cost Estimator")

if model is None:
    st.error("Model files not found. Keep insurance_model.pkl and model_features.pkl in the same folder as this app.")
    st.stop()

left, right = st.columns([1, 1.2], gap="large")

with left:
    st.subheader("Patient Details")

    age = st.slider("AGE", 18, 64, 30)
    sex = st.radio("SEX", ["Male", "Female"], horizontal=True)
    bmi = st.slider("BMI", 15.0, 55.0, 26.0, 0.1)
    children = st.slider("CHILDREN", 0, 5, 0)
    smoker = st.radio("SMOKER", ["No", "Yes"], horizontal=True)
    region = st.selectbox("REGION", ["Northeast", "Northwest", "Southeast", "Southwest"])

    predict_clicked = st.button("Estimate charges", type="primary", use_container_width=True)

with right:
    st.subheader("Estimated charges")

    if predict_clicked:
        try:
            input_row = build_input_row(age, sex, bmi, children, smoker, region, feature_cols)
            pred = model.predict(input_row)[0]
            pred_charges = float(np.expm1(pred)) if USE_LOG_TARGET else float(pred)

            st.metric("Predicted annual charge", f"${pred_charges:,.2f}")

            st.divider()

            compare_df = pd.DataFrame({
                "Group": ["Your prediction", "Non-smoker avg.", "Smoker avg."],
                "Charges": [pred_charges, REF_AVG_NON_SMOKER, REF_AVG_SMOKER]
            }).set_index("Group")

            st.bar_chart(compare_df)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.write("Model expects these features:", feature_cols)
    else:
        st.info("Fill in the details and click **Estimate charges**.")

# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib 
# import os
# import warnings

# warnings.filterwarnings("ignore")

# st.set_page_config(
#     page_title = "Insurance Cost Estimator",
#     layout = "wide"
# )

# MODEL_PATH = "insurance_model.pkl"
# FEATURE_PATH = "model_features.pkl"

# REF_AVG_NON_SMOKER = 8400
# REF_AVG_SMOKER = 32000

# @st.cache_resource
# def load_artifacts():
#     if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_PATH):
#         return None, None
#     model = joblib.load(MODEL_PATH)
#     feature_cols = joblib.load(FEATURE_PATH)

#     return model, feature_cols

# def bmi_category(bmi):
#     if bmi < 18.5:
#         return "Underweight"
#     elif bmi < 25:
#         return "Normal"
#     elif bmi < 30:
#         return "Overweight"
#     else:
#         return "Obese"

# def build_input_row(age ,sex, bmi, children, smoker, region, feature_cols):
#     category = bmi_category(bmi)

#     row = {
#         "Age": age,
#         "Bmi": bmi,
#         "Children": children,
#         "Sex_Male": 1 if sex == "male" else 0 ,
#         "Smoker_Yes": 1 if smoker == "yes" else 0 ,
#         "Region_northwest": 1 if region == "northwest" else 0 ,
#         "Region_southeast": 1 if region == "southeast" else 0 ,
#         "Region_southwest": 1 if region == "southwest" else 0 ,
#         "Bmi_Category_obese": 1 if category == "obese" else 0 ,
#         "Bmi_Category_overweight": 1 if category == "overweight" else 0 ,
#         "Bmi_Category_underweight": 1 if category == "underweight" else 0 ,
#     }

#     encoded = pd.DataFrame([row])

#     return encoded.reindex(columns = feature_cols, fill_value = 0)

# model , feature_cols = load_artifacts()

# st.title("🩺 Insurance Cost Estimator")

# if model is None:
#     st.error("Model files not found.")
#     st.stop()

# left , right = st.columns([1,1.2], gap = "large")

# with left:
#     st.subheader("Patient Details")

#     age = st.slider("AGE", 18, 64 , 30)
#     sex = st.radio("SEX", ["Male","Female"], horizontal = True)
#     bmi = st.slider("BMI", 15.0, 55.0 , 26.0, 0.1)
#     children = st.slider("CHILDREN", 0, 5 , 0)
#     smoker = st.radio("SMOKER",["No","Yes"],horizontal = True)
#     region = st.selectbox ("REGION" , ["Norteast", "Northwest", "Southeast", "Southwest"])

#     predict_clicked = st.button("Estimate charges", type = "primary", use_container_width = True)
    
# with right:
#     st.subheader("Estimated charges")

#     if predict_clicked:
#         input_row = build_input_row(age , sex, bmi, children , smoker , region, feature_cols)
#         log_pred = model.predict(input_row)[0]
#         pred_charges = np.expm1(log_pred)

#         st.metric("Predicted annual charge", f"₹{pred_charges:,.2f}")

#         st.divider()

#         compare_df = pd.DataFrame({
#             "Group": ["Your prediction", "Non-smoker avg.", "Smoker avg"],
#             "Charges": [pred_charges, REF_AVG_NON_SMOKER, REF_AVG_SMOKER]
#         }).set_index("Group")

#         st.bar_chart(compare_df)

#     else:
#         st.info("Fill in the details and click **Estimate charges**.")