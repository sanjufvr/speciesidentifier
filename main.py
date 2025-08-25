# main.py
import streamlit as st
import pandas as pd
from pathlib import Path
import joblib  # use joblib for sklearn models

st.set_page_config(page_title="flower species predictor", page_icon="🌸")
st.title("flower species predictor")

# ---- Inputs (number_input cannot use None as default) ----
petal_length = st.number_input(
    "please choose a petal length between 1.0 to 6.9",
    placeholder="please enter the petal length",
    min_value=1.0, max_value=6.9, value=1.4, step=0.1
)
petal_width = st.number_input(
    "please choose a petal width between 0.1 to 2.5",
    placeholder="please enter the petal width",
    min_value=0.1, max_value=2.5, value=0.2, step=0.1
)
sepal_length = st.number_input(
    "please choose a sepal length between 4.3 to 7.9",
    placeholder="please enter the sepal length",
    min_value=4.3, max_value=7.9, value=5.1, step=0.1
)
sepal_width = st.number_input(
    "please choose a sepal width between 0.1 to 2.5",
    placeholder="please enter the sepal width",
    min_value=0.1, max_value=2.5, value=1.5, step=0.1
)

# Build input frame in the exact column order your model expects
user_input = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
    dtype=float
)
st.write(user_input)

# ---- Load model once, from model/iris_classifier.pkl ----
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "model" / "iris_classifier.pkl"
    if not model_path.exists():
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    return joblib.load(model_path)

iris_predictor = load_model()

species = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

# ---- Predict ----
if st.button("predict species"):
    try:
        pred = iris_predictor.predict(user_input)[0]
        # If the model returns numeric class labels 0/1/2, map them
        label = species.get(int(pred), str(pred))
        st.success(f"the species is {label}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
