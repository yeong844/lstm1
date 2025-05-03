import streamlit as st
import numpy as np
import joblib
import re
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ======================
# STREAMLIT PAGE SETUP
# ======================
st.set_page_config(
    page_title="Coffee Sentiment Analyzer",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ======================
# CONSTANTS
# ======================
MAX_LEN = 100

# ======================
# MODEL & TOKENIZER LOADING
# ======================
@st.cache_resource
def load_model_and_tokenizer():
    try:
        # Use the correct model filename from your training script
        model = load_model("sentiment_lstm.keras", compile=True)
        tokenizer = joblib.load("tokenizer.pkl")
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load model or tokenizer: {e}")
        return None, None

# ======================
# PREDICTION FUNCTION (Binary: Positive/Negative)
# ======================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def predict_sentiment(model, tokenizer, review):
    cleaned_review = clean_text(review)
    sequence = tokenizer.texts_to_sequences([cleaned_review])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Get raw prediction (this is a single value between 0 and 1 for binary classification)
    prediction = float(model.predict(padded, verbose=0)[0][0])
    
    # In the training script:
    # 1 = positive (stars >= 4)
    # 0 = negative (stars <= 2)
    if prediction >= 0.5:
        label = 1  # Positive
    else:
        label = 0  # Negative

    sentiment_map = {
        0: ("Negative", "😠", "red"),
        1: ("Positive", "😊", "green")
    }

    sentiment, emoji, color = sentiment_map[label]
    
    # For confidence, use how far from 0.5 the prediction is
    confidence = prediction if label == 1 else (1 - prediction)
    
    return {
        "sentiment": sentiment,
        "emoji": emoji,
        "color": color,
        "probability": prediction,
        "confidence": float(confidence),  # Ensure it's a Python float
        "label": label
    }

# ======================
# MAIN INTERFACE
# ======================
st.title("Coffee Review Sentiment Analyzer")
st.write("Analyze the sentiment of coffee product reviews using a LSTM model.")

file_status = st.empty()
model_exists = os.path.exists("sentiment_lstm.keras")
tokenizer_exists = os.path.exists("tokenizer.pkl")

if not model_exists or not tokenizer_exists:
    file_status.error("⚠️ Model or tokenizer file missing!")
    missing_files = []
    if not model_exists:
        missing_files.append("sentiment_lstm.keras")
    if not tokenizer_exists:
        missing_files.append("tokenizer.pkl")
    
    st.info(f"""
    Please ensure these files are in the app directory:
    - {', '.join(missing_files)}
    """)
else:
    file_status.success("✅ Model and tokenizer files found")
    model, tokenizer = load_model_and_tokenizer()

# Input section
st.header("Enter a Coffee Review")
user_input = st.text_area(
    "Review Text:",
    height=150,
    value="",
    placeholder="Example: This coffee has a rich aroma with hints of chocolate and nuts. It has a smooth finish with no bitterness."
)

# Add some example reviews
example_reviews = {
    "Select an example...": "",
    "Positive Example": "This coffee is amazing! Rich flavor and perfect aroma. I've been buying it for months and never disappointed.",
    "Negative Example": "Terrible coffee experience. Bitter taste and stale beans. Will not purchase again."
}

selected_example = st.selectbox("Or try an example review:", list(example_reviews.keys()))

if selected_example != "Select an example..." and selected_example in example_reviews:
    user_input = example_reviews[selected_example]

analyze_button = st.button("Analyze Sentiment", type="primary")

if analyze_button:
    if not user_input.strip():
        st.warning("⚠️ Please enter a review first.")
    elif model is None or tokenizer is None:
        st.error("Cannot analyze: Model or tokenizer could not be loaded.")
    else:
        with st.spinner("Analyzing..."):
            results = predict_sentiment(model, tokenizer, user_input)

            sentiment = results["sentiment"]
            emoji = results["emoji"]
            color = results["color"]
            probability = results["probability"]
            confidence = results["confidence"]
            label = results["label"]

            st.markdown(
                f"### <span style='color:{color}; font-size: 28px;'>{emoji} {sentiment}</span>",
                unsafe_allow_html=True
            )

            # Use float() to ensure it's a Python float, not a numpy float or tensor
            st.progress(float(confidence))
            st.caption(f"Confidence: {confidence:.1%}")

            st.subheader("Sentiment Breakdown")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Positive", f"{probability:.1%}")
            with col2:
                st.metric("Negative", f"{(1-probability):.1%}")

            with st.expander("Preprocessing Details"):
                st.write("**Original Text:**")
                st.write(user_input)
                st.write("**Processed Text:**")
                st.write(clean_text(user_input))

# ======================
# SIDEBAR INFORMATION
# ======================
st.sidebar.title("About")
st.sidebar.info("""
This app uses an LSTM model trained to classify sentiment for coffee product reviews.

Sentiment Categories:
- 😠 Negative (1-2 stars)
- 😊 Positive (4-5 stars)

This tool can help coffee sellers understand customer feedback better.
""")

st.sidebar.subheader("How It Works")
st.sidebar.markdown("""
1. You enter a coffee review text
2. The model analyzes the sentiment
3. Results show if the review is positive or negative
4. The confidence score shows how certain the prediction is
""")

st.sidebar.subheader("Tips for Better Results")
st.sidebar.markdown("""
1. Use complete sentences
2. Be specific about what you liked or disliked
3. Mention flavor, aroma, price, or quality aspects
""")
