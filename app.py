import streamlit as st
import tensorflow as tf
import numpy as np

# ✅ Define model path
MODEL_PATH = r"D:\shadow_knight\Sentiment Analysis on IMDb Reviews using a Feedforward Neural Network\training_models\sentiment_analysis_model.keras"

# ✅ Load model safely
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    # Find the TextVectorization layer dynamically
    vectorizer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.TextVectorization):
            vectorizer = layer
            break
    if vectorizer is None:
        raise ValueError("No TextVectorization layer found in the model. Ensure it was included when saving the model.")

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ✅ Function to predict sentiment
def predict_sentiment(review_text):
    if not review_text.strip():
        return "Please enter a review.", 0.0

    # Convert input to a NumPy array with shape (1,)
    review_array = np.array([review_text])

    # ✅ Ensure vectorizer exists
    if vectorizer is None:
        return "Vectorizer not found in model.", 0.0

    # ✅ Vectorize the review using the correct TextVectorization layer
    review_vectorized = vectorizer(review_array)  # Convert text -> token indices

    # ✅ Ensure correct dtype for model input
    review_vectorized = tf.cast(review_vectorized, tf.int32)

    # Make prediction
    prediction = model.predict(review_vectorized)[0][0]

    # Determine sentiment
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    confidence = round(float(prediction) * 100, 2)  # Convert to percentage

    return sentiment, confidence

# ✅ Streamlit App UI
st.title("🎭 Sentiment Analysis on IMDb Reviews")

# User input
user_review = st.text_area("Enter your movie review here:")

if st.button("Analyze Sentiment"):
    sentiment, confidence = predict_sentiment(user_review)
    st.subheader(f"Sentiment: {sentiment}")
    st.write(f"Confidence: {confidence}%")
