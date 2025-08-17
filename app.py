import streamlit as st
import joblib

# Load the model and vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Title
st.title("📰 Fake News Detector")
st.write("Enter a news article to check if it's FAKE or REAL")

# User input
text_input = st.text_area("Enter news content (title or body):")

# Predict button
if st.button("Check"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Vectorize input
        transformed_input = vectorizer.transform([text_input])
        prediction = model.predict(transformed_input)

        # Show result
        if prediction[0] == "FAKE":
            st.error("🚨 This news is likely FAKE!")
        else:
            st.success("✅ This news appears to be REAL.")
import streamlit as st

st.title("Fake News Detector")

