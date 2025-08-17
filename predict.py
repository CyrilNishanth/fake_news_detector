import joblib

# Load the saved model and vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Example input text (you can replace this with your own)
news = input("📰 Enter the news text to classify: ")

# Transform the input using the vectorizer
news_vector = vectorizer.transform([news])

# Predict using the model
prediction = model.predict(news_vector)

# Print result
print("✅ Prediction:", "FAKE" if prediction[0] == "FAKE" else "REAL")
