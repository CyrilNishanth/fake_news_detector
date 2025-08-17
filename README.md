**📰 Fake News Detector WebApp**

A Streamlit-powered web application that detects whether a news article is real or fake using a machine learning model trained on Kaggle datasets.

**🚀 Features**

Upload or paste a news article for classification.

Detect if the article is Fake 🛑 or Real ✅.

Uses pre-trained ML/NLP models trained on Kaggle datasets.

Simple, interactive UI built with Streamlit.

**📊 Dataset**

This project uses Kaggle’s Fake News Detection Dataset (you can download from: Fake News Dataset on Kaggle).

The dataset typically contains:

id → Unique identifier

title → Title of the news article

author → Author of the article

text → Main content of the article

label → 1 = Fake, 0 = Real

**🛠️ Installation**

Clone the repository:

git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector


**(Optional) Create a virtual environment:**

python -m venv venv

source venv/bin/activate   # Mac/Linux

venv\Scripts\activate      # Windows


**Install dependencies:**

pip install -r requirements.txt


**(Optional) Download the Kaggle dataset and place it in the data/ folder:**

mkdir data
True.csv
Fake.csv

**▶️ Usage**

Run the Streamlit app:

streamlit run app.py


Then open the link shown in your terminal (usually http://localhost:8501) in your browser.

**⚙️ Tech Stack**

Frontend: Streamlit

Backend: Python

ML Model: Logistic Regression / Naive Bayes / LSTM (depending on your implementation)

Dataset: Kaggle Fake News Dataset

**📌 Notes**

Currently supports text input for classification.

Accuracy depends on the chosen ML model and preprocessing.

Future plans: Add deep learning models (LSTMs, Transformers) for better accuracy.

**🤝 Contributing**

**Fork the repo.**

**Create a new branch:**

git checkout -b feature-branch


**Commit your changes:**

git commit -m "Added feature XYZ"


**Push to your branch:**

git push origin feature-branch


**Open a Pull Request 🚀**
