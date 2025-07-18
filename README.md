# 📬 Spam Detector API

This is a simple API that takes a message as input and tells you whether it is **spam** or **not spam**.

It uses a deep learning model trained on SMS messages, along with a text vectorizer (`TfidfVectorizer`) to process the input.

The API is built using **FastAPI** and loads:
- A trained model (`spam_model.h5`)
- A fitted vectorizer (`vectorizer.pkl`)
