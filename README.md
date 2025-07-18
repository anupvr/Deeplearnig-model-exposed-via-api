# 📬 Spam Detection Model Trainer

This project trains a spam classification model using the SMS Spam Collection dataset and exports:
- A trained Keras model: `spam_model.h5`
- A TF-IDF vectorizer: `vectorizer.pkl`

These artifacts can be used in a FastAPI server for real-time spam detection.

---

## 📦 Requirements

- Python 3.8 - 3.10
- tensor is not supported above 3.10 at this time when i am creating this repo
- Virtual environment (`venv`) recommended

---

## 🪜 Setup Instructions
- download spam.csv from kaggler
### 1. Clone or download this project
```bash
git clone <your-repo-url>
cd spam_model_trainer