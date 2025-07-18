from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import tensorflow as tf
import numpy as np
import uvicorn  # ✅ ADD THIS
# Load model and vectorizer once at startup
model = tf.keras.models.load_model("spam_model.h5")
#vectorizer = joblib.load("vectorizer.pkl")
print(">>> Loading vectorizer...")
vectorizer = joblib.load("vectorizer.pkl")

print("✅ Type:", type(vectorizer))
print("✅ Has idf_?:", hasattr(vectorizer, "idf_"))

try:
    print("✅ IDF shape:", vectorizer.idf_.shape)
except Exception as e:
    print("❌ ERROR accessing idf_:", e)
app = FastAPI()

class MessageInput(BaseModel):
    message: str

# ✅ Show message on root URL
@app.get("/")
def read_root():
    return {"message": "Spam Detector API is live. Use /predict to POST messages or /docs for swagger."}


@app.post("/predict")
def predict(input: MessageInput):
    # Vectorize input message
    vector = vectorizer.transform([input.message])
    
    # Predict spam probability
    prediction = model.predict(vector)[0][0]
    
    return {
        "spam": bool(prediction > 0.5),
        "confidence": float(np.round(prediction, 4))
    }

# ✅ Run the API directly with `python app.py`
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)