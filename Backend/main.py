from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import io

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained SVM model and vectorizer
svm_model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


def get_sentiment(text):
    transformed_text = vectorizer.transform([text])  # Convert input to TF-IDF features
    prediction = svm_model.predict(transformed_text)[0]  # Get model prediction
    return prediction  # Assuming model returns 'positive', 'negative', 'neutral'


class TextRequest(BaseModel):
    text: str


@app.post("/predict_text/")
async def predict_text(request: TextRequest):
    sentiment = get_sentiment(request.text)
    return {"text": request.text, "sentiment": sentiment}


@app.post("/predict_csv/")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    content = await file.read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8")))

    # Automatically detect the column containing text
    text_column = None
    for col in df.columns:
        sample_value = (
            str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else ""
        )
        if (
            isinstance(sample_value, str) and len(sample_value.split()) > 3
        ):  # Checking if it's likely to be a text column
            text_column = col
            break

    if not text_column:
        raise HTTPException(
            status_code=400, detail="No valid text column found in the CSV."
        )

    df["sentiment"] = df[text_column].apply(get_sentiment)
    return df.to_dict(orient="records")
