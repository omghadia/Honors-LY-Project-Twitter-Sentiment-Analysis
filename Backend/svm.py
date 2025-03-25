import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load the pre-split datasets
train_data = pd.read_csv("train_data.csv")
val_data = pd.read_csv("val_data.csv")
test_data = pd.read_csv("test_data.csv")

# Check if label column exists
if "label" not in train_data.columns:
    raise ValueError("Dataset must contain a 'label' column for training.")

# Preprocess the text data using TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
X_train = vectorizer.fit_transform(train_data["Text"])
X_val = vectorizer.transform(val_data["Text"])
X_test = vectorizer.transform(test_data["Text"])

y_train = train_data["label"]
y_val = val_data["label"]
y_test = test_data["label"]

# Train an SVM model (linear kernel for text classification)
model = SVC(kernel='linear', class_weight='balanced')
model.fit(X_train, y_train)

# Save the trained model & vectorizer
joblib.dump(model, "svm_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully!")

# Evaluate the model on test set
test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)
print(f"SVM Model Test Accuracy: {test_accuracy:.2%}")

