# # Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 📚 Load the pre-split datasets
# -----------------------------
train_data = pd.read_csv("train_data.csv")
val_data = pd.read_csv("val_data.csv")
test_data = pd.read_csv("test_data.csv")

# -------------------------------------
# 📝 Preprocess the text data using TF-IDF
# -------------------------------------
vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
X_train = vectorizer.fit_transform(train_data["Text"])
X_val = vectorizer.transform(val_data["Text"])
X_test = vectorizer.transform(test_data["Text"])

# -------------------------
# 🎯 Encode the labels
# -------------------------
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data["label"])
y_val = label_encoder.transform(val_data["label"])
y_test = label_encoder.transform(test_data["label"])

# -------------------------
# ⚡️ Train Random Forest model
# -------------------------
model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# ------------------------------------------
# 📊 Evaluate the model on test set
# ------------------------------------------
test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)
print(f"Random Forest Model Test Accuracy: {test_accuracy:.2%}")

# -------------------------
# 📡 Plot Confusion Matrix
# -------------------------
cm = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()

# -------------------------
# 🧠 ROC-AUC Calculation
# -------------------------
# Binarize labels for multi-class ROC
n_classes = len(label_encoder.classes_)
y_test_bin = label_binarize(y_test, classes=range(n_classes))
classifier = OneVsRestClassifier(model)
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Class ROC-AUC Curve for Random Forest Model')
plt.legend(loc='lower right')
plt.show()

# -------------------------
# ✨ Prediction Function
# -------------------------
def predict_sentiment(text):
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    return predicted_label

# ------------------------------------
# 🔥 Test the function with a custom input
# ------------------------------------
user_input = input("Enter a sentence for sentiment analysis: ")
predicted_sentiment = predict_sentiment(user_input)
print(f"The sentiment of the given text is: {predicted_sentiment}")

