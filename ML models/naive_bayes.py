# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score




# # Load the pre-split datasets
# train_data = pd.read_csv("train_data.csv")
# val_data = pd.read_csv("val_data.csv")
# test_data = pd.read_csv("test_data.csv")

# # Preprocess the text data using Count Vectorizer
# vectorizer = CountVectorizer(max_features=500, stop_words='english')
# X_train = vectorizer.fit_transform(train_data["Text"])
# X_val = vectorizer.transform(val_data["Text"])
# X_test = vectorizer.transform(test_data["Text"])

# y_train = train_data["label"]
# y_val = val_data["label"]
# y_test = test_data["label"]

# # Train a Naive Bayes model
# epochs = 20
# train_losses = []
# test_losses = []
# model = MultinomialNB()

# for epoch in range(epochs):
#     model.fit(X_train, y_train)
#     train_pred = model.predict(X_train)
#     test_pred = model.predict(X_test)
#     train_loss = 1 - accuracy_score(y_train, train_pred)
#     test_loss = 1 - accuracy_score(y_test, test_pred)
#     train_losses.append(train_loss)
#     test_losses.append(test_loss)
#     print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

# # Plot Training & Testing Loss
# # plt.plot(range(1, epochs+1), train_losses, marker='o', linestyle='-', label='Training Loss')
# # plt.plot(range(1, epochs+1), test_losses, marker='s', linestyle='--', label='Testing Loss')
# # plt.xlabel('Epoch')
# # plt.ylabel('Loss')
# # plt.title('Training & Testing Loss Over Epochs for Naive Bayes Model')
# # plt.legend()
# # plt.show()

# def evaluate_model_metrics(y_true, y_pred):
#     precision = precision_score(y_true, y_pred, average="weighted")
#     recall = recall_score(y_true, y_pred, average="weighted")
#     f1 = f1_score(y_true, y_pred, average="weighted")

#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1-score: {f1:.4f}")

# # Predictions on test set
# test_pred = model.predict(X_test)

# # Calculate metrics
# evaluate_model_metrics(y_test, test_pred)

# # Evaluate the model on test set
# test_pred = model.predict(X_test)
# test_accuracy = accuracy_score(y_test, test_pred)
# print(f"Naive Bayes Model Test Accuracy: {test_accuracy:.2%}")

# # # Function to make predictions on new data
# # def predict_sentiment(text):
# #     transformed_text = vectorizer.transform([text])
# #     prediction = model.predict(transformed_text)
# #     return prediction[0]

# # # Test the function with a custom input
# # user_input = input("Enter a sentence for sentiment analysis: ")
# # predicted_sentiment = predict_sentiment(user_input)
# # print(f"The sentiment of the given text is: {predicted_sentiment}")

# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
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
# ⚡️ Train Naive Bayes model
# -------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# ------------------------------------------
# 📊 Evaluate the model on test set
# ------------------------------------------
test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)
print(f"Naive Bayes Model Test Accuracy: {test_accuracy:.2%}")

# -------------------------
# 📡 Plot Confusion Matrix
# -------------------------
cm = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Naive Bayes Model')
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
plt.title('Multi-Class ROC-AUC Curve for Naive Bayes Model')
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
