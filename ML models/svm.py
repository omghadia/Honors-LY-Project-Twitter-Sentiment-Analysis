import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# Load the pre-split datasets
train_data = pd.read_csv("train_data.csv")
val_data = pd.read_csv("val_data.csv")
test_data = pd.read_csv("test_data.csv")



# Preprocess the text data using TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
X_train = vectorizer.fit_transform(train_data["Text"])
X_val = vectorizer.transform(val_data["Text"])
X_test = vectorizer.transform(test_data["Text"])

y_train = train_data["label"]
y_val = val_data["label"]
y_test = test_data["label"]

# Train an SVM model
epochs = 20
train_losses = []
test_losses = []
model = SVC(kernel='linear', class_weight='balanced')
model.fit(X_train, y_train)
# for epoch in range(epochs):
#     model.fit(X_train, y_train)
#     train_pred = model.predict(X_train)
#     test_pred = model.predict(X_test)
#     train_loss = 1 - accuracy_score(y_train, train_pred)
#     test_loss = 1 - accuracy_score(y_test, test_pred)
#     train_losses.append(train_loss)
#     test_losses.append(test_loss)
#     #print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

# # Plot Training & Testing Loss
# # plt.plot(range(1, epochs+1), train_losses, marker='o', linestyle='-', label='Training Loss')
# # plt.plot(range(1, epochs+1), test_losses, marker='s', linestyle='--', label='Testing Loss')
# # plt.xlabel('Epoch')
# # plt.ylabel('Loss')
# # plt.title('Training & Testing Loss Over Epochs for SVM model')
# # plt.legend()
# # plt.show()
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# # Evaluate the model on test set
test_pred = model.predict(X_test)
# cm = confusion_matrix((y_test), (test_pred))

# # Plot confusion matrix
# plt.figure(figsize=(6,6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix for svm model')
# plt.show()
test_accuracy = accuracy_score(y_test, test_pred)
print(f"SVM Model Test Accuracy: {test_accuracy:.2%}")

# Function to make predictions on new data
def predict_sentiment(text):
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)
    return prediction[0]

# Test the function with a custom input
user_input = input("Enter a sentence for sentiment analysis: ")
predicted_sentiment = predict_sentiment(user_input)
print(f"The sentiment of the given text is: {predicted_sentiment}")
# from sklearn.metrics import roc_curve, auc
# import matplotlib.pyplot as plt

# # Get prediction probabilities
# #y_prob = model.predict_proba(X_test)

# # Get ROC curve values
# y_scores = model.decision_function(X_test)
# fpr, tpr, _ = roc_curve(y_test, y_scores)
# roc_auc = auc(fpr, tpr)

# # Plot ROC curve
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC-AUC Curve for SVM Model')
# plt.legend(loc='lower right')
# plt.show()

# # Import necessary libraries
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.preprocessing import label_binarize
# import matplotlib.pyplot as plt
# import seaborn as sns


# train_data = pd.read_csv("train_data.csv")
# val_data = pd.read_csv("val_data.csv")
# test_data = pd.read_csv("test_data.csv")


# vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
# X_train = vectorizer.fit_transform(train_data["Text"])
# X_val = vectorizer.transform(val_data["Text"])
# X_test = vectorizer.transform(test_data["Text"])


# label_encoder = LabelEncoder()
# y_train = label_encoder.fit_transform(train_data["label"])
# y_val = label_encoder.transform(val_data["label"])
# y_test = label_encoder.transform(test_data["label"])


# model = SVC(kernel='linear', class_weight='balanced')
# model.fit(X_train, y_train)


# test_pred = model.predict(X_test)
# test_accuracy = accuracy_score(y_test, test_pred)
# print(f"SVM Model Test Accuracy: {test_accuracy:.2%}")

# cm = confusion_matrix(y_test, test_pred)
# plt.figure(figsize=(6, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix for SVM Model')
# plt.show()


# n_classes = len(label_encoder.classes_)
# if n_classes > 2:
#     y_test_bin = label_binarize(y_test, classes=range(n_classes))
#     classifier = OneVsRestClassifier(model)
#     y_score = classifier.fit(X_train, y_train).decision_function(X_test)

#     fpr, tpr, roc_auc = dict(), dict(), dict()
#     for i in range(n_classes):
#         fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])

#     # Plot ROC curve for each class
#     plt.figure(figsize=(8, 6))
#     for i in range(n_classes):
#         plt.plot(fpr[i], tpr[i], label=f'Class {label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})')

#     plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Multi-Class ROC-AUC Curve for SVM Model')
#     plt.legend(loc='lower right')
#     plt.show()
# else:
#     # For binary classification
#     y_scores = model.decision_function(X_test)
#     fpr, tpr, _ = roc_curve(y_test, y_scores)
#     roc_auc = auc(fpr, tpr)

#     # Plot ROC curve
#     plt.figure(figsize=(8, 6))
#     plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
#     plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC-AUC Curve for SVM Model')
#     plt.legend(loc='lower right')
#     plt.show()


# def predict_sentiment(text):
#     transformed_text = vectorizer.transform([text])
#     prediction = model.predict(transformed_text)
#     predicted_label = label_encoder.inverse_transform(prediction)[0]
#     return predicted_label


# user_input = input("Enter a sentence for sentiment analysis: ")
# predicted_sentiment = predict_sentiment(user_input)
# print(f"The sentiment of the given text is: {predicted_sentiment}")
