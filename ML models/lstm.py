# import pandas as pd
# import ast
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score
# from gensim.models import Word2Vec
# import pickle

# # Load dataset
# train_data = pd.read_csv("train_data.csv")
# test_data = pd.read_csv("test_data.csv")
# val_data = pd.read_csv("val_data.csv")  # Integrated validation data

# # Preprocess the text data
# train_data['Tokenized text'] = train_data['Tokenized text'].apply(ast.literal_eval)
# test_data['Tokenized text'] = test_data['Tokenized text'].apply(ast.literal_eval)
# val_data['Tokenized text'] = val_data['Tokenized text'].apply(ast.literal_eval)

# # Train Word2Vec Model
# w2v_model = Word2Vec(sentences=train_data['Tokenized text'], vector_size=100, window=5, min_count=1, workers=4)
# vocab = {word: idx + 1 for idx, word in enumerate(w2v_model.wv.index_to_key)}

# # Create Embedding Matrix
# embedding_matrix = np.zeros((len(vocab) + 1, 100))
# for word, idx in vocab.items():
#     embedding_matrix[idx] = w2v_model.wv[word]

# # Convert tokenized text into sequences of indices
# def text_to_sequences(texts, vocab):
#     return [[vocab.get(word, 0) for word in sentence] for sentence in texts]

# X_train = text_to_sequences(train_data['Tokenized text'], vocab)
# X_test = text_to_sequences(test_data['Tokenized text'], vocab)
# X_val = text_to_sequences(val_data['Tokenized text'], vocab)

# # Padding sequences
# max_length = max(max(len(seq) for seq in X_train), max(len(seq) for seq in X_test), max(len(seq) for seq in X_val))
# def pad_sequences(sequences, maxlen, padding='post'):
#     padded = np.zeros((len(sequences), maxlen), dtype=int)
#     for i, seq in enumerate(sequences):
#         padded[i, :len(seq)] = seq[:maxlen]
#     return padded

# X_train = pad_sequences(X_train, max_length, padding='post')
# X_test = pad_sequences(X_test, max_length, padding='post')
# X_val = pad_sequences(X_val, max_length, padding='post')

# # Encode labels
# label_encoder = LabelEncoder()
# y_train = label_encoder.fit_transform(train_data['label'])
# y_test = label_encoder.transform(test_data['label'])
# y_val = label_encoder.transform(val_data['label'])

# # Convert data to tensors
# X_train_tensor = torch.tensor(X_train, dtype=torch.long)
# y_train_tensor = torch.tensor(y_train, dtype=torch.long)
# X_test_tensor = torch.tensor(X_test, dtype=torch.long)
# y_test_tensor = torch.tensor(y_test, dtype=torch.long)
# X_val_tensor = torch.tensor(X_val, dtype=torch.long)
# y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# # Create DataLoader
# batch_size = 32
# train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)

# # Define LSTM Model
# class SentimentLSTM(nn.Module):
#     def __init__(self, vocab_size, embedding_matrix, hidden_dim=128, num_layers=2, num_classes=3):
#         super(SentimentLSTM, self).__init__()
#         self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)
#         self.lstm = nn.LSTM(input_size=100, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(hidden_dim * 2, num_classes)

#     def forward(self, x):
#         x = self.embedding(x)
#         lstm_out, _ = self.lstm(x)
#         x = lstm_out[:, -1, :]
#         x = self.fc(x)
#         return x

# # Initialize Model
# model = SentimentLSTM(vocab_size=len(vocab) + 1, embedding_matrix=embedding_matrix)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Define Loss and Optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Train the Model and Track Loss
# epochs = 20
# train_losses = []
# test_losses = []

# for epoch in range(epochs):
#     total_train_loss = 0
#     model.train()
#     for X_batch, y_batch in train_loader:
#         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#         optimizer.zero_grad()
#         outputs = model(X_batch)
#         loss = criterion(outputs, y_batch)
#         loss.backward()
#         optimizer.step()
#         total_train_loss += loss.item()

#     train_losses.append(total_train_loss)

#     # Compute test loss
#     model.eval()
#     with torch.no_grad():
#         test_outputs = model(X_test_tensor.to(device))
#         test_loss = criterion(test_outputs, y_test_tensor.to(device)).item()
    
#     test_losses.append(test_loss)
#     print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {total_train_loss:.4f}, Test Loss: {test_loss:.4f}")
# torch.save(model.state_dict(), "sentiment_lstm.pth")
# with open("vocab.pkl", "wb") as f:
#     pickle.dump(vocab, f)
# w2v_model.save("word2vec.model")
# # Predictions on test set
# model.eval()
# X_test_tensor, y_test_tensor = X_test_tensor.to(device), y_test_tensor.to(device)

# with torch.no_grad():
#     outputs = model(X_test_tensor)
#     _, predicted = torch.max(outputs, 1)

# accuracy = accuracy_score(y_test_tensor.cpu(), predicted.cpu())
# print(f"LSTM Model Accuracy: {accuracy:.2%}")

import pandas as pd
import ast
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from gensim.models import Word2Vec
import pickle

# Load dataset
train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")
val_data = pd.read_csv("val_data.csv")  # Integrated validation data

# Preprocess the text data
train_data['Tokenized text'] = train_data['Tokenized text'].apply(ast.literal_eval)
test_data['Tokenized text'] = test_data['Tokenized text'].apply(ast.literal_eval)
val_data['Tokenized text'] = val_data['Tokenized text'].apply(ast.literal_eval)

# Train Word2Vec Model
w2v_model = Word2Vec(sentences=train_data['Tokenized text'], vector_size=100, window=5, min_count=1, workers=4)
vocab = {word: idx + 1 for idx, word in enumerate(w2v_model.wv.index_to_key)}

# Create Embedding Matrix
embedding_matrix = np.zeros((len(vocab) + 1, 100))
for word, idx in vocab.items():
    embedding_matrix[idx] = w2v_model.wv[word]

# Convert tokenized text into sequences of indices
def text_to_sequences(texts, vocab):
    return [[vocab.get(word, 0) for word in sentence] for sentence in texts]

X_train = text_to_sequences(train_data['Tokenized text'], vocab)
X_test = text_to_sequences(test_data['Tokenized text'], vocab)
X_val = text_to_sequences(val_data['Tokenized text'], vocab)

# Padding sequences
max_length = max(max(len(seq) for seq in X_train), max(len(seq) for seq in X_test), max(len(seq) for seq in X_val))
def pad_sequences(sequences, maxlen, padding='post'):
    padded = np.zeros((len(sequences), maxlen), dtype=int)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq[:maxlen]
    return padded

X_train = pad_sequences(X_train, max_length, padding='post')
X_test = pad_sequences(X_test, max_length, padding='post')
X_val = pad_sequences(X_val, max_length, padding='post')

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data['label'])
y_test = label_encoder.transform(test_data['label'])
y_val = label_encoder.transform(val_data['label'])

# Convert data to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# Create DataLoader
batch_size = 32
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)

# Define LSTM Model
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_matrix, hidden_dim=128, num_layers=2, num_classes=3):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)
        self.lstm = nn.LSTM(input_size=100, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.fc(x)
        return x

# Initialize Model
model = SentimentLSTM(vocab_size=len(vocab) + 1, embedding_matrix=embedding_matrix)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the Model and Track Loss
epochs = 20
train_losses = []
test_losses = []

for epoch in range(epochs):
    total_train_loss = 0
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    train_losses.append(total_train_loss)

    # Compute test loss
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor.to(device))
        test_loss = criterion(test_outputs, y_test_tensor.to(device)).item()
    
    test_losses.append(test_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {total_train_loss:.4f}, Test Loss: {test_loss:.4f}")

# Save model and word2vec
torch.save(model.state_dict(), "sentiment_lstm.pth")
with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)
w2v_model.save("word2vec.model")

# Plot Training & Testing Loss
plt.plot(range(1, epochs + 1), train_losses, marker='o', linestyle='-', label='Training Loss')
plt.plot(range(1, epochs + 1), test_losses, marker='s', linestyle='--', label='Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Testing Loss Over Epochs for LSTM Model')
plt.legend()
plt.show()

# Predictions on test set
model.eval()
X_test_tensor, y_test_tensor = X_test_tensor.to(device), y_test_tensor.to(device)

with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)

# Model Accuracy
accuracy = accuracy_score(y_test_tensor.cpu(), predicted.cpu())
print(f"LSTM Model Accuracy: {accuracy:.2%}")

# Confusion Matrix
cm = confusion_matrix(y_test_tensor.cpu(), predicted.cpu())
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for LSTM Model')
plt.show()

# ROC-AUC Curve
# Binarize the labels for multi-class
y_test_binarized = label_binarize(y_test_tensor.cpu().numpy(), classes=[0, 1, 2])

# Get model predictions as probabilities
model.eval()
with torch.no_grad():
    outputs_prob = F.softmax(model(X_test_tensor.to(device)), dim=1).cpu().numpy()

# Compute ROC-AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = 3
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], outputs_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC-AUC curve for each class
plt.figure(figsize=(8, 8))
colors = ['blue', 'green', 'red']
class_labels = label_encoder.classes_

for i, color in enumerate(colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {class_labels[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve for LSTM Model')
plt.legend(loc='lower right')
plt.show()

# Print AUC Scores for Each Class
for i, class_name in enumerate(class_labels):
    print(f"AUC for class '{class_name}': {roc_auc[i]:.2f}")
