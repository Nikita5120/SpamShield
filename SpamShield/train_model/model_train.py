# train_model/model_train.py
import spacy
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils.preprocess import preprocess  # Import your preprocessing function

# Example LSTM Model architecture (from previous code)
from model import LSTMModel

# Load the spaCy model for preprocessing
nlp = spacy.load("en_core_web_sm")

# Sample Data
texts = ["this is a sample sentence", "let's preprocess this text", "tokenization is important"]
labels = [0, 1, 0]  # Dummy labels

# Define Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer(text)[:self.max_len]
        return torch.tensor(tokens), torch.tensor(label)

# Define tokenizer function (you can replace with a more complex one)
def tokenizer(text):
    return [hash(w) % 10000 for w in text.split()]

# Hyperparameters
vocab_size = 10000
embed_size = 50
hidden_size = 64
num_classes = 2  # Binary classification (you can change this for multi-class)
learning_rate = 0.001
epochs = 5

# Initialize DataLoader
dataset = TextDataset(texts, labels, tokenizer, max_len=10)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize the model
model = LSTMModel(vocab_size, embed_size, hidden_size, num_classes)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}, Accuracy: {correct/total:.4f}')

# After training, save the model
torch.save(model.state_dict(), 'model.pth')
