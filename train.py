import json
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from preprocess import Preprocessor, ChatDataset
from model import NeuralNet
from multiprocessing import freeze_support

def train():
    # Load JSON file
    with open(r'D:\Esoft\AI\ChatBot\intents.json', 'r') as f:  # Use raw string or correct path
        intents = json.load(f)

    # Initialize preprocessor
    preprocessor = Preprocessor()

    # Prepare data
    all_words = []
    tags = []
    xy = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            tokenized = preprocessor.tokenize(pattern)
            all_words.extend(tokenized)
            xy.append((tokenized, tag))

# have to implement mispelling words

    # Stem and remove duplicates
    ignore_words = ['?', '.', '!']
    all_words = [preprocessor.stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    # Create training data
    X_train = []
    y_train = []
    for (pattern_sentence, tag) in xy:
        bag = preprocessor.bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        label = tags.index(tag)
        y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Hyperparameters
    input_size = len(all_words)
    hidden_size = 128
    output_size = len(tags)
    batch_size = 4
    learning_rate = 0.001
    num_epochs = 3000

    # Create dataset and dataloader
    dataset = ChatDataset(X_train, y_train)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Initialize model, loss function, and optimizer
    device = torch.device('cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("done")
    # Training loop
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)

            # Forward pass
            outputs = model(words)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print(f'Final Loss: {loss.item():.4f}')

    # Save the model
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags
    }

    FILE = "chatbot_model.pth"
    torch.save(data, FILE)
    print(f'Training complete. Model saved to {FILE}')


if __name__ == '__main__':
    freeze_support()  # Optional but recommended for Windows
    train()