import torch
import torch.nn as nn
import Trainer, Dataloader
import json

class MLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(MLP, self).__init__()

        # Create the embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Create the MLP layers
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        # Embed the input
        x = self.embedding(x)

        # Flatten the input
        x = x.view(x.shape[0], -1)

        # Pass through the MLP layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x

# Create the MLP model
model = MLP(20000, 200, 128, 2)

# Load the pre-trained word embeddings
with open('embeddings.json') as f:
    embeddings = json.load(f)
model.embedding.weight.data.copy_(torch.tensor(embeddings))

num_epochs = 10
# Train the model
trainer = Trainer(model, CSVDataLoader(), val_loader, test_loader, 'CrossEntropyLoss', optimizer)
trainer.train(num_epochs)
trainer.test()