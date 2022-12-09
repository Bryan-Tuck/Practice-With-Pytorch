import torch
from torch.nn import BCELogitsLoss, CrossEntropyLoss

if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, criterion, optimizer):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer

        # Choose the loss criterion based on the input
        if criterion == 'BCEWithLogitsLoss':
            self.criterion = BCELogitsLoss()
        elif criterion == 'CrossEntropyLoss':
            self.criterion = CrossEntropyLoss()

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            # Train the model
            self.model.train()
            for i, (inputs, labels) in enumerate(self.train_loader):
                # Move inputs and labels to the device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

            # Evaluate the model
            self.model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for i, (inputs, labels) in enumerate(self.val_loader):
                    # Move inputs and labels to the device
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Forward pass
                    outputs = self.model(inputs)

                    # Compute accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                val_acc = correct / total
                print('Epoch [{}/{}], Validation Acc: {}'.format(
                    epoch+1, num_epochs, val_acc))

    def test(self):
        # Test the model
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (inputs, labels) in enumerate(self.test_loader):
                # Move inputs and labels to the device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = self.model(inputs)

                # Compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            test_acc = correct / total
            print('Test Acc: {}'.format(test_acc))
