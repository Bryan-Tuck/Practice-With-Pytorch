from torch.utils.data import DataLoader, Dataset
import csv, json

class CSVDataLoader(Dataset):
    def __init__(self, csv_file):
        # Load the data from the CSV file
        self.data = []
        with open(csv_file) as f:
            reader = csv.reader(f)
            for row in reader:
                text = row[0]
                label = row[1]
                self.data.append((text, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return the text and label for the data point at the given index
        return self.data[idx]

class EmbeddingDataLoader(Dataset):
    def __init__(self, csv_file, embedding_file):
        # Load the data from the CSV file
        self.data = []
        with open(csv_file) as f:
            reader = csv.reader(f)
            for row in reader:
                text = row[0]
                label = row[1]
                self.data.append((text, label))

        # Load the word embeddings from the file
        with open(embedding_file) as f:
            self.embeddings = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the text and label for the data point at the given index
        text, label = self.data[idx]

        # Convert the text to a sequence of word indices
        indices = []
        for word in text.split():
            if word in self.embeddings:
                indices.append(self.embeddings[word])

        return indices, label