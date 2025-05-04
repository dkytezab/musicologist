import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class LogisiticClassifier(nn.Module):
    def __init__(self, input_dim):
        super(LogisiticClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        return x.squeeze()
        
class EmbeddingDataset(Dataset):
    def __init__(self, pos_file_path, neg_file_path):
        """
        Dataset for loading embeddings from two class folders
        
        Args:
            pos_file_path: Path to the .pt file containing positive embeddings
            neg_file_path: Path to the .pt file containing negative embeddings
        """
        # Load embeddings
        self.pos_embeddings = torch.load(pos_file_path)
        self.neg_embeddings = torch.load(neg_file_path)
        
        # Create labels (0 for pos, 1 for neg)
        self.pos_labels = torch.zeros(len(self.pos_embeddings))
        self.neg_labels = torch.ones(len(self.neg_embeddings))
        
        # Combine embeddings and labels
        self.embeddings = torch.cat((self.pos_embeddings, self.neg_embeddings), dim=0)
        self.labels = torch.cat((self.pos_labels, self.neg_labels), dim=0)
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

def test_accuracy(model, dataloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Test the accuracy of the model on the test set
    Args:
        model: Trained model
        dataloader: DataLoader for the test set
    """
    model = model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for embeddings, labels in dataloader:
            labels = labels.to(device)
            embeddings = embeddings.to(device)
            outputs = model(embeddings)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

def train_classifier(pos_file_path,
                     neg_file_path,
                     batch_size=32,
                     num_epochs=10,
                     learning_rate=0.001):
    """
    Train a logistic classifier on the embeddings
    Args:
        pos_file_path: Path to the .pt file containing positive embeddings
        neg_file_path: Path to the .pt file containing negative embeddings
        batch_size: Batch size for training
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for the optimizer
    """
    # Load dataset
    dataset = EmbeddingDataset(pos_file_path, neg_file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    input_dim = dataset[0][0].shape[0]
    # Initialize model
    model = LogisiticClassifier(input_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Training loop
    for epoch in range(num_epochs):
        for i, (embeddings, labels) in enumerate(dataloader):
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc = test_accuracy(model, dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {acc:.2f}')
    print("Training complete.")
    return model

if __name__ == "__main__":
    # Example usage
    pos_file_path = 'data/tempo/buckets/bucket_160_170/pos/laion-clap_embeddings.pt'
    neg_file_path = 'data/tempo/buckets/bucket_160_170/neg/laion-clap_embeddings.pt'
    model = train_classifier(pos_file_path, neg_file_path, batch_size=32, num_epochs=10, learning_rate=0.0001)
    torch.save(model.state_dict(), 'logistic_classifier.pth')
    print("Model saved.")
