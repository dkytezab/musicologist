import os
import sys
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.sae import TopKAutoencoder

def create_loader(
        data_path: str,
        diff_step: int,
        layer: int,
        batch_size: int,
        verbose: bool = False
) -> torch.utils.data.DataLoader:
    activation_dir = os.path.join(data_path, f'diff_step_{diff_step}')
    if not os.path.exists(activation_dir):
        raise ValueError(f"Directory {activation_dir} does not exist")

    activation_pattern = f'layer_{layer}_prompt_'
    activation_files = [f for f in os.listdir(activation_dir) 
                        if f.startswith(activation_pattern) and f.endswith('.pt')]

    if not activation_files:
        raise ValueError(f"No activation files found for diff_step {diff_step} and layer {layer}")

    if verbose: print(f"Found {len(activation_files)} activation files for diff_step {diff_step} and layer {layer}.")
    datasets = []
    for file in activation_files:
        file_path = os.path.join(activation_dir, file)
        data = torch.load(file_path)
        datasets.append(data)

    dataset = torch.cat(datasets, dim=0)
    dataset = dataset.view(dataset.shape[0], -1)  # Flatten the data
    if verbose: print(f"Loaded dataset with shape {dataset.shape}.")
    dataset = torch.utils.data.TensorDataset(dataset)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_sae(model: TopKAutoencoder,
              train_loader: torch.utils.data.DataLoader,
              num_epochs: int,
              alpha: float,
              lr: float,
              verbose: bool = False,
              device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for batch_idx, (data,)in enumerate(train_loader):
            data = data.to(device)
            encoded, decoded, encoded_aux, decoded_aux = model(data)
            main_loss = F.mse_loss(decoded, data)
            aux_loss = F.mse_loss(decoded_aux, data)
            loss = main_loss + alpha * aux_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if verbose:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()} | '
                        f'Main Loss: {main_loss.item()}, Aux Loss: {aux_loss.item()}')

def create_sae(
        data_path: str,
        diff_step: int,
        layer: int,
        batch_size: int,
        num_epochs: int,
        alpha: float,
        lr: float,
        topk: int = 64,
        topk_aux: int = 256,
        latent_dim: int = 1000,
        verbose: bool = False,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> TopKAutoencoder:
    # seed
    torch.manual_seed(42)
    train_loader = create_loader(data_path, diff_step, layer, batch_size, verbose)
    model = TopKAutoencoder(input_size=train_loader.dataset[0][0].shape[0],
                            hidden_size=latent_dim,
                            topk=topk,
                            topk_aux=topk_aux)
    model = model.to(device)
    train_sae(model, train_loader, num_epochs, alpha, lr, verbose)
    return model

parser = argparse.ArgumentParser(description='Train a Top-K Autoencoder')
parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory')
parser.add_argument('--diff_step', type=int, required=True, help='Diffusion step to use')
parser.add_argument('--layer', type=int, required=True, help='Layer to use')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('--alpha', type=float, default=0.1, help='Weight for auxiliary loss')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
parser.add_argument('--topk', type=int, default=64, help='Top-K value for autoencoder')
parser.add_argument('--topk_aux', type=int, default=256, help='Top-K auxiliary value for autoencoder')
parser.add_argument('--latent_dim', type=int, default=1000, help='Latent dimension for autoencoder')
parser.add_argument('--verbose', action='store_true', help='Print training progress')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training (cpu or cuda)')
args = parser.parse_args()
if __name__ == "__main__":
    model = create_sae(
        data_path=args.data_path,
        diff_step=args.diff_step,
        layer=args.layer,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        alpha=args.alpha,
        lr=args.lr,
        topk=args.topk,
        topk_aux=args.topk_aux,
        latent_dim=args.latent_dim,
        verbose=args.verbose,
        device=args.device
    )
    print("SAE training complete.")
