import torch
from models import WAE
from train import train_WAE
from data_loader import load_data
import matplotlib.pyplot as plt

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model and training parameters
    params = {
        'num_epoch': 400,
        'batch_size': 4,
        'lr': 5e-4,
        'weight_decay': 0.0,
        'sigma': 8.0,
        'MMD_lambda': 1e-4,
        'properties_weight': 1e-1,
        'model_name': '',
    }
    
    latent_size = 16 
    in_features = 30 

    # Load data
    train_loader, test_loader = load_data('', '')

    # Initialize model and optimizer
    model = WAE(in_features, latent_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    # Train model
    loss_recon, loss_properties, Epoch_list = train_WAE(model, optimizer, train_loader, params, device)

    # Plot results
    plt.figure()
    plt.plot(Epoch_list, loss_recon, 'b-', linewidth=2, label='Recon Loss')
    plt.plot(Epoch_list, loss_properties, 'r-', linewidth=2, label='Properties Loss')
    plt.xlabel('Epoch', fontsize=15, weight='bold')
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
