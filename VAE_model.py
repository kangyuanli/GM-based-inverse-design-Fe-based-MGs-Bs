import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model definition with an encoder and decoder.
    The encoder maps input features to latent space, and the decoder reconstructs the input.
    """
    def __init__(self, in_features, latent_size):
        super(VAE, self).__init__()

        # Encoder layers
        self.encoder_fcl_1 = nn.Linear(in_features, 90)
        self.encoder_fcl_1_norm = nn.LayerNorm(90)
        self.encoder_fcl_2 = nn.Linear(90, 48)
        self.encoder_fcl_2_norm = nn.LayerNorm(48)

        # Latent space parameters
        self.latent_mu = nn.Linear(48, latent_size)
        self.latent_var = nn.Linear(48, latent_size)

        # Decoder layers
        self.decoder_fcl_1 = nn.Linear(latent_size, 48)
        self.decoder_fcl_1_norm = nn.LayerNorm(48)
        self.decoder_fcl_2 = nn.Linear(48, 90)
        self.decoder_fcl_2_norm = nn.LayerNorm(90)
        self.out_fcl = nn.Linear(90, in_features)

    def encoder(self, X):
        out = F.relu(self.encoder_fcl_1_norm(self.encoder_fcl_1(X)))
        out = F.relu(self.encoder_fcl_2_norm(self.encoder_fcl_2(out)))
        mu = self.latent_mu(out)
        log_var = self.latent_var(out)
        return mu, log_var

    def decoder(self, z):
        h = F.relu(self.decoder_fcl_1_norm(self.decoder_fcl_1(z)))
        h = F.relu(self.decoder_fcl_2_norm(self.decoder_fcl_2(h)))
        x_reconst = F.softmax(self.out_fcl(h), dim=1)
        return x_reconst

    def reparameterization(self, mu, log_var):
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, X):
        mu, log_var = self.encoder(X)
        z = self.reparameterization(mu, log_var)
        x_reconst = self.decoder(z)
        return x_reconst, mu, log_var


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def train_VAE(train_data_loader, vae, optimizer, num_epochs=400, device='cuda'):
    """
    Trains the VAE model using the provided data loader and optimizer.
    """
    print('Start Training VAE...')

    vae.to(device)
    vae.train()

    Epoch_list = []
    Recon_loss_list = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        total_recon_loss = 0

        for x, _ in train_data_loader:
            x = x.to(device)
            x_reconst, mu, log_var = vae(x)

            # Calculate reconstruction loss and KL divergence
            reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='mean')
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            loss = reconst_loss + kl_div * 1e-4

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon_loss += reconst_loss.item()

        avg_recon_loss = total_recon_loss / len(train_data_loader)
        Epoch_list.append(epoch)
        Recon_loss_list.append(avg_recon_loss)

        print(f'Average Reconstruction Loss: {avg_recon_loss:.6f}')

    # Save the loss values for further use
    torch.save(vae.state_dict(), 'vae_inverse_design_model.pth')

    # Plot reconstruction loss over epochs
    plt.figure()
    plt.plot(Epoch_list, Recon_loss_list, c='r', linewidth=2)
    plt.xlabel('Epoch', fontsize=15, weight='bold')
    plt.ylabel('Reconstruction Loss', fontsize=15, weight='bold')
    plt.legend(['Recon Train Loss'], fontsize=18)

    bwith = 1.5
    TK = plt.gca()
    TK.spines['bottom'].set_linewidth(bwith)
    TK.spines['left'].set_linewidth(bwith)
    TK.spines['top'].set_linewidth(bwith)
    TK.spines['right'].set_linewidth(bwith)

    plt.show()

    return Epoch_list, Recon_loss_list
