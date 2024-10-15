import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import imq_kernel
import os

def train_WAE(model, optimizer, dataloader, params, device, root='./'):
    model_name = params['model_name']
    num_epoch = params['num_epoch']
    sigma = params['sigma']
    MMD_lambda = params['MMD_lambda']
    pro_weight = params['properties_weight']

    folder_dir = os.path.join(root, model_name)
    if not os.path.isdir(folder_dir):
        os.mkdir(folder_dir)

    loss_recon = []
    loss_properties = []
    Epoch_list = []

    for epoch in range(num_epoch):
        total_loss = []
        total_recon = []
        total_MMD = []
        total_properties = []

        for i, data in enumerate(tqdm(dataloader)):
            x, y = data
            x, y = x.to(device), y.to(device)

            model.train()
            recon_x, z_tilde, pro_out = model(x)
            z = sigma * torch.randn(z_tilde.size()).to(device)

            recon_loss = F.binary_cross_entropy(recon_x, x, reduction='mean')
            properties_loss = F.mse_loss(pro_out, y, reduction='mean')

            MMD_loss = imq_kernel(z_tilde, z, h_dim=model.latent_space.out_features).to(device)
            MMD_loss = MMD_loss / x.size(0)
            
            loss = recon_loss + MMD_loss * MMD_lambda + properties_loss * pro_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())
            total_recon.append(recon_loss.item())
            total_MMD.append(MMD_loss.item())
            total_properties.append(properties_loss.item())

        avg_loss = sum(total_loss) / len(total_loss)
        avg_recon = sum(total_recon) / len(total_recon)
        avg_MMD = sum(total_MMD) / len(total_MMD)
        avg_pro = sum(total_properties) / len(total_properties)
        
        loss_recon.append(avg_recon)
        loss_properties.append(avg_pro)
        Epoch_list.append(epoch)

        print(f'[Epoch {epoch+1}/{num_epoch}] Loss: {avg_loss:.6f}, Recon: {avg_recon:.6f}, MMD: {avg_MMD:.6f}, Properties: {avg_pro:.6f}')
        
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(folder_dir, f'{model_name}_{epoch+1}.pth'))

    return loss_recon, loss_properties, Epoch_list
