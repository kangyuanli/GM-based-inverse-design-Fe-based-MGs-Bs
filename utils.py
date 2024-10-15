import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def imq_kernel(X, Y, h_dim):

    # Computes the Inverse Multiquadratic (IMQ) kernel for Maximum Mean Discrepancy (MMD) calculation.

    batch_size = X.size(0)
    norms_x = X.pow(2).sum(1, keepdim=True)
    prods_x = torch.mm(X, X.t())
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True)
    prods_y = torch.mm(Y, Y.t())
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x) + C / (C + dists_y)
        res1 = (1 - torch.eye(batch_size).to(X.device)) * res1
        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / batch_size
        stats += res1 - res2

    return stats


def existing_data_latent_space(model, composition_matrix_tensor, Tx_matrix_tensor, device, output_file='joint_WAE_data_latent_space.png'):

    # Visualizes the latent space of existing data using PCA in a Joint-WAE model.

    with torch.no_grad():
        test = torch.FloatTensor(composition_matrix_tensor[:]).to(device)
        x_reconst, z, pro_out = model(test)


    plt.figure(figsize=(8, 6))
    latent_space = z.detach().cpu().numpy()


    PCA_model = PCA(n_components=2, random_state=0)
    latent_space = PCA_model.fit_transform(latent_space)


    color_list = Tx_matrix_tensor.detach().cpu().numpy()
    global_vmin = color_list.min()
    global_vmax = color_list.max()

    plt.scatter(latent_space[:, 0], latent_space[:, 1], c=color_list, marker='D', s=20, cmap='coolwarm', vmin=global_vmin, vmax=global_vmax)

    plt.xlabel('PCA Dimension 1', fontsize=24, weight='bold')
    plt.ylabel('PCA Dimension 2', fontsize=24, weight='bold')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.xlim([-30, 30])
    plt.ylim([-30, 30])

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)

    bwith = 2  
    TK = plt.gca()  
    TK.spines['bottom'].set_linewidth(bwith)
    TK.spines['left'].set_linewidth(bwith)
    TK.spines['top'].set_linewidth(bwith)
    TK.spines['right'].set_linewidth(bwith)

    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.show()




def generated_latent_space(model, Random_sample, latent_size, device, global_vmin, global_vmax, output_file='joint_WAE_latent_space.png'):

     # Visualizes the latent space of generated data using PCA in a Joint-WAE model.

    with torch.no_grad():
        Input = torch.zeros((Random_sample, latent_size)).to(device)


        torch.manual_seed(8)
        sigma = 8.0
        random_z = sigma * torch.randn_like(Input).to(device)


        recon_x = model.decoder(random_z).cpu().detach().numpy()
        predict_pro = model.Predict(random_z).cpu().detach().numpy()


    latent_space = random_z.cpu().detach().numpy()
    PCA_model = PCA(n_components=2, random_state=0)
    latent_space = PCA_model.fit_transform(latent_space)

    plt.figure(figsize=(8, 6))

    color_list = predict_pro
    plt.scatter(latent_space[:, 0], latent_space[:, 1], c=color_list, marker='o', s=20, cmap='coolwarm', vmin=global_vmin, vmax=global_vmax)

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)

    plt.xlabel('PCA Dimension 1', fontsize=24, weight='bold')
    plt.ylabel('PCA Dimension 2', fontsize=24, weight='bold')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.xlim([-30, 30])
    plt.ylim([-30, 30])

    bwith = 2.0  
    TK = plt.gca()  
    TK.spines['bottom'].set_linewidth(bwith)
    TK.spines['left'].set_linewidth(bwith)
    TK.spines['top'].set_linewidth(bwith)
    TK.spines['right'].set_linewidth(bwith)

   
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.show()


def plot_tsne(original, generated):

    # Plots the t-SNE visualization comparing original and generated alloys.

    tsne = TSNE(n_components=2, random_state=0)
    all_data = np.concatenate([original, generated], axis=0)
    tsne_result = tsne.fit_transform(all_data)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_result[:len(original), 0], tsne_result[:len(original), 1], label='Existing alloys', alpha=0.6, c='red', edgecolor='k')
    plt.scatter(tsne_result[len(original):, 0], tsne_result[len(original):, 1], label='Generated alloys', alpha=0.6, c='blue', edgecolor='k')
    
    plt.xlabel('t-SNE Dimension 1', fontsize=24, weight='bold')
    plt.ylabel('t-SNE Dimension 2', fontsize=24, weight='bold')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.legend(loc='upper right',frameon=False, prop={'weight': 'bold', 'size': 15})
    
   
    #plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    
    
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    
    
    plt.tight_layout()
    plt.savefig('tsne_visualization.png', dpi=300)
    plt.show()



def plot_element_comparison(original, reconstructed, periodic_table):

    # Compares the element density between original and reconstructed alloys in a bar chart.
    
    original_count = np.count_nonzero(original, axis=0)
    reconstructed_count = np.count_nonzero(reconstructed, axis=0)
    
    
    original_freq = original_count / np.sum(original_count)
    reconstructed_freq = reconstructed_count / np.sum(reconstructed_count)
    
    plt.figure(figsize=(12, 6))
    width = 0.35
    ind = np.arange(len(periodic_table))
    
    plt.bar(ind - width/2, original_freq, width, label='Density of existing alloys', color='red', edgecolor='black')
    plt.bar(ind + width/2, reconstructed_freq, width, label='Density of Generated alloys',color='blue', edgecolor='black')
    
    plt.xlabel('Element', fontsize=18, weight='bold')
    plt.ylabel('Density of elements',fontsize=18, weight='bold')
    
    plt.xticks(ind, periodic_table, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)


    plt.gca().spines['top'].set_linewidth(1.5)
    plt.gca().spines['right'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    plt.gca().spines['left'].set_linewidth(1.5)


    plt.tight_layout()
    plt.savefig('element_comparison.png', dpi=300)
    plt.show()

def plot_property_distribution(original, generated, title):

    # Plots the property distribution of original and generated alloys using histogram and KDE.
    
    original = original[np.isfinite(original)]
    generated = generated[np.isfinite(generated)]

    
    original_hist, original_bins = np.histogram(original, bins=30, density=True)
    generated_hist, generated_bins = np.histogram(generated, bins=30, density=True)
    
    
    original_bin_centers = 0.5 * (original_bins[1:] + original_bins[:-1])
    generated_bin_centers = 0.5 * (generated_bins[1:] + generated_bins[:-1])
    
    
    original_kde = gaussian_kde(original, bw_method=0.3)
    generated_kde = gaussian_kde(generated, bw_method=0.3)
    
    
    x_min = min(min(original), min(generated))
    x_max = max(max(original), max(generated))
    x_points = np.linspace(x_min, x_max, 1000)
    
    
    original_density = original_kde(x_points)
    generated_density = generated_kde(x_points)
    
    
    plt.figure(figsize=(8, 6))
    plt.hist(original, bins=30, density=True, alpha=0.6, label='Existing alloys', color='red', edgecolor='black')
    plt.hist(generated, bins=30, density=True, alpha=0.6, label='Generated alloys', color='blue', edgecolor='black')
    
    
    plt.plot(x_points, original_density, color='red', linewidth=2)
    plt.plot(x_points, generated_density, color='blue', linewidth=2)
    
    plt.xlabel('Bs (T)', fontsize=24, weight='bold')
    plt.ylabel('Density of property', fontsize=24, weight='bold')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(frameon=False, prop={'weight': 'bold', 'size': 15})
    
    # 添加网格线
    #plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    
    # 设置边框宽度
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    
    # 保存高分辨率图像
    plt.tight_layout()
    plt.savefig('property_distribution.png', dpi=300)
    plt.show()


def plot_joint_wae_latent_space(joint_wae_latent_space, Bs_matrix_tensor, composition_matrix_tensor, element_names, element_indices, output_file='joint_wae_latent_space_elements_concentration.png'):

    # Visualizes the Joint-WAE latent space and element concentrations

    color_list = Bs_matrix_tensor.detach().cpu().numpy()

    
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.ravel()

    
    axes[0].scatter(joint_wae_latent_space[:, 0], joint_wae_latent_space[:, 1], c=color_list, marker='o', s=50, cmap='coolwarm')
    axes[0].set_xlabel('Dimension 1', fontsize=23, weight='bold')
    axes[0].set_ylabel('Dimension 2', fontsize=23, weight='bold')
    cbar = fig.colorbar(axes[0].collections[0], ax=axes[0])
    cbar.ax.tick_params(labelsize=19)

    
    for i, (element_name, element_index) in enumerate(zip(element_names, element_indices), start=1):
        element_concentration = composition_matrix_tensor[:, element_index].detach().cpu().numpy()
        sc = axes[i].scatter(joint_wae_latent_space[:, 0], joint_wae_latent_space[:, 1], c=element_concentration, marker='o', s=50, cmap='coolwarm')
        axes[i].set_xlabel('Dimension 1', fontsize=23, weight='bold')
        axes[i].set_ylabel('Dimension 2', fontsize=23, weight='bold')
        cbar = fig.colorbar(sc, ax=axes[i])
        cbar.ax.tick_params(labelsize=19)

    
    bwith = 2.0
    for ax in axes:
        ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['left'].set_linewidth(bwith)
        ax.spines['top'].set_linewidth(bwith)
        ax.spines['right'].set_linewidth(bwith)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.xaxis.set_tick_params(width=1.5)
        ax.yaxis.set_tick_params(width=1.5)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.show()
