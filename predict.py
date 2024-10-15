import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import metrics
from torch.utils.data import DataLoader, TensorDataset
from models import WAE  # Import your model class from models.py
from data_loader import get_element_index, create_composition_matrix  # Import helper functions

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
root = './'

def main():
    latent_size = 16 
    in_features = 30 
    
    # Load your saved model
    params = {
        'model_name': '',  # Adjust your model name if necessary
        'num_epoch': 400,
    }
    
    model_dir = os.path.join(root, f'{params["model_name"]}/{params["model_name"]}_{params["num_epoch"]}.pth')
    model = WAE(in_features, latent_size).to(device)
    model.load_state_dict(torch.load(model_dir))
    model.eval()

    # Load composition and target data
    f = open('','r')
    composition_lines = f.readlines()
    f.close()

    f1 = open('','r')
    target_lines = f1.readlines()
    f1.close()

    Composition, Fraction, Bs = [], [], []

    for lines in composition_lines[1:]:
        s = lines.split()
        composition_number = int(s[1])
        Composition.append(s[2:2+composition_number])
        Fraction.append(s[2+composition_number:2+2*composition_number])

    for lines in target_lines[1:]:
        s = lines.split()
        Bs.append(float(s[0]))

    periodic_element_table = ['Fe', 'B', 'Si', 'P', 'C', 'Co', 'Nb', 'Ni', 'Mo', 'Zr', 'Ga', 'Al',
                              'Dy', 'Cu', 'Cr', 'Y', 'Nd', 'Hf', 'Ti', 'Tb', 'Ho', 'Ta', 'Er', 'Sn',
                              'W', 'Tm', 'Gd', 'Sm', 'V', 'Pr']

    element_index_list = get_element_index(Composition, periodic_element_table)
    composition_matrix = create_composition_matrix(element_index_list, Fraction, periodic_element_table)

    Bs_matrix = np.array(Bs).reshape(-1, 1)

    composition_matrix_tensor = torch.from_numpy(composition_matrix).float()
    Bs_matrix_tensor = torch.from_numpy(Bs_matrix).float()

    # Split data into training and testing sets
    train_size = int(0.8 * len(composition_matrix_tensor))
    train_data, test_data = torch.utils.data.random_split(
        TensorDataset(composition_matrix_tensor, Bs_matrix_tensor),
        [train_size, len(composition_matrix_tensor) - train_size]
    )

    train_loader = DataLoader(train_data, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False)

    # Predictions
    with torch.no_grad():
        train_x, train_y = next(iter(train_loader))
        test_x, test_y = next(iter(test_loader))
        
        # Perform prediction on train data
        z_train = model.encoder(train_x.to(device))
        recon_train_x = model.decoder(z_train).cpu().detach().numpy()
        pred_train_y = model.Predict(z_train).cpu().detach().numpy()

        # Perform prediction on test data
        z_test = model.encoder(test_x.to(device))
        recon_test_x = model.decoder(z_test).cpu().detach().numpy()
        pred_test_y = model.Predict(z_test).cpu().detach().numpy()

    # Calculate R² and RMSE for test data
    r2 = metrics.r2_score(test_y.cpu().detach().numpy(), pred_test_y)
    rmse = metrics.mean_squared_error(test_y.cpu().detach().numpy(), pred_test_y) ** 0.5

    print(f'Test Pro R²: {r2}')
    print(f'Test Pro RMSE: {rmse}')

    # Visualize results
    plt.figure()
    plt.scatter(train_y.cpu().detach().numpy(), pred_train_y, c='darkblue', label='Train')
    plt.scatter(test_y.cpu().detach().numpy(), pred_test_y, c='darkred', label='Test')
    plt.plot([min(train_y.min(), test_y.min()), max(train_y.max(), test_y.max())],
             [min(train_y.min(), test_y.min()), max(train_y.max(), test_y.max())], c='r', linestyle='--')
    plt.xlabel('Actual Bs (T)', fontsize=18, weight='bold')
    plt.ylabel('Predicted Bs (T)', fontsize=18, weight='bold')
    plt.legend(frameon=False, fontsize=17)
    plt.grid(True)
    plt.savefig('WAE_properties_regression.png', bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()
