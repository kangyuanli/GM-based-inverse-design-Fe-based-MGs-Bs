def obj_func(geneinfo):
    """
    Objective function that takes a gene (latent feature vector) as input and returns the predicted property (Bs value) from the predictor model.
    """
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    root = './'

    model_dir = os.path.join(root, f'{params["model_name"]}/{params["model_name"]}_{params["num_epoch"]}.pth')
    model = WAE(in_features=30, latent_size=16).to(device)
    model.load_state_dict(torch.load(model_dir))
    model.eval()

    input_feature = torch.zeros((16))
    for i in range(len(geneinfo)):
        input_feature[i] = geneinfo[i]
    input_feature = input_feature.to(device)

    predict_pro = model.Predict(input_feature)
    predict_pro = predict_pro.cpu().detach().numpy()

    return 1 / predict_pro[0]


def Random_optimizer(search_number, iteration_number, wae):
    """
    Performs a random search using the WAE model to optimize the alloy composition and Bs value.
    The best composition and Bs value are saved in an Excel file.
    """
    import pandas as pd
    import numpy as np

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    periodic_element_table = [
        'Fe', 'B', 'Si', 'P', 'C', 'Co', 'Nb', 'Ni', 'Mo', 'Zr', 'Ga', 'Al',
        'Dy', 'Cu', 'Cr', 'Y', 'Nd', 'Hf', 'Ti', 'Tb', 'Ho', 'Ta', 'Er', 'Sn',
        'W', 'Tm', 'Gd', 'Sm', 'V', 'Pr'
    ]

    compositions_str_list = []
    bs_values_list = []

    for seed in range(search_number):
        torch.manual_seed(seed)
        Input = torch.zeros((iteration_number, 16))
        random_z = 8.0 * torch.randn_like(Input).to(device)

        recon_x = wae.decoder(random_z).cpu().detach().numpy()
        properties = wae.Predict(random_z).cpu().detach().numpy()

        bs_values = properties[:, 0]
        max_index = np.argmax(bs_values)

        best_bs_value = bs_values[max_index]
        best_composition = recon_x[max_index]
        best_composition[best_composition < 0.001] = 0
        composition_str = "".join([f"{periodic_element_table[j]}{best_composition[j]*100:.2f}" for j in range(len(periodic_element_table)) if best_composition[j] > 0])

        compositions_str_list.append(composition_str)
        bs_values_list.append(best_bs_value)

        print(f'Random Search {seed + 1}: Best Composition: {composition_str}, Best Bs value: {best_bs_value:.4f}')

    df = pd.DataFrame({'Composition': compositions_str_list, 'Bs_value': bs_values_list})
    df.to_excel('Random_optimizer_results.xlsx', index=False)
    print('Results saved to Random_optimizer_results.xlsx')

    return compositions_str_list, bs_values_list

def GA_optimization(feature_dimension):
    """
    Optimizes the latent space features using a Genetic Algorithm (GA) to maximize the Bs value of 
    the generated alloys. The best results are saved in an Excel file.
    """
    import numpy as np
    from sko.GA import GA  # Make sure you have installed `scikit-opt` package
    import pandas as pd

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    model = load_wae_model()  # Define load_wae_model to load the trained WAE model

    pre_d_list = []
    GA_composition_matrix = np.zeros((50, 30))

    periodic_element_table = get_periodic_element_table()

    compositions_str_list = []
    bs_values_list = []

    for i in range(50):
        random_z = 8.0 * torch.randn((2000, 16)).to(device)
        lb_list = random_z.min(dim=0).values.cpu().numpy()
        ub_list = random_z.max(dim=0).values.cpu().numpy()

        ga = GA(func=obj_func, n_dim=16, size_pop=20, max_iter=20, lb=lb_list, ub=ub_list, precision=1e-3)
        best_x_tensor = torch.tensor(ga.run()[0], dtype=torch.float32).to(device)

        pre_out_d = model.Predict(best_x_tensor)
        pre_out_com = model.decoder(best_x_tensor.unsqueeze(0))

        composition = pre_out_com.cpu().detach().numpy().flatten()
        pre_d = pre_out_d.cpu().detach().numpy()

        composition[composition < 0.001] = 0
        GA_composition_matrix[i] = composition

        bs_value = pre_d[0]
        composition_str = "".join([f"{periodic_element_table[j]}{composition[j]*100:.2f}" for j in range(len(periodic_element_table)) if composition[j] > 0])

        compositions_str_list.append(composition_str)
        bs_values_list.append(bs_value)

        print(f'GA Number {i + 1}: Composition: {composition_str}, Bs value: {bs_value:.4f}')

    df = pd.DataFrame({'Composition': compositions_str_list, 'Bs_value': bs_values_list})
    df.to_excel('GA_optimization_results.xlsx', index=False)
    print('Results saved to GA_optimization_results.xlsx')

    return pre_d_list, GA_composition_matrix


def PSO_optimization(feature_dimension):
    """
    Optimizes the latent space features using Particle Swarm Optimization (PSO) to maximize the Bs value 
    of the generated alloys. The best results are saved in an Excel file.
    """
    import numpy as np
    from sko.PSO import PSO  # Ensure `scikit-opt` package is installed
    import pandas as pd

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    model = load_wae_model()  # Define load_wae_model to load the trained WAE model

    pre_d_list = []
    PSO_composition_matrix = np.zeros((50, 30))

    periodic_element_table = get_periodic_element_table()

    compositions_str_list = []
    bs_values_list = []

    for i in range(50):
        random_z = 8.0 * torch.randn((2000, 16)).to(device)
        lb_list = random_z.min(dim=0).values.cpu().numpy()
        ub_list = random_z.max(dim=0).values.cpu().numpy()

        pso = PSO(func=obj_func, n_dim=feature_dimension, pop=20, max_iter=20, lb=lb_list, ub=ub_list)
        best_x_tensor = torch.tensor(pso.gbest_x, dtype=torch.float32).to(device)

        pre_out_d = model.Predict(best_x_tensor)
        pre_out_com = model.decoder(best_x_tensor.unsqueeze(0))

        composition = pre_out_com.cpu().detach().numpy().flatten()
        pre_d = pre_out_d.cpu().detach().numpy()

        composition[composition < 0.001] = 0
        PSO_composition_matrix[i] = composition

        bs_value = pre_d[0]
        composition_str = "".join([f"{periodic_element_table[j]}{composition[j]*100:.2f}" for j in range(len(periodic_element_table)) if composition[j] > 0])

        compositions_str_list.append(composition_str)
        bs_values_list.append(bs_value)

        print(f'PSO Number {i + 1}: Composition: {composition_str}, Bs value: {bs_value:.4f}')

    df = pd.DataFrame({'Composition': compositions_str_list, 'Bs_value': bs_values_list})
    df.to_excel('PSO_optimization_results.xlsx', index=False)
    print('Results saved to PSO_optimization_results.xlsx')

    return pre_d_list, PSO_composition_matrix

