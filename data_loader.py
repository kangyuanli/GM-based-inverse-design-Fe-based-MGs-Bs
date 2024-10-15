import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def load_data(composition_file, target_file, test_size=0.2, random_state=7):

    # Loads composition and target data, splits into training and test sets, and returns DataLoader objects

    with open(composition_file, 'r') as f:
        composition_data = f.readlines()

    with open(target_file, 'r') as f1:
        target_data = f1.readlines()

    Composition, Fraction, Bs = [], [], []
    for line in composition_data[1:]:
        s = line.split()
        composition_number = int(s[1])
        Composition.append(s[2:2+composition_number])
        Fraction.append(s[2+composition_number:2+2*composition_number])

    for line in target_data[1:]:
        s = line.split()
        Bs.append(float(s[0]))

    periodic_element_table = ['Fe', 'B', 'Si', 'P', 'C', 'Co', 'Nb', 'Ni', 'Mo', 'Zr', 'Ga', 'Al', 'Dy', 'Cu', 'Cr',
                              'Y', 'Nd', 'Hf', 'Ti', 'Tb', 'Ho', 'Ta', 'Er', 'Sn', 'W', 'Tm', 'Gd', 'Sm', 'V', 'Pr']
    
    element_index_list = get_element_index(Composition, periodic_element_table)
    composition_matrix = create_composition_matrix(element_index_list, Fraction, periodic_element_table)

    Bs_matrix = np.array(Bs).reshape(-1, 1)

    composition_tensor = torch.tensor(composition_matrix, dtype=torch.float32)
    Bs_tensor = torch.tensor(Bs_matrix, dtype=torch.float32)

    train_idx, test_idx = train_test_split(range(composition_matrix.shape[0]), test_size=test_size, random_state=random_state)

    train_data = TensorDataset(composition_tensor[train_idx], Bs_tensor[train_idx])
    test_data = TensorDataset(composition_tensor[test_idx], Bs_tensor[test_idx])

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False)

    return train_loader, test_loader

def get_element_index(element_list, periodic_element_table):
    # Maps elements from the composition list to their indices in the periodic element table.
    element_index_list = []
    for e_list in element_list:
        e_index_list = [periodic_element_table.index(e) for e in e_list if e in periodic_element_table]
        element_index_list.append(e_index_list)
    return element_index_list

def create_composition_matrix(element_index_list, Fraction, periodic_element_table):
    # Creates a composition matrix using element indices and their fractions for each sample.
    composition_matrix = np.zeros((len(element_index_list), len(periodic_element_table)))

    for sample_idx, index_list in enumerate(element_index_list):
        for element_idx, index in enumerate(index_list):
            composition_matrix[sample_idx][index] = float(Fraction[sample_idx][element_idx]) * 0.01

    return composition_matrix

