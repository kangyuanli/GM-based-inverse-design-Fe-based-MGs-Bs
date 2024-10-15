
# Generative-model-based-inverse-design-of-Fe-based-metallic-glasses-with-high-Bs

This repository contains the implementation of a generative model-based inverse design approach for discovering Fe-based metallic glasses (MGs) with high saturation magnetic flux density (Bs). The repository leverages machine learning models to explore the vast composition space of metallic glasses, helping to discover new materials without manual intervention.

## File Descriptions:

- **WAE_model.py**: Defines the Wasserstein Autoencoder model, which is designed to learn latent space representations of Fe-based MG compositions and predict the magnetic property (Bs).
- **VAE_model.py**: Defines the Variational Autoencoder (VAE) model and includes the training process for this model.
- **utils.py**: Contains utility functions used across the project, including data preprocessing and plotting.
- **train.py**: Script to train the Wasserstein Autoencoder.
- **predict.py**: Script for using trained models to generate new compositions and predict Bs values based on the latent space representations.
- **Optimizer.py**: Implements different optimization techniques (Genetic Algorithm, Particle Swarm Optimization, and Random Search) to explore the latent space and discover new compositions with desired properties.
- **main.py**: Serves as the main entry point to run various experiments.
- **data_loader.py**: Handles the loading and preprocessing of the composition and property datasets (such as `Composition_feature.txt` and `Bs_target.txt`), preparing the data for input into the models.

## Installation

To install the required dependencies, use the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model:
Run `train.py` to train the Wasserstein Autoencoder model:

```bash
python train.py
```

### Predicting Bs Values:
Run `predict.py` to generate new compositions and predict Bs values from the latent space:

```bash
python predict.py
```

### Optimization:
Run the following command to explore the latent space using optimization algorithms (e.g., Genetic Algorithm):

```bash
python Optimizer.py --method GA
```

### Data

Make sure the following input data files are present in the working directory:
- **Composition_feature.txt**: Contains the composition data of metallic glasses.
- **Bs_target.txt**: Contains the corresponding Bs values for the compositions.

## License

This project is licensed under the MIT License.

