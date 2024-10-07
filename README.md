# PyTorch Projects

Welcome to my PyTorch projects repository! This is a collection of various experiments and models I've worked on using PyTorch, covering different areas of machine learning. Each subdirectory contains different types of projects and examples, ranging from Artificial Neural Networks (ANNs) to Transfer Learning.

Feel free to explore the code and learn from the commented explanations throughout the notebooks. I'll continue to add more details and updates to this repository over time, so stay tuned!

## Contents
- [Repository Structure](#repository-structure)
  - [ANN](#1-ann)
  - [Autoencoders](#2-autoencoders)
  - [Metaparameters](#3-metaparameters)
  - [Transfer Learning](#4-transfer-learning)
  - [Weights](#5-weights)
- [Future Plans](#future-plans)
- [How to Use](#how-to-use)
- [Contribution](#contribution)
- [License](#license)

## Repository Structure

### 1. ANN
This folder contains experiments related to Artificial Neural Networks (ANNs). These notebooks explore different approaches to understanding and optimizing ANNs.

- **Experiments with Hyperparameters**: This subfolder contains various experiments to understand the impact of different hyperparameters on ANN performance.
  - `PyTorch_ANN_BatchSize.ipynb`: Experiment on how varying batch sizes affect the model performance.
  - `PyTorch_ANN_BreastCancer.ipynb`: Using a breast cancer dataset to build and train an ANN, exploring hyperparameters specific to this problem.
  - `PyTorch_ANN_Depth_Breadth.ipynb`: Comparing the effects of model depth versus width in ANNs.
  - `PyTorch_ANN_Learning_Rates.ipynb`: Experiment to determine the influence of different learning rates.
  - `PyTorch_ANN_MeasurePerformance_MNIST.ipynb`: Evaluating ANN performance on MNIST using different metrics.
  - `PyTorch_ANN_Multilayer_BatchNormed.ipynb`: Experimenting with batch normalization in multilayer ANNs.
  - `PyTorch_FFN_Optimizer_Type_MNIST.ipynb`: Comparing different optimizers (e.g., Adam, SGD) for an ANN on MNIST data.

- **MNIST Dataset**: Examples using the MNIST dataset to demonstrate different ANN implementations.
  - `PyTorch_ANN_MNIST.ipynb`: A simple ANN model trained on the MNIST dataset.

### 2. Autoencoders
This folder includes experiments with autoencoders, exploring their utility in various scenarios:

- `PyTorch_autoenc_denoisingMNIST.ipynb`: An experiment on how a denoising autoencoder can reconstruct images corrupted by noise.
- `PyTorch_autoenc_occlusion.ipynb`: Using an autoencoder to handle occluded inputs and attempt to reconstruct the missing parts.

### 3. Metaparameters
This folder explores various metaparameters and their effects across different models. These projects are intended to experiment with hyperparameter tuning and other configuration choices that affect model training and outcomes.

- `PyTorch_metaparam_ActivationComparisons_Relu.ipynb`: A comparative study on different activation functions, focusing on ReLU and its variants.
- `PyTorch_Metaparams_OptimizersTest.ipynb`: An exploration of different optimizers and their influence on the model’s convergence and performance.
- `PyTorch_Metaparams_WineSugar.ipynb`: Using a dataset related to wine sugar content to understand hyperparameter interactions.
- `PyTorch_MetaParams.ipynb`: General experiments on various metaparameters in model training.

### 4. Transfer Learning
Transfer learning examples, where pre-trained models are used and fine-tuned for new tasks:

- `PyTorch_TransferLearning_EMNIST.ipynb`: Using transfer learning to adapt a pre-trained model to classify the EMNIST dataset.
- `PyTorch_TransferLearning_resnetToSTL.ipynb`: An application of ResNet on the STL dataset, using transfer learning to reduce training time and improve model performance.

### 5. Weights
This folder contains projects related to understanding the effect of different weight initialization methods, which are crucial for model convergence:

- `PyTorch_Weight_Histograms.ipynb`: Visualizing the distribution of weights during training to understand model stability and performance.
- `PyTorch_Weights_Xavier_Kaiming_ipynb.ipynb`: A comparison of Xavier and Kaiming weight initialization techniques and their impact on training dynamics.

## Future Plans
I plan to add more information and detailed explanations about the projects and the various experiments. These include:
- Expanding the current explanations in the notebooks.
- Adding more datasets and training experiments.
- Providing a more extensive discussion on the results obtained.

If you're interested in specific areas or have questions, feel free to open an issue or contribute to the repository!

## How to Use
1. Clone this repository to your local machine:
   ```sh
   git clone https://github.com/username/PyTorch-Projects.git

2. Navigate to the directory of interest and open the Jupyter notebooks (.ipynb files) to explore the code and the comments.

3. Requirements:

- PyTorch
- Jupyter Notebook
- Python 3.7+

# Contribution

This is a work in progress, and I am continuously adding new projects and information. Contributions, suggestions, and discussions are highly appreciated! Feel free to fork the repository and make your changes.

# License

This project is open-source, feel free to use the code for learning purposes. Make sure to cite or reference this repository if you use it for your work.
