# Thesis
This repository contains the code to reproduce my thesis work on Latent Neural Cellular Automata (LNCA).

## Overview
This project had the objective of reducing the computational requirements (the minimum VRAM and the training/inference latency) of Neural Cellular Automata (NCA) while minimizing the impact on the raw performance of such an optimization.

To achieve this goal we proposed a new architecture easily extendable to many NCA models, based on the use of an Autoencoder (AE) to transfer the computation to a lower dimensional manifold tailored to the task.

We then performed extensive tests against the State-of-the-Art to check the goodness of this solution.

To have a deeper understanding of this architecture and the experiments performed, I invite you to read the [thesis article](Presentation.pptx) (TBD) or the [presentation](Presentation.pptx) attached to the repository.

## Repository Structure
This repository is composed as follows:
* **Dataset**, contains the dataset to train and test the models (*not included*)
* **LatentCA**, contains the architecture and the trainer for the LNCA model (for both its variants)
* **NAFNet**, contains the architecture and the trainer for the [NAFNet](https://arxiv.org/abs/2204.04676) model (re-implemented in *TensorFlow*)
* **Notebooks**, contains an exploration notebook to visualize the results of the models
* **Restormer**, contains the architecture and the trainer for the [Restormer](https://arxiv.org/abs/2111.09881) model (re-implemented in *TensorFlow*)
* **Results**, contains the results of the training procedure and the tests of all models (*not included*)
* **Utils**, contains the utilities and the global trainer script of the repository
* **ViTCA**, contains the architecture and the trainer for the [ViTCA](https://arxiv.org/abs/2211.01233) model (re-implemented in *TensorFlow*)

## Use
This is a brief guide to use this repository:
1. Download the repository and install the requirements
```
git clone https://github.com/Menta99/Thesis
cd Thesis
pip install -r requirements.txt
```
2. Create the ```Dataset``` and the ```Results``` folders and download in the first the desired datasets (synthetic datasets should contain only the images, real datasets should contain 2 *sub-folders* named ```GT``` and ```DAMAGED```, containing the ground truths and the corrupted versions, respectively) 
3. Go into the ```Utils``` folder, customize the ```trainer.py``` script (change hyperparameters if you want, the defaults are the ones used for the experiments), and run the script (if interrupted the training resumes from the last completed epoch)
4. Go into the ```Notebooks``` folder and run the ```exploration.ipynb``` notebook to visualize the results (contains metric data-frames, training plots, latent space embeddings, and visual results)

## Contributors
* **Matteo Matteucci** (*Advisor*)
* **Alberto Archetti** (*Co-Advisor*)
