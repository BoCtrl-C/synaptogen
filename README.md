# SynaptoGen

This repository contains the implementation of SynaptoGen, the model introduced in ["Optimizing Genetically-Driven Synaptogenesis"](https://arxiv.org/abs/2402.07242), along with the code used for its validation.

---

**Note**

The code provided in this repository is experimental and intended as a resource for readers interested in studying the methodology behind SynaptoGen and the experiments conducted. Please note that some of the terminology used in the code may differ from that used in the corresponding paper. A more polished version of the code will be released soon.

---

## Repository Content

Below is a brief description of the main files included in the repository:
- `dqn_custom_policies.py`: Re-implementation of classes from the Stable Baselines3 library to train custom models using the DQN algorithm.
- `eval.ipynb`: Code used to perform the simulations that produced the results presented in the paper.
- `genetic_rules.ipynb`: Implementation of our nDGE variant, used to generate co-expression data.
- `hyperparams.py`: Hyperparameters used for training models with the DQN algorithm.
- `models.py`: Synaptogen implementation.
- `snes.py`: Code used to generate genetic profiles based on SNES optimization.
- `sweep_analysis.ipynb`: Notebook used for analyzing the hyperparameter sweeps' results.
- `train.py`: Code for training SynaptoGen using gradient descent.
- `utils.py`: Various helper functions.