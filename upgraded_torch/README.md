# Multi-Agent Safe Control via Neural Control Barrier Functions (PyTorch)



This repository contains a fully modernized, GPU-accelerated PyTorch implementation for learning decentralized, safe multi-agent control. It uses a dual-network architecture to generate trajectories while strictly enforcing collision avoidance through Control Barrier Functions (CBFs).

Originally based on a TensorFlow 1.x static-graph architecture, this codebase has been completely overhauled to utilize PyTorch's dynamic execution, native auto-differentiation, and object-oriented `nn.Module` classes.

## 🗂️ Project Structure

* `config.py`: The master control panel. Contains all hyperparameters for the physics engine, physical distances, time-to-collision constraints, and learning rates.
* `core_upgraded_torch.py`: The math and model engine. Contains the environment generation, physics calculations, safety/action loss functions, and the neural network architectures (`NetworkCBF` and `NetworkAction`).
* `train_upgraded_torch.py`: The main training loop. Handles gradient accumulation, separated network updates (ping-pong scheduling), and baseline (LQR) comparisons.
* `evaluate_torch.py`: The testing and rendering engine. Implements test-time Action Refinement via gradient descent and renders a side-by-side Matplotlib comparison of the AI vs. the baseline.

## ⚙️ Installation

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install torch numpy matplotlib

1. Training the Model

To train the neural networks from scratch, run the training script and specify the number of agents and the target GPU.
Bash

python train_upgraded_torch.py --num_agents 8 --gpu 0

(Note: If no GPU is detected, the script will automatically fall back to CPU execution).

The script will output loss and accuracy metrics to the console. Model checkpoints (.pth files) will be automatically saved to a models/ directory according to the SAVE_STEPS parameter in config.py.
2. Evaluating and Visualizing

To evaluate a trained model and watch the agents navigate the environment in real-time, use the evaluation script. Pass the --vis 1 flag to enable the Matplotlib rendering engine.

python evaluate_torch.py --num_agents 8 --model_path models/model_iter_69999.pth --vis 1 --gpu 0


