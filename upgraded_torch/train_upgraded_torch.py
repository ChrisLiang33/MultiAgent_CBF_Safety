import sys
import os
import argparse
import numpy as np
import torch
import config
import core_upgraded_torch as core

sys.dont_write_bytecode = True
np.set_printoptions(4)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', type=int, required=True)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    return args

def count_accuracy(accuracy_lists):
    acc = np.array(accuracy_lists)
    acc_list = []
    for i in range(acc.shape[1]):
        valid_scores = acc[acc[:, i] >= 0, i]
        if len(valid_scores) > 0:
            acc_list.append(np.mean(valid_scores))
        else:
            acc_list.append(-1.0)
    return acc_list

def main_torch():
    args = parse_args()
    
    # 1. Hardware & Setup
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Initialize our modern Object-Oriented Networks
    cbf_network = core.NetworkCBF(top_k=config.TOP_K, obs_radius=config.OBSERVATION_RADIUS).to(device)
    action_network = core.NetworkAction(top_k=config.TOP_K, obs_radius=config.OBSERVATION_RADIUS).to(device)
    
    # Initialize separate optimizers WITH weight decay
    optimizer_h = torch.optim.Adam(cbf_network.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    optimizer_a = torch.optim.Adam(action_network.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # Transfer Learning (Loading a saved brain)
    if args.model_path:
        checkpoint = torch.load(args.model_path)
        cbf_network.load_state_dict(checkpoint['cbf'])
        action_network.load_state_dict(checkpoint['action'])

    # The Expert LQR Matrix (Moved to GPU)
    state_gain = torch.tensor(np.eye(2, 4) + np.eye(2, 4, k=2) * np.sqrt(3), dtype=torch.float32, device=device)

    # 2. THE TRAINING LOOP
    for istep in range(config.TRAIN_STEPS):
        # Generate new map using our PyTorch generator
        s, g = core.generate_data_torch(args.num_agents, config.DIST_MIN_THRES)
        s, g = s.to(device), g.to(device)
        
        s_lqr, g_lqr = s.clone(), g.clone()
        
        # Zero out the gradients at the start of the episode
        optimizer_h.zero_grad()
        optimizer_a.zero_grad()
        
        loss_lists, acc_lists = [], []

        # --- THE AI INNER LOOP (Simulation) ---
        for i in range(config.INNER_LOOPS):
            # 1. Forward Pass & Loss Calculation
            a, loss_list, total_loss, acc_list = core.compute_loss_torch(cbf_network, action_network, s, g, config)
            
            # 2. Accumulate Gradients! 
            (total_loss / config.INNER_LOOPS).backward()
            
            # 3. Exploration Noise
            if torch.rand(1).item() < config.ADD_NOISE_PROB:
                a = a + torch.randn_like(a) * config.NOISE_SCALE
                
            # 4. The Physics Engine (Crucial .detach() added here!)
            s = (s + torch.cat([s[:, 2:], a], dim=1) * config.TIME_STEP).detach()
            
            # Fixed variable shadowing here
            loss_lists.append([l_val.item() for l_val in loss_list])
            acc_lists.append([acc_val.item() for acc_val in acc_list])
            
            # Early Stopping
            if torch.mean(torch.norm(s[:, :2] - g, dim=1)) < config.DIST_MIN_CHECK:
                break

        # --- THE DUMMY BASELINE (LQR) ---
        for i in range(config.INNER_LOOPS):
            s_ref_lqr = torch.cat([s_lqr[:, :2] - g_lqr, s_lqr[:, 2:]], dim=1)
            a_lqr = -torch.matmul(s_ref_lqr, state_gain.T)
            s_lqr = s_lqr + torch.cat([s_lqr[:, 2:], a_lqr], dim=1) * config.TIME_STEP
            
            if torch.mean(torch.norm(s_lqr[:, :2] - g_lqr, dim=1)) < config.DIST_MIN_CHECK:
                break

        # --- THE PING-PONG OPTIMIZATION STEP ---
        if (istep // 10) % 2 == 0:
            optimizer_h.step()
        else:
            optimizer_a.step()

        # --- LOGGING & SAVING ---
        if istep % config.DISPLAY_STEPS == 0:
            print(f'Step: {istep}, Loss: {np.mean(loss_lists, axis=0)}, Accuracy: {count_accuracy(acc_lists)}')

        if istep % config.SAVE_STEPS == 0 or istep + 1 == config.TRAIN_STEPS:
            os.makedirs('models', exist_ok=True)
            torch.save({
                'cbf': cbf_network.state_dict(),
                'action': action_network.state_dict()
            }, f'models/model_iter_{istep}.pth')

if __name__ == '__main__':
    main_torch()