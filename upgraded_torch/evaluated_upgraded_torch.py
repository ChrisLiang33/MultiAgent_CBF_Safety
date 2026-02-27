import sys
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import core_upgraded_torch as core
import config

sys.dont_write_bytecode = True
np.set_printoptions(4)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', type=int, required=True)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--vis', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    return args

def evaluate_step_torch(cbf_network, action_network, s, g):
    # --- Phase 1: The Initial Guess ---
    x = s.unsqueeze(1) - s.unsqueeze(0)
    h, mask, indices = cbf_network(x, config.DIST_MIN_THRES, indices=None)
    
    # The Action Network's "First Instinct" (We detach it so we don't change the network weights)
    a = action_network(s, g, indices=indices).detach()
    
    # --- Phase 2: The Split-Second Imagination (Action Refinement) ---
    # We create our Delta Action and explicitly tell PyTorch to track its gradients
    a_res = torch.zeros_like(a, requires_grad=True)
    
    for loop_count in range(config.REFINE_LOOPS):
        a_opt = a + a_res
        
        # 1. Simulate the future
        dsdt = core.dynamics_torch(s, a_opt)
        s_next = s + dsdt * config.TIME_STEP
        x_next = s_next.unsqueeze(1) - s_next.unsqueeze(0)
        h_next, mask_next, _ = cbf_network(x_next, config.DIST_MIN_THRES, indices=indices)
        
        # 2. Check the Physics Constraints
        deriv = h_next - h + config.TIME_STEP * config.ALPHA_CBF * h
        deriv = deriv * mask * mask_next  # Apply sensor blindfolds
        
        # 3. Calculate Error (Hinge Loss)
        error = torch.sum(F.relu(-deriv), dim=1)
        
        # 4. Calculate Gradient w.r.t the Steering Wheel (a_res)
        # allow_unused=True prevents a crash if the error is exactly 0
        grad = torch.autograd.grad(error.sum(), a_res, allow_unused=True)[0]
        
        # If the gradient is None (perfectly safe), just treat it as zero
        if grad is None:
            grad = torch.zeros_like(a_res)
        
        # 5. Push the steering wheel away from danger!
        # Detach drops the gradient history, requires_grad_(True) prepares it for the next loop
        a_res = (a_res - config.REFINE_LEARNING_RATE * grad).detach().requires_grad_(True)

    # --- Phase 3: The Final Grade ---
    # The final safe action! We detach it completely from the imagination loop
    a_opt = (a + a_res).detach()
    
    # Run physics one last time for real
    dsdt = core.dynamics_torch(s, a_opt)
    s_next = s + dsdt * config.TIME_STEP
    x_next = s_next.unsqueeze(1) - s_next.unsqueeze(0)
    h_next, mask_next, _ = cbf_network(x_next, config.DIST_MIN_THRES, indices=indices)

    # Grade the newly optimized state (Notice eps=[0,0] exactly like the original code)
    loss_dang, loss_safe, acc_dang, acc_safe = core.loss_barrier_torch(
        h_next, s_next, config.DIST_MIN_THRES, config.TIME_TO_COLLISION, config.TOP_K, indices=indices, eps=[0, 0])

    loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv = core.loss_derivatives_torch(
        cbf_network, s_next, a_opt, h_next, config.DIST_MIN_THRES, config.TIME_TO_COLLISION, config.ALPHA_CBF, config.TOP_K, config.TIME_STEP, indices=indices)

    # Action loss is evaluated against the ORIGINAL action instinct, not the optimized one!
    loss_action = core.loss_action_torch(s, g, a)

    loss_list = [loss_dang, loss_safe, loss_dang_deriv, loss_safe_deriv, loss_action]
    acc_list = [acc_dang, acc_safe, acc_dang_deriv, acc_safe_deriv]

    return s, g, a_opt, loss_list, acc_list

def print_accuracy(accuracy_lists):
    acc = np.array(accuracy_lists)
    acc_list = []
    for i in range(acc.shape[1]):
        valid_scores = acc[acc[:, i] >= 0, i]
        if len(valid_scores) > 0:
            acc_list.append(np.mean(valid_scores))
        else:
            acc_list.append(-1.0)
    print('Accuracy: {}'.format(acc_list))


def render_init():
    fig = plt.figure(figsize=(9, 4))
    return fig

def main_torch():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize and Load Models
    cbf_network = core.NetworkCBF(top_k=config.TOP_K, obs_radius=config.OBS_RADIUS).to(device)
    action_network = core.NetworkAction(top_k=config.TOP_K, obs_radius=config.OBS_RADIUS).to(device)
    
    if args.model_path:
        checkpoint = torch.load(args.model_path)
        cbf_network.load_state_dict(checkpoint['cbf'])
        action_network.load_state_dict(checkpoint['action'])
        
    # Set networks to evaluation mode (disables training-specific layers like Dropout)
    cbf_network.eval()
    action_network.eval()

    state_gain = torch.tensor(np.eye(2, 4) + np.eye(2, 4, k=2) * np.sqrt(3), dtype=torch.float32, device=device)

    safety_ratios_epoch, safety_ratios_epoch_lqr = [], []
    dist_errors, init_dist_errors, accuracy_lists = [], [], []
    safety_reward, dist_reward, safety_reward_baseline, dist_reward_baseline = [], [], [], []

    if args.vis:
        plt.ion()
        plt.close()
        fig = render_init()

    # 2. The Evaluation Loop
    for istep in range(config.EVALUATE_STEPS):
        start_time = time.time()
        safety_info, safety_info_baseline = [], []
        
        # Spawn agents further apart for safe testing
        s_ori, g_ori = core.generate_data_torch(args.num_agents, config.DIST_MIN_THRES * 1.5)
        s_ori, g_ori = s_ori.to(device), g_ori.to(device)

        s, g = s_ori.clone(), g_ori.clone()
        init_dist_errors.append(torch.mean(torch.norm(s[:, :2] - g, dim=1)).item())
        
        s_np_ours, s_np_lqr = [], []
        safety_ours, safety_lqr = [], []

        # --- THE AI RUN ---
        for i in range(config.INNER_LOOPS):
            # Run test-time optimization
            _, _, a_opt, loss_list, acc_list = evaluate_step_torch(cbf_network, action_network, s, g)
            
            # Physics Step
            s = (s + torch.cat([s[:, 2:], a_opt], dim=1) * config.TIME_STEP).detach()
            
            # Save state to CPU for Matplotlib
            s_np = s.cpu().numpy()
            g_np = g.cpu().numpy()
            s_np_ours.append(s_np)
            
            # Calculate safety
            safety_mask = core.ttc_dangerous_mask_np(s_np, config.DIST_MIN_CHECK, config.TIME_TO_COLLISION_CHECK)
            safety_ratio_array = 1 - np.mean(safety_mask, axis=1)
            safety_ours.append(safety_ratio_array)
            safety_info.append((safety_ratio_array == 1).astype(np.float32).reshape((1, -1)))
            
            safety_ratio_val = np.mean(safety_ratio_array == 1)
            safety_ratios_epoch.append(safety_ratio_val)
            accuracy_lists.append([acc.item() for acc in acc_list])

            # End-of-episode logic & LQR Hack
            dist_to_goal = np.linalg.norm(s_np[:, :2] - g_np, axis=1)
            if args.vis:
                if np.amax(dist_to_goal) < config.DIST_MIN_CHECK / 3:
                    time.sleep(1)
                    break
                if np.mean(dist_to_goal) < config.DIST_MIN_CHECK / 2:
                    # Switch to LQR to avoid target jitter
                    s_ref = torch.cat([s[:, :2] - g, s[:, 2:]], dim=1)
                    a_lqr = -torch.matmul(s_ref, state_gain.T)
                    s = (s + torch.cat([s[:, 2:], a_lqr], dim=1) * config.TIME_STEP).detach()
            else:
                if np.mean(dist_to_goal) < config.DIST_MIN_CHECK:
                    break

        dist_errors.append(np.mean(np.linalg.norm(s.cpu().numpy()[:, :2] - g.cpu().numpy(), axis=1)))
        safety_reward.append(np.mean(np.sum(np.concatenate(safety_info, axis=0) - 1, axis=0)))
        dist_reward.append(np.mean((np.linalg.norm(s.cpu().numpy()[:, :2] - g.cpu().numpy(), axis=1) < 0.2).astype(np.float32) * 10))

        # --- THE LQR BASELINE RUN ---
        s_lqr, g_lqr = s_ori.clone(), g_ori.clone()
        for i in range(config.INNER_LOOPS):
            s_ref_lqr = torch.cat([s_lqr[:, :2] - g_lqr, s_lqr[:, 2:]], dim=1)
            a_lqr = -torch.matmul(s_ref_lqr, state_gain.T)
            s_lqr = (s_lqr + torch.cat([s_lqr[:, 2:], a_lqr], dim=1) * config.TIME_STEP).detach()
            
            s_np_baseline = s_lqr.cpu().numpy()
            g_np_baseline = g_lqr.cpu().numpy()
            s_np_lqr.append(s_np_baseline)
            
            safety_mask_lqr = core.ttc_dangerous_mask_np(s_np_baseline, config.DIST_MIN_CHECK, config.TIME_TO_COLLISION_CHECK)
            safety_ratio_array_lqr = 1 - np.mean(safety_mask_lqr, axis=1)
            safety_lqr.append(safety_ratio_array_lqr)
            safety_info_baseline.append((safety_ratio_array_lqr == 1).astype(np.float32).reshape((1, -1)))
            
            safety_ratios_epoch_lqr.append(np.mean(safety_ratio_array_lqr == 1))
            
            if np.mean(np.linalg.norm(s_np_baseline[:, :2] - g_np_baseline, axis=1)) < config.DIST_MIN_CHECK / 3:
                break

        safety_reward_baseline.append(np.mean(np.sum(np.concatenate(safety_info_baseline, axis=0) - 1, axis=0)))
        dist_reward_baseline.append(np.mean((np.linalg.norm(s_lqr.cpu().numpy()[:, :2] - g_lqr.cpu().numpy(), axis=1) < 0.2).astype(np.float32) * 10))

        # --- VISUALIZATION RENDERING ---
        if args.vis:
            s_ori_np = s_ori.cpu().numpy()
            g_np = g.cpu().numpy()
            vis_range = max(1, np.amax(np.abs(s_ori_np[:, :2])))
            agent_size = 100 / vis_range ** 2
            g_np = g_np / vis_range
            
            for j in range(max(len(s_np_ours), len(s_np_lqr))):
                plt.clf()
                
                # Plot AI (Left)
                plt.subplot(121)
                j_ours = min(j, len(s_np_ours)-1)
                s_np_plot = s_np_ours[j_ours] / vis_range
                plt.scatter(s_np_plot[:, 0], s_np_plot[:, 1], color='darkorange', s=agent_size, label='Agent', alpha=0.6)
                plt.scatter(g_np[:, 0], g_np[:, 1], color='deepskyblue', s=agent_size, label='Target', alpha=0.6)
                safety = np.squeeze(safety_ours[j_ours])
                plt.scatter(s_np_plot[safety<1, 0], s_np_plot[safety<1, 1], color='red', s=agent_size, label='Collision', alpha=0.9)
                plt.xlim(-0.5, 1.5)
                plt.ylim(-0.5, 1.5)
                ax = plt.gca()
                for side in ax.spines.keys():
                    ax.spines[side].set_linewidth(2)
                    ax.spines[side].set_color('grey')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                plt.legend(loc='upper right', fontsize=14)
                plt.title('Ours: Safety Rate = {:.3f}'.format(np.mean(safety_ratios_epoch)), fontsize=14)

                # Plot LQR Baseline (Right)
                plt.subplot(122)
                j_lqr = min(j, len(s_np_lqr)-1)
                s_np_plot = s_np_lqr[j_lqr] / vis_range
                plt.scatter(s_np_plot[:, 0], s_np_plot[:, 1], color='darkorange', s=agent_size, label='Agent', alpha=0.6)
                plt.scatter(g_np[:, 0], g_np[:, 1], color='deepskyblue', s=agent_size, label='Target', alpha=0.6)
                safety = np.squeeze(safety_lqr[j_lqr])
                plt.scatter(s_np_plot[safety<1, 0], s_np_plot[safety<1, 1], color='red', s=agent_size, label='Collision', alpha=0.9)
                plt.xlim(-0.5, 1.5)
                plt.ylim(-0.5, 1.5)
                ax = plt.gca()
                for side in ax.spines.keys():
                    ax.spines[side].set_linewidth(2)
                    ax.spines[side].set_color('grey')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                plt.legend(loc='upper right', fontsize=14)
                plt.title('LQR: Safety Rate = {:.3f}'.format(np.mean(safety_ratios_epoch_lqr)), fontsize=14)

                fig.canvas.draw()
                plt.pause(0.01)
            plt.clf()

        end_time = time.time()
        print('Evaluation Step: {} | {}, Time: {:.4f}'.format(istep + 1, config.EVALUATE_STEPS, end_time - start_time))

    # --- FINAL REPORT ---
    print_accuracy(accuracy_lists)
    print('Distance Error (Final | Initial): {:.4f} | {:.4f}'.format(np.mean(dist_errors), np.mean(init_dist_errors)))
    print('Mean Safety Ratio (Learning | LQR): {:.4f} | {:.4f}'.format(np.mean(safety_ratios_epoch), np.mean(safety_ratios_epoch_lqr)))
    print('Reward Safety (Learning | LQR): {:.4f} | {:.4f}, Reward Distance: {:.4f} | {:.4f}'.format(
        np.mean(safety_reward), np.mean(safety_reward_baseline), 
        np.mean(dist_reward), np.mean(dist_reward_baseline)))

if __name__ == '__main__':
    main_torch()