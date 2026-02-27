import tensorflow as tf
import core_upgraded_tf as core
import config
import sys
sys.dont_write_bytecode = True

import os
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import core
import config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', type=int, required=True)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--vis', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    return args

@tf.function
def evaluate_step_tf(cbf_network, action_network, s, g):
    # --- Phase 1: The Initial Guess ---
    x = tf.expand_dims(s, 1) - tf.expand_dims(s, 0)
    h, mask, indices = cbf_network(x, config.DIST_MIN_THRES, indices=None)
    
    a = tf.stop_gradient(action_network(s, g, indices=indices))
    
    # --- Phase 2: The Split-Second Imagination (Action Refinement) ---
    a_res = tf.zeros_like(a)
    
    # AutoGraph converts this cleanly into a highly optimized C++ loop
    for _ in tf.range(config.REFINE_LOOPS):
        with tf.GradientTape() as tape:
            tape.watch(a_res)
            a_opt = a + a_res
            
            # Simulate the future
            dsdt = core.dynamics_tf(s, a_opt)
            s_next = s + dsdt * config.TIME_STEP
            x_next = tf.expand_dims(s_next, 1) - tf.expand_dims(s_next, 0)
            h_next, mask_next, _ = cbf_network(x_next, config.DIST_MIN_THRES, indices=indices)
            
            # Check Physics Constraints
            deriv = h_next - h + config.TIME_STEP * config.ALPHA_CBF * h
            deriv = deriv * mask * mask_next
            
            # Calculate Error
            error = tf.reduce_sum(tf.nn.relu(-deriv), axis=1)
            
        # Calculate Gradient and Push steering wheel away from danger
        grad = tape.gradient(error, a_res)
        a_res = a_res - config.REFINE_LEARNING_RATE * grad

    # --- Phase 3: The Final Grade ---
    a_opt = a + a_res
    
    dsdt = core.dynamics_tf(s, a_opt)
    s_next = s + dsdt * config.TIME_STEP
    x_next = tf.expand_dims(s_next, 1) - tf.expand_dims(s_next, 0)
    h_next, mask_next, _ = cbf_network(x_next, config.DIST_MIN_THRES, indices=indices)

    loss_dang, loss_safe, acc_dang, acc_safe = core.loss_barrier_tf(
        h_next, s_next, config.DIST_MIN_THRES, config.TIME_TO_COLLISION, config.TOP_K, indices=indices, eps=[0, 0])

    loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv = core.loss_derivatives_tf(
        cbf_network, s_next, a_opt, h_next, config.DIST_MIN_THRES, config.TIME_TO_COLLISION, config.ALPHA_CBF, config.TOP_K, config.TIME_STEP, indices=indices)

    loss_action = core.loss_action_tf(s, g, a)

    loss_list = [loss_dang, loss_safe, loss_dang_deriv, loss_safe_deriv, loss_action]
    acc_list = [acc_dang, acc_safe, acc_dang_deriv, acc_safe_deriv]

    return s, g, a_opt, loss_list, acc_list

def print_accuracy(accuracy_lists):
    acc = np.array(accuracy_lists)
    acc_list = []
    for i in range(acc.shape[1]):
        acc_i = acc[:, i]
        acc_list.append(np.mean(acc_i[acc_i > 0]))
    print('Accuracy: {}'.format(acc_list))


def render_init():
    fig = plt.figure(figsize=(9, 4))
    return fig

import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import core_upgraded_tf as core
import config

def main_tf():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # 1. Initialize and Load Models
    cbf_network = core.NetworkCBFTF(config.TOP_K, config.OBS_RADIUS)
    action_network = core.NetworkActionTF(config.TOP_K, config.OBS_RADIUS)
    
    # TF2 Checkpoint Loading (We create dummy optimizers just to match the saved checkpoint structure)
    optimizer_h = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    optimizer_a = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    checkpoint = tf.train.Checkpoint(optimizer_h=optimizer_h, optimizer_a=optimizer_a, 
                                     cbf_network=cbf_network, action_network=action_network)
    
    if args.model_path:
        # expect_partial() suppresses warnings if we don't use the optimizers during eval
        checkpoint.restore(args.model_path).expect_partial()

    state_gain = tf.constant(np.eye(2, 4) + np.eye(2, 4, k=2) * np.sqrt(3), dtype=tf.float32)

    safety_ratios_epoch, safety_ratios_epoch_lqr = [], []
    dist_errors, init_dist_errors, accuracy_lists = [], []
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
        s_ori, g_ori = core.generate_data_tf(args.num_agents, config.DIST_MIN_THRES * 1.5)
        
        s, g = tf.identity(s_ori), tf.identity(g_ori)
        init_dist_errors.append(tf.reduce_mean(tf.norm(s[:, :2] - g, axis=1)).numpy())
        
        s_np_ours, s_np_lqr = [], []
        safety_ours, safety_lqr = [], []

        # --- THE AI RUN ---
        for i in range(config.INNER_LOOPS):
            # Run test-time optimization
            _, _, a_opt, loss_list, acc_list = evaluate_step_tf(cbf_network, action_network, s, g)
            
            # Physics Step (No .detach() needed in TF2, Eager Execution handles this naturally)
            s = s + tf.concat([s[:, 2:], a_opt], axis=1) * config.TIME_STEP
            
            # Save state to CPU memory for Matplotlib
            s_np = s.numpy()
            g_np = g.numpy()
            s_np_ours.append(s_np)
            
            # Calculate safety using pure NumPy to save GPU cycles
            safety_mask = core.ttc_dangerous_mask_np(s_np, config.DIST_MIN_CHECK, config.TIME_TO_COLLISION_CHECK)
            safety_ratio_array = 1 - np.mean(safety_mask, axis=1)
            safety_ours.append(safety_ratio_array)
            safety_info.append((safety_ratio_array == 1).astype(np.float32).reshape((1, -1)))
            
            safety_ratio_val = np.mean(safety_ratio_array == 1)
            safety_ratios_epoch.append(safety_ratio_val)
            accuracy_lists.append([acc.numpy() for acc in acc_list])

            # End-of-episode logic & LQR Hack
            dist_to_goal = np.linalg.norm(s_np[:, :2] - g_np, axis=1)
            if args.vis:
                if np.amax(dist_to_goal) < config.DIST_MIN_CHECK / 3:
                    time.sleep(1)
                    break
                if np.mean(dist_to_goal) < config.DIST_MIN_CHECK / 2:
                    # Switch to LQR to avoid target jitter
                    s_ref = tf.concat([s[:, :2] - g, s[:, 2:]], axis=1)
                    a_lqr = -tf.matmul(s_ref, tf.transpose(state_gain))
                    s = s + tf.concat([s[:, 2:], a_lqr], axis=1) * config.TIME_STEP
            else:
                if np.mean(dist_to_goal) < config.DIST_MIN_CHECK:
                    break

        dist_errors.append(np.mean(np.linalg.norm(s.numpy()[:, :2] - g.numpy(), axis=1)))
        safety_reward.append(np.mean(np.sum(np.concatenate(safety_info, axis=0) - 1, axis=0)))
        dist_reward.append(np.mean((np.linalg.norm(s.numpy()[:, :2] - g.numpy(), axis=1) < 0.2).astype(np.float32) * 10))

        # --- THE LQR BASELINE RUN ---
        s_lqr, g_lqr = tf.identity(s_ori), tf.identity(g_ori)
        for i in range(config.INNER_LOOPS):
            s_ref_lqr = tf.concat([s_lqr[:, :2] - g_lqr, s_lqr[:, 2:]], axis=1)
            a_lqr = -tf.matmul(s_ref_lqr, tf.transpose(state_gain))
            s_lqr = s_lqr + tf.concat([s_lqr[:, 2:], a_lqr], axis=1) * config.TIME_STEP
            
            s_np_baseline = s_lqr.numpy()
            g_np_baseline = g_lqr.numpy()
            s_np_lqr.append(s_np_baseline)
            
            safety_mask_lqr = core.ttc_dangerous_mask_np(s_np_baseline, config.DIST_MIN_CHECK, config.TIME_TO_COLLISION_CHECK)
            safety_ratio_array_lqr = 1 - np.mean(safety_mask_lqr, axis=1)
            safety_lqr.append(safety_ratio_array_lqr)
            safety_info_baseline.append((safety_ratio_array_lqr == 1).astype(np.float32).reshape((1, -1)))
            
            safety_ratios_epoch_lqr.append(np.mean(safety_ratio_array_lqr == 1))
            
            if np.mean(np.linalg.norm(s_np_baseline[:, :2] - g_np_baseline, axis=1)) < config.DIST_MIN_CHECK / 3:
                break

        safety_reward_baseline.append(np.mean(np.sum(np.concatenate(safety_info_baseline, axis=0) - 1, axis=0)))
        dist_reward_baseline.append(np.mean((np.linalg.norm(s_lqr.numpy()[:, :2] - g_lqr.numpy(), axis=1) < 0.2).astype(np.float32) * 10))

        # --- VISUALIZATION RENDERING ---
        if args.vis:
            s_ori_np = s_ori.numpy()
            g_np = g.numpy()
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
    main_tf()