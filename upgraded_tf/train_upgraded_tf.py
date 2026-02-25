import sys
import os
import argparse
import math
import core_upgraded_tf as core
import numpy as np
import tensorflow as tf
import config

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
            acc_list.append(-1.0) # Keeps the flag if the event never happened
    return acc_list

def main_tf():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Initialize Object-Oriented Networks
    cbf_network = core.NetworkCBFTF(config.TOP_K, config.OBS_RADIUS)
    action_network = core.NetworkActionTF(config.TOP_K, config.OBS_RADIUS)
    
    # Initialize Optimizers
    optimizer_h = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    optimizer_a = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)

    # Checkpoint Manager (TF2 style loading/saving)
    checkpoint = tf.train.Checkpoint(optimizer_h=optimizer_h, optimizer_a=optimizer_a, 
                                     cbf_network=cbf_network, action_network=action_network)
    if args.model_path:
        checkpoint.restore(args.model_path)

    state_gain = tf.constant(np.eye(2, 4) + np.eye(2, 4, k=2) * np.sqrt(3), dtype=tf.float32)

    # 2. THE TRAINING LOOP
    for istep in range(config.TRAIN_STEPS):
        s, g = core.generate_data_tf(args.num_agents, config.DIST_MIN_THRES)
        s_lqr, g_lqr = tf.identity(s), tf.identity(g)
        
        # We must push data through the networks once before making our gradient buckets!
        if istep == 0:
            core.compute_loss_tf(cbf_network, action_network, s, g, config)

        # Initialize empty buckets for gradient accumulation
        accumulated_grads_h = [tf.zeros_like(var) for var in cbf_network.trainable_variables]
        accumulated_grads_a = [tf.zeros_like(var) for var in action_network.trainable_variables]
        
        loss_lists, acc_lists = [], []

        # --- THE AI INNER LOOP ---
        for i in range(config.INNER_LOOPS):
            with tf.GradientTape() as tape_h, tf.GradientTape() as tape_a:
                # 1. Forward Pass & Loss Calculation
                a, loss_list, total_loss, acc_list = core.compute_loss_tf(cbf_network, action_network, s, g, config)
                scaled_loss = total_loss / float(config.INNER_LOOPS)

            # 2. Calculate and Accumulate Gradients
            grads_h = tape_h.gradient(scaled_loss, cbf_network.trainable_variables)
            grads_a = tape_a.gradient(scaled_loss, action_network.trainable_variables)
            
            accumulated_grads_h = [acc + grad for acc, grad in zip(accumulated_grads_h, grads_h)]
            accumulated_grads_a = [acc + grad for acc, grad in zip(accumulated_grads_a, grads_a)]

            # 3. Exploration Noise
            if tf.random.uniform([]) < config.ADD_NOISE_PROB:
                a = a + tf.random.normal(tf.shape(a)) * config.NOISE_SCALE
                
            # 4. Physics Engine
            s = s + tf.concat([s[:, 2:], a], axis=1) * config.TIME_STEP
            
            loss_lists.append([l_val.numpy() for l_val in loss_list])
            acc_lists.append([acc_val.numpy() for acc_val in acc_list])
            
            if tf.reduce_mean(tf.norm(s[:, :2] - g, axis=1)) < config.DIST_MIN_CHECK:
                break

        # --- THE DUMMY BASELINE (LQR) ---
        for i in range(config.INNER_LOOPS):
            s_ref_lqr = tf.concat([s_lqr[:, :2] - g_lqr, s_lqr[:, 2:]], axis=1)
            a_lqr = -tf.matmul(s_ref_lqr, tf.transpose(state_gain))
            s_lqr = s_lqr + tf.concat([s_lqr[:, 2:], a_lqr], axis=1) * config.TIME_STEP
            
            if tf.reduce_mean(tf.norm(s_lqr[:, :2] - g_lqr, axis=1)) < config.DIST_MIN_CHECK:
                break

        # --- THE PING-PONG OPTIMIZATION STEP ---
        if (istep // 10) % 2 == 0:
            optimizer_h.apply_gradients(zip(accumulated_grads_h, cbf_network.trainable_variables))
        else:
            optimizer_a.apply_gradients(zip(accumulated_grads_a, action_network.trainable_variables))

        # --- LOGGING & SAVING ---
        if istep % config.DISPLAY_STEPS == 0:
            print(f'Step: {istep}, Loss: {np.mean(loss_lists, axis=0)}, Accuracy: {count_accuracy(acc_lists)}')

        if istep % config.SAVE_STEPS == 0 or istep + 1 == config.TRAIN_STEPS:
            os.makedirs('models', exist_ok=True)
            checkpoint.save(file_prefix=f'models/model_iter_{istep}')

if __name__ == '__main__':
    main_tf()