import numpy as np
import tensorflow as tf
import config
import math

def generate_data_tf(num_agents, distance_min_threshold, max_retries=1000):
    side_length = math.sqrt(max(1.0, num_agents / 8.0))
    
    # Pre-allocate using tf.Variable so we can modify rows in a loop
    states = tf.Variable(tf.zeros([num_agents, 2], dtype=tf.float32))
    goals = tf.Variable(tf.zeros([num_agents, 2], dtype=tf.float32))

    # 1. Spawn States
    i = 0
    retries = 0
    while i < num_agents:
        candidate = tf.random.uniform([2]) * side_length
        if i > 0:
            # tf.norm calculates distance, tf.reduce_min grabs the smallest
            distance_min = tf.reduce_min(tf.norm(states[:i] - candidate, axis=1))
            if distance_min <= distance_min_threshold:
                retries += 1
                if retries > max_retries:
                    raise RuntimeError("Map too crowded for agents!")
                continue
                
        # Must use .assign() to mutate a tf.Variable slice
        states[i].assign(candidate)
        i += 1
        retries = 0

    # 2. Spawn Goals
    i = 0
    retries = 0
    while i < num_agents:
        candidate = tf.random.uniform([2], minval=-0.5, maxval=0.5) + states[i]
        if i > 0:
            distance_min = tf.reduce_min(tf.norm(goals[:i] - candidate, axis=1))
            if distance_min <= distance_min_threshold:
                retries += 1
                if retries > max_retries:
                    raise RuntimeError("Goals are clustering too tightly!")
                continue
                
        goals[i].assign(candidate)
        i += 1
        retries = 0

    # 3. Add zero velocities using tf.concat
    # We convert states back to a normal tensor here during concatenation
    states_tensor = tf.concat([states, tf.zeros([num_agents, 2], dtype=tf.float32)], axis=1)
    goals_tensor = tf.convert_to_tensor(goals)
    
    return states_tensor, goals_tensor

def remove_distant_agents_v2(x, k, indices=None):
    # Dynamically get the batch size
    n = tf.shape(x)[0]

    # 1. The Empty Room Check
    if n <= k:
        return x, False
    
    # 2. The Memory Shortcut (Now a 1-liner!)
    if indices is not None:
        x = tf.gather(x, indices, batch_dims=1)
        return x, indices

    # 3. Calculate Distances (Epsilon moved to a safer spot)
    # Adding 1e-6 AFTER the sum but INSIDE the sqrt is mathematically safer
    sq_dist = tf.reduce_sum(tf.square(x[:, :, :2]), axis=2)
    d_norm = tf.sqrt(sq_dist + 1e-6)

    # 4. Find Nearest Neighbors (No more negative hack!)
    # Modern TF has argsort, so we can just sort by smallest distance natively
    indices = tf.argsort(d_norm, axis=1, direction='ASCENDING')[:, :k]

    # 5. Extract Data (No more coordinate maps!)
    x = tf.gather(x, indices, batch_dims=1)
    
    return x, indices

@tf.function
def dynamics_tf(s, a):
    """
    s (N, 4): the current state [x, y, vx, vy]
    a (N, 2): the acceleration by each agent [ax, ay]
    returns:
        dsdt (N, 4): the time derivative of s [vx, vy, ax, ay]
    """
    dsdt = tf.concat([s[:, 2:], a], axis=1)
    return dsdt

class NetworkCBFTF(tf.keras.Model):
    def __init__(self, top_k, obs_radius):
        super(NetworkCBFTF, self).__init__()
        self.top_k = top_k
        self.obs_radius = obs_radius
        
        # Define the layers once
        self.conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu')
        self.conv2 = tf.keras.layers.Conv1D(filters=128, kernel_size=1, activation='relu')
        self.conv3 = tf.keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu')
        # BUG FIXED: activation=None is the default, but we explicitly state it for clarity
        self.conv4 = tf.keras.layers.Conv1D(filters=1, kernel_size=1, activation=None)

    def call(self, x, r, indices=None):
        # 1. Calculate distances
        d_norm = tf.sqrt(tf.reduce_sum(tf.square(x[:, :, :2]), axis=2) + 1e-4)
        
        # 2. Augment the profile
        eye_tag = tf.expand_dims(tf.eye(tf.shape(x)[0]), 2)
        margin = tf.expand_dims(d_norm - r, 2)
        x = tf.concat([x, eye_tag, margin], axis=2)
        
        # 3. The Bouncer (Using our upgraded TF2 function)
        x, indices = remove_distant_agents_v2(x, k=self.top_k, indices=indices)
        
        # 4. The Blindfold
        distance = tf.sqrt(tf.reduce_sum(tf.square(x[:, :, :2]), axis=2, keepdims=True) + 1e-4)
        mask = tf.cast(tf.less_equal(distance, self.obs_radius), tf.float32)
        
        # 5. Pass through the Neural Network
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # 6. Apply the mask
        x = x * mask
        
        return x, mask, indices

class NetworkActionTF(tf.keras.Model):
    def __init__(self, top_k, obs_radius):
        super(NetworkActionTF, self).__init__()
        self.top_k = top_k
        self.obs_radius = obs_radius
        
        # 1. The Crowd Summarizer
        self.crowd_dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.crowd_dense2 = tf.keras.layers.Dense(128, activation='relu')
        
        # 2. The Gain Calculator
        self.gain_fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.gain_fc2 = tf.keras.layers.Dense(128, activation='relu')
        self.gain_fc3 = tf.keras.layers.Dense(64, activation='relu')
        self.gain_fc4 = tf.keras.layers.Dense(4, activation=None) # Output raw numbers

    def call(self, s, g, indices=None):
        # --- Phase 1: The 360-Degree Radar ---
        x = tf.expand_dims(s, 1) - tf.expand_dims(s, 0)
        eye_tag = tf.expand_dims(tf.eye(tf.shape(x)[0]), 2)
        x = tf.concat([x, eye_tag], axis=2)
        
        x, indices = remove_distant_agents_v2(x, k=self.top_k, indices=indices)
        
        distance = tf.norm(x[:, :, :2], axis=2, keepdims=True)
        mask = tf.cast(tf.less(distance, self.obs_radius), tf.float32)
        
        # --- Phase 2: The Crowd Summarizer ---
        x = self.crowd_dense1(x)
        x = self.crowd_dense2(x)
        x = x * mask
        x = tf.reduce_max(x, axis=1) # PointNet Max Pooling
        
        # --- Phase 3: The Big Picture ---
        state_goal_diff = s[:, :2] - g
        velocity = s[:, 2:]
        x = tf.concat([x, state_goal_diff, velocity], axis=1)
        
        # --- Phase 4: The Gain Brain ---
        x = self.gain_fc1(x)
        x = self.gain_fc2(x)
        x = self.gain_fc3(x)
        x = self.gain_fc4(x)
        x = 2.0 * tf.nn.sigmoid(x) + 0.2
        
        k1, k2, k3, k4 = tf.split(x, 4, axis=1)
        
        # --- Phase 5: The PD Controller (Algebraic Upgrade) ---
        dx, dy = tf.split(state_goal_diff, 2, axis=1)
        vx, vy = tf.split(velocity, 2, axis=1)
        
        a_x = -k1 * dx - k2 * vx
        a_y = -k3 * dy - k4 * vy
        
        a = tf.concat([a_x, a_y], axis=1)
        return a
    
@tf.function
def time_to_collision_dangerous_mask_tf(s, r, ttc, top_k, indices=None):
    N = tf.shape(s)[0]
    
    # --- Phase 1: Setting up the Radar ---
    s_diff = tf.expand_dims(s, 1) - tf.expand_dims(s, 0)
    eye_tag = tf.expand_dims(tf.eye(N), 2)
    s_diff = tf.concat([s_diff, eye_tag], axis=2)
    
    # Filter to Top-K 
    s_diff, _ = remove_distant_agents_v2(s_diff, k=top_k, indices=indices)
    
    # Split into 5 separate tensors
    x, y, vx, vy, eye = tf.split(s_diff, 5, axis=2)
    
    # The Self-Collision Hack
    x = x + eye
    y = y + eye

    # --- Phase 2: The Physics Translation ---
    alpha = vx**2 + vy**2
    beta = 2 * (x*vx + y*vy)
    gamma = x**2 + y**2 - r**2

    # --- Phase 3 & 4: The Logic Flags ---
    dist_dangerous = (gamma < 0)

    has_two_positive_roots = ((beta**2 - 4*alpha*gamma) > 0) & (gamma > 0) & (beta < 0)

    root_less_than_ttc = (-beta - 2*alpha*ttc < 0) | (((beta + 2*alpha*ttc)**2) < (beta**2 - 4*alpha*gamma))

    # --- Phase 5: Final Assembly ---
    has_root_less_than_ttc = has_two_positive_roots & root_less_than_ttc
    ttc_dangerous = dist_dangerous | has_root_less_than_ttc

    return ttc_dangerous

def ttc_dangerous_mask_np(s, r, ttc):
    N = s.shape[0]
    
    # --- Phase 1: Setting up the Radar ---
    # np.newaxis (or None) is the modern, cleaner shortcut for np.expand_dims
    s_diff = s[:, None, :] - s[None, :, :]
    
    # Split directly into x, y, vx, vy
    x, y, vx, vy = np.split(s_diff, 4, axis=2)
    
    # The Self-Collision Hack
    eye_tag = np.eye(N)[:, :, None]
    x = x + eye_tag
    y = y + eye_tag

    # --- Phase 2: The Physics Translation ---
    alpha = vx**2 + vy**2
    beta = 2 * (x*vx + y*vy)
    gamma = x**2 + y**2 - r**2

    # --- Phase 3 & 4: The Logic Flags (Algebraic Upgrade!) ---
    # We completely drop np.logical_and and np.less for standard math operators
    dist_dangerous = (gamma < 0)

    has_two_positive_roots = ((beta**2 - 4*alpha*gamma) > 0) & (gamma > 0) & (beta < 0)

    root_less_than_ttc = (-beta - 2*alpha*ttc < 0) | (((beta + 2*alpha*ttc)**2) < (beta**2 - 4*alpha*gamma))

    # --- Phase 5: Final Assembly ---
    has_root_less_than_ttc = has_two_positive_roots & root_less_than_ttc
    ttc_dangerous = dist_dangerous | has_root_less_than_ttc

    return ttc_dangerous

@tf.function
def loss_action_tf(s, g, a):
    # 1. Calculate the relative state
    dx_dy = s[:, :2] - g
    vx_vy = s[:, 2:]
    
    # 2. The Expert Equation (Algebraic Shortcut!)
    action_ref = -dx_dy - math.sqrt(3.0) * vx_vy
    
    # 3. Calculate the Energy Magnitudes
    action_ref_norm = tf.reduce_sum(tf.square(action_ref), axis=1)
    action_net_norm = tf.reduce_sum(tf.square(a), axis=1)
    
    # 4. Calculate the Mean Absolute Error
    norm_diff = tf.abs(action_net_norm - action_ref_norm)
    loss = tf.reduce_mean(norm_diff)
    
    return loss

@tf.function
def loss_barrier_tf(h, s, r, ttc, top_k, indices=None, eps=[5e-2, 1e-3]):
    # Flatten the safety scores
    h_flat = tf.reshape(h, [-1])
    
    # 1. Get the physical truth and flatten it
    dang_mask = time_to_collision_dangerous_mask_tf(s, r, ttc, top_k, indices=indices)
    dang_mask_flat = tf.reshape(dang_mask, [-1])
    safe_mask_flat = ~dang_mask_flat  # TF2 supports the ~ operator!
    
    # 2. The Sorting Hat
    dang_h = tf.boolean_mask(h_flat, dang_mask_flat)
    safe_h = tf.boolean_mask(h_flat, safe_mask_flat)
    
    num_dang = tf.cast(tf.shape(dang_h)[0], tf.float32)
    num_safe = tf.cast(tf.shape(safe_h)[0], tf.float32)
    
    # 3. Hinge Loss (Using nn.relu instead of math.maximum)
    loss_dang = tf.reduce_sum(tf.nn.relu(dang_h + eps[0])) / (1e-5 + num_dang)
    loss_safe = tf.reduce_sum(tf.nn.relu(-safe_h + eps[1])) / (1e-5 + num_safe)
    
    # 4. The Report Card & Empty Room Edge Case
    if num_dang > 0:
        accuracy_dang = tf.reduce_sum(tf.cast(dang_h <= 0, tf.float32)) / num_dang
    else:
        accuracy_dang = tf.constant(-1.0)
        
    if num_safe > 0:
        accuracy_safe = tf.reduce_sum(tf.cast(safe_h > 0, tf.float32)) / num_safe
    else:
        accuracy_safe = tf.constant(-1.0)
        
    return loss_dang, loss_safe, accuracy_dang, accuracy_safe

@tf.function
def loss_derivatives_tf(cbf_network, s, a, h, r, ttc, alpha, top_k, time_step, indices=None, eps=[1e-3, 0]):
    # --- Phase 1: The Time Machine ---
    dsdt = dynamics_tf(s, a)
    s_next = s + dsdt * time_step
    
    x_next = tf.expand_dims(s_next, 1) - tf.expand_dims(s_next, 0)
    
    # Ask the Safety Network to grade the future
    h_next, _, _ = cbf_network(x_next, r, indices=indices)

    # --- Phase 2: The Golden Rule (CBF Constraint) ---
    deriv = h_next - h + time_step * alpha * h
    deriv_flat = tf.reshape(deriv, [-1])

    # --- Phase 3: The Sorting Hat ---
    dang_mask = time_to_collision_dangerous_mask_tf(s, r, ttc, top_k, indices=indices)
    dang_mask_flat = tf.reshape(dang_mask, [-1])
    safe_mask_flat = ~dang_mask_flat

    dang_deriv = tf.boolean_mask(deriv_flat, dang_mask_flat)
    safe_deriv = tf.boolean_mask(deriv_flat, safe_mask_flat)

    num_dang = tf.cast(tf.shape(dang_deriv)[0], tf.float32)
    num_safe = tf.cast(tf.shape(safe_deriv)[0], tf.float32)

    # --- Phase 4: The Brake Pedal Penalty (Hinge Loss via ReLU) ---
    loss_dang_deriv = tf.reduce_sum(tf.nn.relu(-dang_deriv + eps[0])) / (1e-5 + num_dang)
    loss_safe_deriv = tf.reduce_sum(tf.nn.relu(-safe_deriv + eps[1])) / (1e-5 + num_safe)

    # --- Phase 5: The Report Card & Empty Room Edge Case ---
    if num_dang > 0:
        acc_dang_deriv = tf.reduce_sum(tf.cast(dang_deriv >= 0, tf.float32)) / num_dang
    else:
        acc_dang_deriv = tf.constant(-1.0)

    if num_safe > 0:
        acc_safe_deriv = tf.reduce_sum(tf.cast(safe_deriv >= 0, tf.float32)) / num_safe
    else:
        acc_safe_deriv = tf.constant(-1.0)

    return loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv

@tf.function
def compute_loss_tf(cbf_network, action_network, s, g, config):
    x = tf.expand_dims(s, 1) - tf.expand_dims(s, 0)
    
    h, mask, indices = cbf_network(x, config.DIST_MIN_THRES, indices=None)
    a = action_network(s, g, indices=indices)

    loss_dang, loss_safe, acc_dang, acc_safe = loss_barrier_tf(
        h, s, config.DIST_MIN_THRES, config.TIME_TO_COLLISION, config.TOP_K, indices=indices)

    loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv = loss_derivatives_tf(
        cbf_network, s, a, h, config.DIST_MIN_THRES, config.TIME_TO_COLLISION, config.ALPHA_CBF, config.TOP_K, config.TIME_STEP, indices=indices)

    loss_action = loss_action_tf(s, g, a)

    loss_list = [2 * loss_dang, loss_safe, 2 * loss_dang_deriv, loss_safe_deriv, 0.01 * loss_action]
    acc_list = [acc_dang, acc_safe, acc_dang_deriv, acc_safe_deriv]

    all_variables = cbf_network.trainable_variables + action_network.trainable_variables
    weight_loss = [config.WEIGHT_DECAY * tf.nn.l2_loss(v) for v in all_variables]
    
    total_loss = 10.0 * tf.math.add_n(loss_list + weight_loss)

    return a, loss_list, total_loss, acc_list