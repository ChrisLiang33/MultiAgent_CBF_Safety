import torch
import torch.nn as nn
import math
import config
import torch.nn.functional as F
import numpy as np


def generate_data_torch(num_agents, distance_min_threshold, max_retries=1000):
    side_length = math.sqrt(max(1.0, num_agents / 8.0))
    
    # Pre-allocate PyTorch tensors
    states = torch.zeros((num_agents, 2), dtype=torch.float32)
    goals = torch.zeros((num_agents, 2), dtype=torch.float32)

    # 1. Spawn States
    i = 0
    retries = 0
    while i < num_agents:
        candidate = torch.rand(2) * side_length  # torch.rand is uniform [0, 1)
        if i > 0:
            # torch.norm handles the distance, torch.min grabs the smallest
            distance_min = torch.min(torch.norm(states[:i] - candidate, dim=1))
            if distance_min <= distance_min_threshold:
                retries += 1
                if retries > max_retries:
                    raise RuntimeError("Map too crowded for agents!")
                continue
                
        states[i] = candidate
        i += 1
        retries = 0

    # 2. Spawn Goals
    i = 0
    retries = 0
    while i < num_agents:
        candidate = (torch.rand(2) - 0.5) + states[i]
        if i > 0:
            distance_min = torch.min(torch.norm(goals[:i] - candidate, dim=1))
            if distance_min <= distance_min_threshold:
                retries += 1
                if retries > max_retries:
                    raise RuntimeError("Goals are clustering too tightly!")
                continue
                
        goals[i] = candidate
        i += 1
        retries = 0

    # 3. Add zero velocities using torch.cat
    states = torch.cat([states, torch.zeros((num_agents, 2))], dim=1)
    
    return states, goals

def remove_distant_agents(x, k, indices=None):
    n = x.size(0)

    # 1. The Empty Room Check
    if n <= k:
        return x, False

    # Helper: Create a row map for PyTorch Advanced Indexing -> [[0], [1], [2], ...]
    row_idx = torch.arange(n).unsqueeze(1)

    # 2. The VIP List Shortcut
    if indices is not None:
        # PyTorch instantly maps the rows to the neighbor indices
        x_filtered = x[row_idx, indices] 
        return x_filtered, indices

    # 3. Calculate Distances
    sq_dist = torch.sum(torch.square(x[:, :, :2]), dim=2)
    d_norm = torch.sqrt(sq_dist + 1e-6)

    # 4. Find Nearest Neighbors
    _, indices = torch.topk(d_norm, k=k, dim=1, largest=False)

    # 5. Extract the Data
    x_filtered = x[row_idx, indices]
    
    return x_filtered, indices

def dynamics_torch(s, a):
    """
    s (N, 4): the current state [x, y, vx, vy]
    a (N, 2): the acceleration by each agent [ax, ay]
    returns:
        dsdt (N, 4): the time derivative of s [vx, vy, ax, ay]
    """
    # Grab the velocities (columns 2 and 3) and glue them to the accelerations
    dsdt = torch.cat([s[:, 2:], a], dim=1)
    return dsdt

class NetworkCBF(nn.Module):
    def __init__(self, top_k=config.TOP_K, obs_radius=config.OBSERVATION_RADIUS, in_features=6):
        super(NetworkCBF, self).__init__()
        self.top_k = top_k
        self.obs_radius = obs_radius
        
        # We define the "Brain" once here. 
        # Notice we FIXED THE BUG: No ReLU on the final output layer!
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Linear output so it can return negative danger scores
        )

    def forward(self, x, r, indices=None):
        N = x.size(0)
        
        # 1. Calculate distances (with safe epsilon)
        d_norm = torch.sqrt(torch.sum(torch.square(x[:, :, :2]), dim=2) + 1e-4)
        
        # 2. Augment the profile (Add ID tag and danger margin)
        # torch.eye(N).unsqueeze(2) creates the (N, N, 1) identity matrix
        eye_tag = torch.eye(N, device=x.device, dtype=x.dtype).unsqueeze(2)
        margin = (d_norm - r).unsqueeze(2)
        x = torch.cat([x, eye_tag, margin], dim=2)
        
        # 3. The Bouncer (Assuming you imported our upgraded PyTorch function)
        x, indices = remove_distant_agents(x, k=self.top_k, indices=indices)
        
        # 4. The Blindfold (Sensor Mask)
        distance = torch.sqrt(torch.sum(torch.square(x[:, :, :2]), dim=2, keepdim=True) + 1e-4)
        mask = (distance <= self.obs_radius).float()
        
        # 5. Pass through the Neural Network
        x = self.net(x)
        
        # 6. Apply the mask
        x = x * mask
        
        return x, mask, indices
    
class NetworkAction(nn.Module):
    def __init__(self, top_k=config.TOP_K, obs_radius=config.OBSERVATION_RADIUS, in_features=5):
        super(NetworkAction, self).__init__()
        self.top_k = top_k
        self.obs_radius = obs_radius
        
        # 1. The Crowd Summarizer (PointNet local feature extractor)
        self.crowd_net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # 2. The Gain Calculator (Takes 128 crowd features + 4 state features)
        self.gain_net = nn.Sequential(
            nn.Linear(128 + 4, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4) # Outputs the 4 raw variables
        )

    def forward(self, s, g, indices=None):
        N = s.size(0)
        
        # --- Phase 1: The 360-Degree Radar ---
        x = s.unsqueeze(1) - s.unsqueeze(0)
        eye_tag = torch.eye(N, device=s.device, dtype=x.dtype).unsqueeze(2)
        x = torch.cat([x, eye_tag], dim=2)
        
        # Filter to Top-K neighbors (using our upgraded function from earlier)
        x, indices = remove_distant_agents(x, k=self.top_k, indices=indices)
        
        # Calculate distances for the blindfold mask
        distance = torch.norm(x[:, :, :2], dim=2, keepdim=True)
        mask = (distance < self.obs_radius).float()
        
        # --- Phase 2: The Crowd Summarizer ---
        x = self.crowd_net(x)
        x = x * mask
        x, _ = torch.max(x, dim=1) # PointNet Max Pooling!
        
        # --- Phase 3: The Big Picture ---
        # Concat crowd summary, distance to goal, and current velocity
        state_goal_diff = s[:, :2] - g
        velocity = s[:, 2:]
        x = torch.cat([x, state_goal_diff, velocity], dim=1)
        
        # --- Phase 4: The Gain Brain ---
        x = self.gain_net(x)
        x = 2.0 * torch.sigmoid(x) + 0.2
        
        # Split into the 4 PD gains (shape: [N, 1] each)
        k1, k2, k3, k4 = torch.split(x, 1, dim=1)
        
        # --- Phase 5: The PD Controller (Algebraic Upgrade!) ---
        # Instead of weird matrix math, we just write the physics equation natively:
        dx, dy = state_goal_diff[:, 0:1], state_goal_diff[:, 1:2]
        vx, vy = velocity[:, 0:1], velocity[:, 1:2]
        
        a_x = -k1 * dx - k2 * vx
        a_y = -k3 * dy - k4 * vy
        
        a = torch.cat([a_x, a_y], dim=1)
        return a

def time_to_collision_dangerous_mask_torch(s, r, ttc, top_k, indices=None):
    N = s.size(0)
    
    # --- Phase 1: Setting up the Radar ---
    s_diff = s.unsqueeze(1) - s.unsqueeze(0)
    eye_tag = torch.eye(N, device=s.device, dtype=s.dtype).unsqueeze(2)
    s_diff = torch.cat([s_diff, eye_tag], dim=2)
    
    # Filter to Top-K (assuming the PyTorch version of remove_distant_agents)
    s_diff, _ = remove_distant_agents(s_diff, k=top_k, indices=indices)
    
    # Split the 5 features into individual tensors (size 1 each)
    x, y, vx, vy, eye = torch.split(s_diff, 1, dim=2)
    
    # The Self-Collision Hack
    x = x + eye
    y = y + eye

    # --- Phase 2: The Physics Translation ---
    alpha = vx**2 + vy**2
    beta = 2 * (x*vx + y*vy)
    gamma = x**2 + y**2 - r**2

    # --- Phase 3 & 4: The Logic Flags ---
    # Are we already inside each other?
    dist_dangerous = (gamma < 0)

    # Are we on a crash course? (Discriminant > 0, moving towards each other)
    has_two_positive_roots = ((beta**2 - 4*alpha*gamma) > 0) & (gamma > 0) & (beta < 0)

    # Will it happen before the TTC clock runs out?
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

def loss_barrier_torch(h, s, r, ttc, top_k, indices=None, eps=[5e-2, 1e-3]):
    # Flatten the safety scores
    h_flat = h.view(-1)
    
    # 1. Get the physical truth and flatten it
    dang_mask = time_to_collision_dangerous_mask_torch(s, r, ttc, top_k, indices=indices)
    dang_mask_flat = dang_mask.view(-1)
    safe_mask_flat = ~dang_mask_flat  # ~ is the PyTorch NOT operator
    
    # 2. The Sorting Hat (Boolean Masking)
    dang_h = h_flat[dang_mask_flat]
    safe_h = h_flat[safe_mask_flat]
    
    num_dang = float(dang_h.size(0))
    num_safe = float(safe_h.size(0))
    
    # 3. Hinge Loss (Using ReLU as a shortcut for max(..., 0))
    loss_dang = torch.sum(F.relu(dang_h + eps[0])) / (1e-5 + num_dang)
    loss_safe = torch.sum(F.relu(-safe_h + eps[1])) / (1e-5 + num_safe)
    
    # 4. The Report Card & Empty Room Edge Case
    # Look how clean standard Python 'if' statements are!
    if num_dang > 0:
        accuracy_dang = torch.sum((dang_h <= 0).float()) / num_dang
    else:
        accuracy_dang = torch.tensor(-1.0, device=h.device)
        
    if num_safe > 0:
        accuracy_safe = torch.sum((safe_h > 0).float()) / num_safe
    else:
        accuracy_safe = torch.tensor(-1.0, device=h.device)
        
    return loss_dang, loss_safe, accuracy_dang, accuracy_safe

def loss_action_torch(s, g, a):
    # 1. Calculate the relative state
    dx_dy = s[:, :2] - g
    vx_vy = s[:, 2:]
    
    # 2. The Expert Equation (Algebraic Shortcut!)
    # We completely removed the matrix multiplication. 
    # This computes exactly: a_ref = -1.0 * Position - sqrt(3) * Velocity
    action_ref = -dx_dy - math.sqrt(3.0) * vx_vy
    
    # 3. Calculate the Energy Magnitudes (Squared)
    action_ref_norm = torch.sum(torch.square(action_ref), dim=1)
    action_net_norm = torch.sum(torch.square(a), dim=1)
    
    # 4. Calculate the Mean Absolute Error of the magnitudes
    norm_diff = torch.abs(action_net_norm - action_ref_norm)
    loss = torch.mean(norm_diff)
    
    return loss

def loss_derivatives_torch(cbf_network, s, a, h, r, ttc, alpha, top_k, time_step, indices=None, eps=[1e-3, 0]):
    # --- Phase 1: The Time Machine ---
    dsdt = dynamics_torch(s, a)
    s_next = s + dsdt * time_step
    
    # Calculate the future relative state
    x_next = s_next.unsqueeze(1) - s_next.unsqueeze(0)
    
    # Ask the Safety Network to grade the future
    h_next, _, _ = cbf_network(x_next, r, indices=indices)

    # --- Phase 2: The Golden Rule (CBF Constraint) ---
    deriv = h_next - h + time_step * alpha * h
    deriv_flat = deriv.view(-1)

    # --- Phase 3: The Sorting Hat ---
    dang_mask = time_to_collision_dangerous_mask_torch(s, r, ttc, top_k, indices=indices)
    dang_mask_flat = dang_mask.view(-1)
    safe_mask_flat = ~dang_mask_flat

    dang_deriv = deriv_flat[dang_mask_flat]
    safe_deriv = deriv_flat[safe_mask_flat]

    num_dang = float(dang_deriv.size(0))
    num_safe = float(safe_deriv.size(0))

    # --- Phase 4: The Brake Pedal Penalty (Hinge Loss via ReLU) ---
    # We negate the derivative so that positive (safe) derivatives become negative and get zeroed out by ReLU
    loss_dang_deriv = torch.sum(F.relu(-dang_deriv + eps[0])) / (1e-5 + num_dang)
    loss_safe_deriv = torch.sum(F.relu(-safe_deriv + eps[1])) / (1e-5 + num_safe)

    # --- Phase 5: The Report Card & Empty Room Edge Case ---
    if num_dang > 0:
        acc_dang_deriv = torch.sum((dang_deriv >= 0).float()) / num_dang
    else:
        acc_dang_deriv = torch.tensor(-1.0, device=s.device)

    if num_safe > 0:
        acc_safe_deriv = torch.sum((safe_deriv >= 0).float()) / num_safe
    else:
        acc_safe_deriv = torch.tensor(-1.0, device=s.device)

    return loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv

def compute_loss_torch(cbf_network, action_network, s, g, config):
    x = s.unsqueeze(1) - s.unsqueeze(0)
    
    h, mask, indices = cbf_network(x, config.DIST_MIN_THRES, indices=None)
    a = action_network(s, g, indices=indices)

    loss_dang, loss_safe, acc_dang, acc_safe = loss_barrier_torch(
        h, s, config.DIST_MIN_THRES, config.TIME_TO_COLLISION, config.TOP_K, indices=indices)

    loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv = loss_derivatives_torch(
        cbf_network, s, a, h, config.DIST_MIN_THRES, config.TIME_TO_COLLISION, config.ALPHA_CBF, config.TOP_K, config.TIME_STEP, indices=indices)

    loss_action = loss_action_torch(s, g, a)

    loss_list = [2 * loss_dang, loss_safe, 2 * loss_dang_deriv, loss_safe_deriv, 0.01 * loss_action]
    acc_list = [acc_dang, acc_safe, acc_dang_deriv, acc_safe_deriv]

    # Note: Weight decay is handled natively by the Adam optimizer in train.py!
    total_loss = 10.0 * sum(loss_list)

    return a, loss_list, total_loss, acc_list