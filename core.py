import numpy as np
import tensorflow as tf
import config  

# generates circle 
def generate_obstacle_circle(center, radius, num=12):
    theta = np.linspace(0, np.pi * 2, num=num, endpoint=False).reshape(-1,1)
    unit_circle = np.concatenate([np.cos(theta), np.sin(theta)], axis=1)
    circle = np.array(center) + unit_circle * radius
    return circle

# generate squares 
def generate_obstacle_ractangle(center, sides,num=12):
    a, b = sides
    n_side_1 = int(num // 2 * a /(a+b))
    n_side_2 = num // 2 - n_side_1
    n_side_3 = n_side_1
    n_side_4 = num - n_side_1 - n_side_2 - n_side_3

    side_1 = np.concatenate([
        np.linspace(-a/2, a/2, n_side_1, endpoint=False).reshape(-1, 1), 
        b/2 * np.ones(n_side_1).reshape(-1, 1)], axis=1)
    side_2 = np.concatenate([
        a/2 * np.ones(n_side_2).reshape(-1, 1),
        np.linspace(b/2, -b/2, n_side_2, endpoint=False).reshape(-1, 1)], axis=1)
    side_3 = np.concatenate([
        np.linspace(a/2, -a/2, n_side_3, endpoint=False).reshape(-1, 1), 
        -b/2 * np.ones(n_side_3).reshape(-1, 1)], axis=1)
    side_4 = np.concatenate([
        -a/2 * np.ones(n_side_4).reshape(-1, 1),
        np.linspace(-b/2, b/2, n_side_4, endpoint=False).reshape(-1, 1)], axis=1)

    rectangle = np.concatenate([side_1, side_2, side_3, side_4], axis=0)
    rectangle = rectangle + np.array(center)
    return rectangle

##change made with the states[:i] and goals[:i] and if i > 0
def generate_data(num_agents, distance_min_threshold):
    """
    this is a map generator
    where it spwans starting position (start_x, start_y, velocity_x, velocity_y) and goals (goal_x, goal_y) for every agent

    Args:
        num_agents (int): number of agents we are generating
        distance_min_threshold (_type_): minimun distance in between each agents

    Returns:
        _type_: states goal matrix
    """
    side_length = np.sqrt(max(1.0, num_agents/ 8.0))
    states = np.zeros(shape=(num_agents, 2), dtype=np.float32)
    goals = np.zeros(shape=(num_agents, 2), dtype=np.float32)

    #spawning starting positions, reject sample if sample is too close to an exsiting agent comepared to the threshold
    i = 0
    while i < num_agents:
        candidate = np.random.uniform(size=(2,)) * side_length
        if i > 0:

            #calculate the distance between the dart and every agent on the board, pick the min one to see if it violets the condition
            distance_min = np.linalg.norm(states[:i] - candidate, axis=1).min()
            if distance_min <= distance_min_threshold:
                continue
        states[i] = candidate
        i += 1
    
    #same thing, sample a point that is reletive close -0.5 and 0.5, relatively close to where started and reject sampleing
    i = 0
    while i < num_agents:
        candidate = np.random.uniform(-0.5, 0.5, size=(2,)) + states[i]
        if i > 0:
            distance_min = np.linalg.norm(goals[:i] - candidate, axis=1).min()
            if distance_min <= distance_min_threshold:
                continue
        goals[i] = candidate
        i += 1

    states = np.concatenate([states, np.zeros(shape=(num_agents, 2), dtype=np.float32)], axis=1)
    return states, goals

def remove_distant_agents(x, k, indices=None):
    """filter to the nearest k agents

    Args:
        x (_type_): _description_
        k (_type_): config.topk
        indices (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    n, _, c = x.get_shape().as_list()

    # number of agent less than topk
    if n <= k:
        return x, False
    
    # indices is already calculated closes neighbors, so no need to recaulculate again
    if indices is not None:
        x = tf.reshape(tf.gather_nd(x, indices), [n, k, c])
        return x, indices
    
    #d_norm is distance between you and everyone else, bottomk == negative topk
    d_norm = tf.sqrt(tf.reduce_sum(tf.square(x[:, :, :2]) + 1e-6, axis=2)) #4x4 grid
    _, indices = tf.nn.top_k(-d_norm, k=k)

    #extract data and pull out nearest agents
    row_indices = tf.expand_dims(tf.range(tf.shape(indices)[0]), 1) * tf.ones_like(indices)
    row_indices = tf.reshape(row_indices, [-1, 1])
    column_indices = tf.reshape(indices, [-1,1])
    indices = tf.concat([row_indices, column_indices], axis=1)
    x = tf.reshape(tf.gather_nd(x, indices), [n, k, c])
    return x, indices

# the function takes the state of agents, calc relative distances, filters out agents too far away, filters out people outside of our sensors reach
# and pass the remaining neighbors through as NN. the output a scalar value AKA safety score for each neighbors, positive means safe.
def network_cbf(x, r, indices=None):
    # distance between you and everyone else, + id tag and danger margin
    d_norm = tf.sqrt(tf.reduce_sum(tf.square(x[:,:,:2]) + 1e-4, axis=2))
    x = tf.concat([x, tf.expand_dims(tf.eye(tf.shape(x)[0]), 2), tf.expand_dims(d_norm - r, 2)], axis = 2)

    # neighbor filtering 
    x, indices = remove_distant_agents(x=x, k=config.TOP_K, indices=indices)

    #observation masking, even though i have narrowed down to top-k nearest people towards me,
    #they might still be outside of my sensor range, the mask is a switch, 1(on), 0(off)
    distance = tf.sqrt(tf.reduce_sum(tf.square(x[:,:,:2]) + 1e-4, axis=2, keepdims=True))
    mask = tf.cast(tf.less_equal(distance, config.OBSERVATION_RADIUS), tf.float32)

    # take the profiles of the closes people and pass through a NN, where you process each person speed, distance, angel
    #the network spits out a single number for each person AKA safety score
    x = tf.contrib.layers.conv1d(input=x,num_outputs=64,kernel_size=1,reuse=tf.AUTO_REUSE,scope='obs/conv_1',activation_fn=tf.nn.relu)
    x = tf.contrib.layers.conv1d(input=x,num_outputs=128,kernel_size=1,reuse=tf.AUTO_REUSE,scope='obs/conv_2',activation_fn=tf.nn.relu)
    x = tf.contrib.layers.conv1d(input=x,num_outputs=64,kernel_size=1,reuse=tf.AUTO_REUSE,scope='obs/conv_3',activation_fn=tf.nn.relu)
    x = tf.contrib.layers.conv1d(input=x,num_outputs=1,kernel_size=1,reuse=tf.AUTO_REUSE,scope='obs/conv_4',activation_fn=None)

    #apply the score if mask is 1 else 0
    x = x * mask
    return x, mask, indices

#look at where the agent is, where the goal is, who is around it, then calculate acceleration forces (A_x, A_y)
def network_action(s, g, obervsation_radius=1.0, indices=None):
    #s is list of all agents states [x, y, V_x, V_y]
    #calculate relative distances, add id tag then kick everyone out of the way
    x = tf.expand_dims(s, 1) - tf.expand_dims(s, 0)
    x = tf.concat([x, tf.expand_dims(tf.eye(tf.shape(x)[0]), 2)], axis=2)
    x, _ = remove_distant_agents(x=x, k=config.TOP_K, indices=indices)

    #create sensor blind where agent cant react to thing cant see
    distance = tf.norm(x[:, :, :2], axis=2, keepdims=True)
    mask = tf.cast(tf.less(distance, obervsation_radius), tf.float32)

    #nn needs a fixed input size, and that can be different when everyone have different numbers of neighbours
    #this NN takes any number of neighbors down to a 1 size of 128 that represents the crowd density
    x = tf.contrib.layers.conv1d(input=x, num_outputs=64,kernel_size=1, reuse=tf.AUTO_REUSE,scope='action/conv1',activation_fn=tf.nn.relu)
    x = tf.contrib.layers.conv1d(input=x, num_outputs=128,kernel_size=1, reuse=tf.AUTO_REUSE,scope='action/conv2',activation_fn=tf.nn.relu)
    x = tf.reduce_max(x * mask, axis=1)

    # 128dim vector + 
    # s[:, :2] - g, the exact distance to the goal(x-G_x, y-G_y)
    #s[:, 2:] agents current velocity
    x = tf.concat([x, s[:, :2] - g, s[:, 2:]], axis=1)

    #outputs 4 variables
    x = tf.contrib.layer.fully_connected(inputs=x,num_outputs=64,resue=tf.AUTO_REUSE,scope='action/fc_1',activation_fn=tf.nn.relu)
    x = tf.contrib.layer.fully_connected(inputs=x,num_outputs=128,resue=tf.AUTO_REUSE,scope='action/fc_2',activation_fn=tf.nn.relu)
    x = tf.contrib.layer.fully_connected(inputs=x,num_outputs=64,resue=tf.AUTO_REUSE,scope='action/fc_3',activation_fn=tf.nn.relu)
    x = tf.contrib.layer.fully_connected(inputs=x,num_outputs=4,resue=tf.AUTO_REUSE,scope='action/fc_4',activation_fn=None)
    x = 2.0 * tf.nn.sigmoid(x) + 0.2
    k_1, k_2, k_3, k_4 = tf.split(x, 4, axis=1)

    #it outputed spring tightness(k1,k3), braje friction(k2, k4)
    zeros = tf.zeros_like(k_1)
    gain_x = -tf.concat([k_1, zeros, k_2, zeros], axis=1)
    gain_y = -tf.concat([zeros, k_3, zeros, k_4], axis=1)
    state = tf.concat([s[:, :2] - g, s[:, 2:]], axis=1)
    a_x = tf.reduce_sum(state * gain_x, axis=1, keepdims=True)
    a_y = tf.reduce_sum(state * gain_y, axis=1, keepdims=True)
    a = tf.concat([a_x, a_y], axis=1)
    #it calculates the X acceleration, calculates the Y acceleration, glue them together into [ax ay]
    return a

def dynamics(s, a) :
    """
    s(N,4): the current state
    a(N,2): the acceleration by each agent
    returns:
    dsdt (N,4): the time derivative of s
    """
    dsdt = tf.concat([s[:, 2:], a], axis=1)
    return dsdt

#looks into future to see if agent will collaps if they maintain their current speed and direction
def time_to_collision_dangerous_mask(s, r, ttc, indices=None):
    #predictive radar, it asks, if everyone moves at the speed they are now, will i crash into someone at X seconds
    #using quadratic formula

    #setting up radar, it calculates the relative distance and relative velocity between agents and keep nearest topk neighbors
    s_diff = tf.expand_dims(s, 1) - tf.expand_dims(s, 0)
    s_diff = tf.concat([s_diff, tf.expand_dims(tf.eye(tf.shape(s)[0]), 2)], axis=2)
    s_diff, _ = remove_distant_agents(s_diff, config.TOP_K, indices)

    x, y, vx, vy, eye = tf.split(s_diff, 5, axis=2)
    x = x + eye
    y = y + eye

    # quadratic formulas
    alpha = vx ** 2 + vy ** 2
    beta = 2 * (x * vx + y * vy)
    gamma = x ** 2 + y ** 2 - r ** 2

    #if true then already danger
    distance_dangerous = tf.less(gamma, 0)

    #are we crashing
    has_two_positive_roots = tf.logical_and(tf.greater(beta ** 2 - 4 * alpha * gamma, 0), tf.logical_and(tf.greater(gamma, 0), tf.less(beta, 0)))

    #will it be soon?
    root_less_than_ttc = tf.logical_or(tf.less(-beta - 2 * alpha * ttc, 0), tf.less((beta + 2 * alpha * ttc) ** 2, beta ** 2 - 4 * alpha * gamma))

    has_root_less_than_ttc = tf.logical_and(has_two_positive_roots, root_less_than_ttc)
    ttc_dangerouse = tf.logical_or(distance_dangerous, has_root_less_than_ttc)

    #Combines the flags. The mask returns True if you are currently inside an agent (distance_dangerous) OR if you will hit them before the clock runs out (has_root_less_than_ttc).
    return ttc_dangerouse

def ttc_dangerous_mask_np(s, r, ttc):
    s_diff = np.expand_dims(s, 1) - np.expand_dims(s, 0)
    x, y, vx, vy = np.split(s_diff, 4, axis=2)
    x = x + np.expand_dims(np.eye(np.shape(s)[0]), 2)
    y = y + np.expand_dims(np.eye(np.shape(s)[0]), 2)
    alpha = vx ** 2 + vy ** 2
    beta = 2 * (x * vx + y * vy)
    gamma = x ** 2 + y ** 2 - r ** 2
    dist_dangerous = np.less(gamma, 0)

    has_two_positive_roots = np.logical_and(np.greater(beta ** 2 - 4 * alpha * gamma, 0), np.logical_and(np.greater(gamma, 0), np.less(beta, 0)))
    root_less_than_ttc = np.logical_or( np.less(-beta - 2 * alpha * ttc, 0), np.less((beta + 2 * alpha * ttc) ** 2, beta ** 2 - 4 * alpha * gamma))
    has_root_less_than_ttc = np.logical_and(has_two_positive_roots, root_less_than_ttc)
    ttc_dangerous = np.logical_or(dist_dangerous, has_root_less_than_ttc)

    return ttc_dangerous

#how to see danger
def loss_barrier(h, s, r, ttc, indices=None, eps=[5e-2, 1e-3]):
    """
    args:
        h(N,N,1): the control barrier function
        s(N,4): the current state of N agents
        r(float): the radius of the safe regions
        ttc(float): the threshold of time to collision
    """
    #sort prediction into two buckets. things are dangerous and safe buckets
    h_reshape = tf.reshape(h, [-1])
    dang_mask = time_to_collision_dangerous_mask(s, r=r, ttc=ttc, indices=indices)
    dang_mask_reshape = tf.reshape(dang_mask, [-1])
    safe_mask_reshape = tf.logical_not(dang_mask_reshape)
    dang_h = tf.boolean_mask(h_reshape, dang_mask_reshape)
    safe_h = tf.boolean_mask(h_reshape, safe_mask_reshape)
    
    #The Dangerous Bucket Penalty: If a state is physically dangerous, the AI should have output a negative number.
    #The Safe Bucket Penalty: It does the exact opposite here. If a state is safe, h should be positive.
    num_dang = tf.cast(tf.shape(dang_h)[0], tf.float32)
    num_safe = tf.cast(tf.shape(safe_h)[0], tf.float32)
    loss_dang = tf.reduce_sum(tf.math.maximum(dang_h + eps[0], 0)) / (1e-5 + num_dang)
    loss_safe = tf.reduce_sum(tf.math.maximum(-safe_h + eps[1], 0)) / (1e-5 + num_safe)

    #This is just calculating the Accuracy Percentage (0.0 to 1.0) so you can print it to your console/TensorBoard during training
    accuracy_dang = tf.reduce_sum(tf.cast(tf.less_equal(dang_h, 0), tf.float32)) / (1e-5 + num_dang)
    accuracy_safe = tf.reduce_sum(tf.cast(tf.greater(safe_h, 0), tf.float32)) / (1e-5 + num_safe)
    accuracy_dang = tf.cond(tf.greater(num_dang, 0), lambda: accuracy_dang, lambda: -tf.constant(1.0))
    accuracy_safe = tf.cond(tf.greater(num_safe, 0), lambda: accuracy_safe, lambda: -tf.constant(1.0))

    return loss_dang, loss_safe, accuracy_dang, accuracy_safe

#teach it how to brake and steer
def loss_derivatives(s, a, h, x, r, ttc, alpha, indices=None, eps=[1e-3,0]):

    #it takes the agents current state (s) and action(a), use dynamics to figure out where the agent will be a second from now(s_next)
    #"If I hit the gas pedal like this, where will I be in a fraction of a second, and will I be safe?"
    dsdt = dynamics(s, a)
    s_next = s + dsdt * config.TIME_STEP
    x_next = tf.expand_dims(s_next, 1) - tf.expand_dims(s_next, 0)
    h_next, mask_next, _ = network_cbf(x=x_next, r=config.DISTANCE_MIN_THRESHOLD, indices=indices)

    #h_next - h in how much safety score changed
    # if you are far away from danger h will be a large positive number, then you are allowed to drive towards and obsticle,
    #if h is almost 0 and you are at the edge then h_next - h will be positive, forcing a action to increase the score
    #Applying the strict mathematical constraint that guarantees collision avoidance.
    deriv= h_next - h + config.TIME_STEP * alpha * h
    deriv_reshape = tf.reshape(deriv, [-1])

    #splits agents into 2 groups, one on the crash course and safe group
    # It consults the Predictive Radar to separate the deriv scores into two buckets: agents on a crash course (dang_deriv), and agents who are currently fine (safe_deriv).
    dang_mask = time_to_collision_dangerous_mask(s=s, r=r, ttc=ttc, indices=indices)
    dang_mask_reshape = tf.reshape(dang_mask, [-1])
    safe_mask_reshape = tf.logical_not(dang_mask_reshape)
    dang_deriv = tf.boolean_mask(deriv_reshape, dang_mask_reshape)
    safe_deriv = tf.boolean_mask(deriv_reshape, safe_mask_reshape)
    num_dang = tf.cast(tf.shape(dang_deriv)[0], tf.float32)
    num_safe = tf.cast(tf.shape(safe_deriv)[0], tf.float32)

    # If deriv is positive (the action successfully steered away from danger), -dang_deriv becomes negative, the maximum against 0 returns 0, and the network gets no penalty.
    # If deriv is negative (the action made the crash worse), -dang_deriv becomes positive, generating a high loss score. The network updates its weights to say, "Don't ever do that again."
    loss_dang_deriv = tf.reduce_sum(tf.math.maximum(-dang_deriv + eps[0], 0)) / (1e-5 + num_dang)
    loss_safe_deriv = tf.reduce_sum(tf.math.maximum(-safe_deriv + eps[1], 0)) / (1e-5 + num_safe)

    # this calculates the human-readable Accuracy Percentage (0.0 to 1.0).
    acc_dang_deriv = tf.reduce_sum(tf.cast(
        tf.greater_equal(dang_deriv, 0), tf.float32)) / (1e-5 + num_dang)
    acc_safe_deriv = tf.reduce_sum(tf.cast(
        tf.greater_equal(safe_deriv, 0), tf.float32)) / (1e-5 + num_safe)
    acc_dang_deriv = tf.cond(
        tf.greater(num_dang, 0), lambda: acc_dang_deriv, lambda: -tf.constant(1.0))
    acc_safe_deriv = tf.cond(
        tf.greater(num_safe, 0), lambda: acc_safe_deriv, lambda: -tf.constant(1.0))

    return loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv

def loss_action(s, g, a, r, ttc):
    #math matrix that represents the perect PD controller
    state_gain = -tf.constant(np.eye(2,4) + np.eye(2, 4, k=2) * np.sqrt(3), dtype=tf.float32)

    #ideal acceleration to the goal
    s_ref = tf.concat([s[:, :2] - g, s[:, 2:]], axis=1)
    action_ref = tf.linalg.matmul(s_ref, state_gain, False, True)

    # it compares the Energy Magnitude of the actions, completely ignoring the Direction. "I don't care what direction you steer. If there is an obstacle, you can turn 90 degrees left to avoid it. BUT, you are only allowed to press the gas pedal exactly as hard as the Expert Driver would."
    action_ref_norm = tf.reduce_sum(tf.square(action_ref), axis=1)
    action_net_norm = tf.reduce_sum(tf.square(a), axis=1)
    norm_diff = tf.abs(action_net_norm - action_ref_norm)
    loss = tf.reduce_mean(norm_diff)
    return loss

# def statics(s, a, h, alpha, indices=None):
#     dsdt = dynamics(s, a)
#     s_next = s + dsdt * config.TIME_STEP

#     x_next = tf.expand_dims(s_next, 1) - tf.expand_dims(s_next, 0)
#     h_next, mask_next, _ = network_cbf(x=x_next, r=config.DIST_MIN_THRES, indices=indices)

#     deriv = h_next - h + config.TIME_STEP * alpha * h

#     mean_deriv = tf.reduce_mean(deriv)
#     std_deriv = tf.sqrt(tf.reduce_mean(tf.square(deriv - mean_deriv)))
#     prob_neg = tf.reduce_mean(tf.cast(tf.less(deriv, 0), tf.float32))

#     return mean_deriv, std_deriv, prob_neg










