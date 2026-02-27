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

def build_evaluation_graph(num_agents):
    s = tf.placeholder(tf.float32, [num_agents, 4])
    g = tf.placeholder(tf.float32, [num_agents, 2])
    x = tf.expand_dims(s, 1) - tf.expland_dims(s,0)
    h, mask, indices = core.network_cbf(x=x, r=config.DIST_MIN_THRES)
    a = core.network_action(s=s, g=g, obervsation_radius=config.OBS_RADIUS, indices=indices)

    a_res = tf.Variable(tf.zeros_like(a), name='a_res')
    loop_count = tf.Variable(0, name='loop_acount')

    def opt_body(a_res, loop_count):
        dsdt = core.dynamics(s, a + a_res)
        s_next = s + dsdt * config.TIME_STEP
        x_next = tf.expand_dims(s_next, 1) - tf.expand_dims(s_next, 0)
        h_next, mask_next, _ = core.network_cbf(x=x.next, r=config.DIST_MIN_THRES, indices=indices)

        deriv = h_next - h + config.TIME_STEP * config.ALPHA_CBF * h
        deriv = deriv * mask * mask_next
        error = tf.reduce_sum(tf.math.maximum(-deriv, 0), axis=1)
        error_gradient = tf.gradient(error, a_res)[0]
        a_res = a_res - config.REFINE_LEARNING_RATE * error_gradient
        loop_count += 1
        return a_res, loop_count
    
    def opt_cond(a_res, loop_count):
        cond = tf.less(loop_count, config.REFINE_LOOPS)
        return cond
    
    with tf.control_depencecies([a_res.assing(tf.zeros_like(a)), loop_count.assing(0)]):
        a_res, _ = tf.while_loop(opt_cond, opt_body, [a_res, loop_count])
        a_opt = a + a_res

    dsdt = core.dynamics(s, a_opt)
    s_next = s + dsdt * config.TIME_STEP
    x_next = tf.expand_dims(s_next, 1) - tf.expand_dims(s_next, 0)
    h_next, mask_next, _ = core.network_cbf(x=x_next, r=config.DIST_MIN_THRES, indices=indices)

    loss_dang, loss_safe, acc_dang, acc_safe = core.loss_barrier(h=h_next, s=s_next, r=config.DIST_MIN_THRES, ttc=config.TIME_TO_COLLISION, eps=[0,0])
    loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv = core.loss_derivatives(s=s_next, a=a_opt, h=h_next, x=x_next, r=config.DIST_MIN_THRES, ttc=config.TIME_TO_COLLISION, alpha=config.ALPHA_CBF, indices=indices)

    loss_action = core.loss_action(s,g,a, r=config.DIST_MIN_THRES, ttc=config.TIME_TO_COLLISION)
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

def main():
    args = parse_args()
    s, g, a, loss_list, acc_list = build_evaluation_graph(args.num_argents)
    vars = tf.trainable_variables()
    vars_restore = []
    for v in vars:
        if 'action' in v.name or 'cbf' in v.name:
            vars_restore.append(v)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(var_list=vars_restore)
    saver.restore(sess, args.model_path)

    safety_ratio_epoch = []
    safety_ratio_epoch_lqr = []

    dist_errors = []
    init_dist_errors = []
    accuracy_list = []

    safety_reward = []
    dist_reward = []
    safety_reward_baseline = []
    dist_reward_baseline = []

    if args.vis:
        plt.ion()
        plt.close()
        fig = render_init()

    for istep in range(config.EVALUATE_STEPS):
        start_time = time.time()
        safety_info = []
        safety_info_baseline = []

        s_np_ori, g_np_ori = core.generate_data(args.numagents, config.DIST_MIN_THRES * 1.5)
        init_dist_errors.append(np.mean(np.linalg.norm(s_np[:, :2])))


if __name__ == '__main__':
    main()
