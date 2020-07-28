# -*- coding: utf-8 -*-
"""
:mod:`orion.core.transfer` --
Learning search spaces for Bayesian optimization:
Another view of hyperparameter transfer learning
======================================================================

.. module:: transfer
   :platform: Unix
   :synopsis: Executes transfer optimization of learning search space.

"""
import logging

import numpy as np


log = logging.getLogger(__name__)
# It should apply experiment and trials objects (in its same root path?)
# to represent current task (experiment) and historical tasks (trials)
# and update the space domain in the experiment object
# The data structure is aligned with the object of experiments and trials
# curr_task = {"space":
#                   [{"lr": [lower, upper]},
#                    ...,
#                    {"wd": [lower, upper]}]}

# hist_task = [{"params":
#               [{"name": "lr", "type": real, "value": 0.05},
#                ...,
#                {"name": "wd", "type": real, "value": 0.0001}]},
#              ...,
#              {"params":
#               [{"name": "lr", "type": real, "value": 0.02},
#                ...,
#                {"name": "wd", "type": real, "value": 0.0003}]}]

# constraints: 1) hyper-params are the same in curr and hist;
#              2) all types are real


def rep_task(hist_tasks, curr_task):
    """Represent search space as matrix"""
    hpp_idx_map = {}
    # idx_hpp_map = {}
    curr_bound = np.zeros((2, len(curr_task["space"])))  # 0: lower, 1: upper
    hist_opt = np.zeros((len(hist_tasks), len(curr_task["space"])))

    for i in range(len(curr_task["space"])):
        value_item = next(iter(curr_task["space"][i].values()))
        key_item = next(iter(curr_task["space"][i].keys()))
        hpp_idx_map[key_item] = i
        # idx_hpp_map[i] = key_item
        curr_bound[0, i] = value_item[0]
        curr_bound[1, i] = value_item[1]

    for i in range(len(hist_tasks)):
        for j in range(len(hist_tasks[i]["params"])):
            hist_opt[i, hpp_idx_map[hist_tasks[i]["params"][j]["name"]]] = \
                hist_tasks[i]["params"][j]["value"]

    return hist_opt, curr_bound


def shrink_bbox_space(hist_tasks, curr_task):
    """Shrink search space with bounding box representation according to the historical knowledge"""
    hist_bound = np.zeros((2, len(curr_task["space"])))
    adj_bound = np.zeros((2, len(curr_task["space"])))

    hist_opt, curr_bound = rep_task(hist_tasks, curr_task)

    hist_bound[0, :] = np.min(hist_opt, axis=0)
    hist_bound[1, :] = np.max(hist_opt, axis=0)

    adj_bound[0, :] = np.max(np.stack((hist_bound[0], curr_bound[0])), axis=0)
    adj_bound[1, :] = np.min(np.stack((hist_bound[1], curr_bound[1])), axis=0)

    if np.any(adj_bound[0] > adj_bound[1]):
        return  # if no intersection, reture curr setting space
    for i in range(len(curr_task["space"])):
        key_item = next(iter(curr_task["space"][i].keys()))
        curr_task["space"][i][key_item][0] = adj_bound[0, i]
        curr_task["space"][i][key_item][1] = adj_bound[1, i]
    return


def shrink_ellips_space(hist_tasks, curr_task):
    """Shrink search space with ellipsoidal representation according to the historical knowledge"""
    hyper_param_dim = len(curr_task["space"])
    hist_opt, curr_bound = rep_task(hist_tasks, curr_task)
    matrix_a = np.zeros((hyper_param_dim, hyper_param_dim))
    vector_b = np.zeros((1, hyper_param_dim))

    # need to implement interior-points algorithm to solve the convex problem by CVXPY in future#

    return matrix_a, vector_b


def rej_sampling_ellips_space(matrix_a, vector_b, curr_task):
    """Reject sample from ellipsoidal search space example"""
    hyper_param_dim = vector_b.shape[0]
    _, curr_bound = rep_task([], curr_task)

    is_feasible = False
    while not is_feasible:
        mean = np.zeros(hyper_param_dim)
        cov = np.eye(hyper_param_dim)
        z = np.random.multivariate_normal(mean, cov)
        r = np.random.random()
        scale = np.power(r, 1.0 / hyper_param_dim) / np.linalg.norm(z)
        t = scale * z
        x = np.mat(np.linalg.inv(matrix_a)) * np.matrix.transpose(np.mat(t - vector_b))
        x = np.array(x).ravel()
        diff = curr_bound - np.tile(x, (2, 1))
        if np.all(diff[0, :] <= 0) and np.all(diff[1, :] >= 0):  # x in curr search space
            is_feasible = True

    # x is in the same order as curr_task representation
    return x


if __name__ == "__main__":
    # curr_task = {"space":
    #                   [{"lr": [0.001, 0.04]},
    #                    {"wd": [0.00001, 0.00005]}]}

    curr_task = {"space":
                 [{"lr": [0.001, 0.04]},
                  {"wd": [0.00001, 0.01]}]}

    hist_task = [{"params":
                  [{"name": "lr", "type": "real", "value": 0.05},
                   {"name": "wd", "type": "real", "value": 0.0001}]},
                 {"params":
                  [{"name": "lr", "type": "real", "value": 0.02},
                   {"name": "wd", "type": "real", "value": 0.0003}]},
                 {"params":
                  [{"name": "lr", "type": "real", "value": 0.007},
                   {"name": "wd", "type": "real", "value": 0.0002}]}]

    shrink_bbox_space(hist_task, curr_task)
    print("update curr task:", curr_task)
