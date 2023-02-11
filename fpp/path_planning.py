import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from itertools import accumulate
from bisect import bisect_left

def build_optimal_control(B, T, N, lam, verbose=False, tol=1e-5, **kwargs):

    # Variables.
    D = len(lam)
    var = [cp.Variable((N + 1 - j, B.d)) for j in range(D + 1)]

    # Parameters.
    par = {}
    par['start'] = cp.Parameter(B.d)
    par['goal'] = cp.Parameter(B.d)
    par['l'] = cp.Parameter((N + 1, B.d))
    par['u'] = cp.Parameter((N + 1, B.d))

    # Cost function.
    cost = 0
    for j in range(D):
        pj = var[j + 1].flatten()
        cost += lam[j] * cp.sum_squares(pj) / (N - j + 1)

    # Boundary conditions.
    h = T / N
    constr = {}
    constr['start'] = var[0][0] == par['start']
    constr['goal'] = var[0][-1] == par['goal']
    constr['l'] = par['l'] <= var[0]
    constr['u'] = var[0] <= par['u']
    for j in range(D):
        constr[f'der_{j}'] = var[j][1:] == var[j][:-1] + h * var[j + 1]

    # Optimal control problem.
    prob = cp.Problem(cp.Minimize(cost), constr.values())

    # Solve dummy instance.
    for p in par.values():
        p.value = np.zeros(p.shape)
    prob.solve(**kwargs)

    def planner(start, goal, safety_seq):

        # Set initial and final conditions.
        par['start'].value[:] = start
        par['goal'].value[:] = goal

        # Set initial boxes.
        par['l'].value[:] = [B.boxes[k].l for k in safety_seq]
        par['u'].value[:] = [B.boxes[k].u for k in safety_seq]

        # Solve sequence of optimal control problems.
        return optimal_control(B, prob, var, par, constr, safety_seq, verbose, tol, **kwargs)

    return planner

def optimal_control(B, prob, var, par, constr, safety_seq, verbose, tol, **kwargs):

    # Initialize iterations.
    solver_time = 0
    N = var[0].shape[0] - 1

    while True:

        # Solve optimal control.
        prob.solve(**kwargs)
        p = var[0].value
        solver_time += prob.solver_stats.solve_time
        if verbose:
            print(f'Cost: {prob.value}')
            print(f'Solver time: {prob.solver_stats.solve_time}')

        # Find where Lagrange multipliers are nonzero.
        mul_l = constr['l'].dual_value
        mul_u = constr['u'].dual_value
        # Line below uses mul_l >= 0 and mul_u >= 0.
        mul_norm = np.linalg.norm(mul_l + mul_u, axis=1)

        # Improve path.
        improvement = False
        for i in np.where(mul_norm > tol)[0]:

            # Stabbing problem.
            k_prev, k0, k_next = safety_seq[i - 1:i + 2]
            box0 = B.boxes[k0]
            subset = B.inters[k_prev] & B.inters[k_next]
            subset = list(subset.union({k_prev, k_next}))
            other_boxes = list(B.contain(p[i], tol, subset))
            if len(other_boxes) > 0:

                # Shape variations when moving to another box.
                dl = np.vstack([B.boxes[k].l - box0.l for k in other_boxes])
                du = np.vstack([box0.u - B.boxes[k].u for k in other_boxes])

                # Pick the box with the lowest cost drop.
                cost_drops = dl.dot(mul_l[i]) + du.dot(mul_u[i])
                kk = cost_drops.argmin()
                cost_drop = cost_drops[kk]

                # If cost drop is negative update box.
                if cost_drop < - tol:
                    k = other_boxes[kk]
                    safety_seq[i] = k
                    boxk = B.boxes[k]
                    par['l'].value[i] = boxk.l
                    par['u'].value[i] = boxk.u
                    improvement = True

        if not improvement:
            break

    return prob.value, p, safety_seq, solver_time

def initial_safety_sequence(G, N, start, goal, box_seq):

    # Reconstruct corners from shortest path problem.
    corners = np.zeros((len(box_seq) + 1, G.B.d))
    corners[0] = start
    corners[-1] = goal
    path = [G.node(k, l) for k, l in zip(box_seq[:-1], box_seq[1:])]
    corners[1:-1] = [G.nodes[v]['point'] for v in path]

    # Sample box sequence.
    distances = np.linalg.norm(corners[1:] - corners[:-1], axis=1)
    partial_lengths = list(accumulate(distances))
    samples = np.linspace(0, partial_lengths[-1], N + 1)
    safety_sequence = [box_seq[bisect_left(partial_lengths, t)] for t in samples]

    return safety_sequence
