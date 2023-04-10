import numpy as np
import cvxpy as cp

from time import time
from itertools import accumulate
from bisect import bisect

from scipy.special import binom


def build_min_distance(B, max_boxes, improve_box_seq=False, **kwargs):

    # Variables and parameters.
    x = cp.Variable((max_boxes + 1, B.d))
    start = cp.Parameter(B.d)
    goal = cp.Parameter(B.d)
    l = cp.Parameter((max_boxes - 1, B.d))
    u = cp.Parameter((max_boxes - 1, B.d))
    # Lower and upper limits for the box intersections. (This is why these
    # matrices have max_boxes - 1 rows.)

    cost = cp.sum([cp.norm(x[i + 1] - x[i], 2) for i in range(max_boxes)])
    constr = [x[0] == start, x[1:-1] >= l, x[1:-1] <= u, x[-1] == goal]

    prob = cp.Problem(cp.Minimize(cost), constr)
    p = {'start': start, 'goal': goal, 'l': l, 'u': u}
    solve_dummy_instance(prob, p, **kwargs)

    def planner(box_seq, start, goal):

        tic = time()
        p['start'].value = start
        p['goal'].value = goal

        if improve_box_seq:
            sol = iterative_planner(prob, x, p, B, box_seq, **kwargs)
        else:
            sol = solve_min_distance(prob, x, p, B, box_seq, **kwargs)

        runtime = time() - tic
        return *sol, runtime

    return planner


def solve_dummy_instance(prob, p, **kwargs):

    for pi in p.values():
        pi.value = np.zeros(pi.shape)
    prob.solve(**kwargs)


def solve_min_distance(prob, x, p, B, box_seq, **kwargs):

    max_boxes = x.shape[0] - 1
    goal = p['goal'].value
    l, u = intersection_bounds(B, box_seq, max_boxes, goal)
    p['l'].value = l
    p['u'].value = u

    prob.solve(**kwargs)
    length = prob.value
    traj = x.value[:len(box_seq) + 1]
    solver_time = prob.solver_stats.solve_time

    return traj, length, solver_time


def intersection_bounds(B, box_seq, max_boxes, goal):

    boxes = [B.boxes[i] for i in box_seq]

    l0 = np.array([np.maximum(b.l, c.l) for b, c in zip(boxes[:-1], boxes[1:])])
    u0 = np.array([np.minimum(b.u, c.u) for b, c in zip(boxes[:-1], boxes[1:])])

    l1 = np.full((max_boxes - l0.shape[0] - 1, B.d), goal)
    u1 = np.full((max_boxes - u0.shape[0] - 1, B.d), goal)

    l = np.vstack((l0, l1))
    u = np.vstack((u0, u1))

    return l, u

def iterative_planner(prob, x, p, B, box_seq, tol=1e-5, **kwargs):

    box_seq = np.array(box_seq)
    solver_time = 0
    n_iters = 0
    while True:
        n_iters += 1

        box_seq = jump_box_repetitions(box_seq)
        traj, length, solver_time_i = solve_min_distance(prob, x, p, B, box_seq, **kwargs)
        solver_time += solver_time_i
        box_seq, traj = merge_overlaps(box_seq, traj, tol)
        # box_seq, traj = remove_redundant_boxes(B, box_seq, traj, tol)

        kinks = find_kinks(traj, tol)

        insert_k = []
        insert_i = []
        for k in kinks:

            i1 = box_seq[k - 1]
            i2 = box_seq[k]
            B1 = B.boxes[i1]
            B2 = B.boxes[i2]
            cached_finf = 0

            subset = list(B.inters[i1] & B.inters[i2])
            for i in B.contain(traj[k], tol, subset):
                B3 = B.boxes[i]
                B13 = B1.intersect(B3)
                B23 = B2.intersect(B3)
                f = dual_box_insertion(*traj[k-1:k+2], B13, B23, tol)
                f2 = np.linalg.norm(f)
                finf = np.linalg.norm(f, ord=np.inf)

                if f2 > 1 + tol and finf > cached_finf + tol:
                    cached_i = i
                    cached_finf = finf

            if cached_finf > 0:
                insert_k.append(k)
                insert_i.append(cached_i)

        if len(insert_k) > 0:
            box_seq = np.insert(box_seq, insert_k, insert_i)
        else:
            return list(box_seq), traj, length, n_iters, solver_time

def merge_overlaps(box_seq, traj, tol):

    keep = list(np.linalg.norm(traj[:-1] - traj[1:], axis=1) > tol)
    box_seq = box_seq[keep]
    traj = traj[keep + [True]]

    return box_seq, traj


def find_kinks(traj, tol):
    '''
    Detects the indices of the points where the trajectory bends. To do so, it
    uses the triangle inequality (division free):

        |traj[i] - traj[i-1]| + |traj[i+1] - traj[i]| > |traj[i+1] - traj[i-1]|

    implies that traj[i] is a kink.
    '''

    xy = np.linalg.norm(traj[1:-1] - traj[:-2], axis=1)
    yz = np.linalg.norm(traj[2:] - traj[1:-1], axis=1)
    xz = np.linalg.norm(traj[2:] - traj[:-2], axis=1)

    return np.where(xy + yz - xz > tol)[0] + 1

def dual_box_insertion(a, x, b, B13, B23, tol=1e-9):
    '''
    Given a point x that solves

        minimize     |x - a| + |x - b|
        subject to   x in B1 cap B2,

    checks if x1 = x2 = x is also optimal for the following problem:

        minimize     |x1 - a| + |x1 - x2| + |x2 - b|
        subject to   x1 in B1 cap B3
                     x2 in B2 cap B3.

    The arguments B13 and B23 denote B1 cap B3 and B2 cap B3, respectively.
    Assumes that neither a nor b is equal to x.

    Returns the dual variable f associated with the additional constraint
    x1 = x2 needed for the check. The multiplier f gives the rate of change of
    the cost as the points x1 and x2 are moved away from each other, i.e., 
    f = d length / d (x2 - x1). If the two-norm of f is greater than one then
    inserting the box B3 decreases the cost.
    '''

    lam1 = a - x
    lam2 = b - x
    lam1 /= np.linalg.norm(lam1)
    lam2 /= np.linalg.norm(lam2)
    
    L1, U1 = B13.active_set(x, tol)
    L2, U2 = B23.active_set(x, tol)

    fmin1 = - lam1
    fmax1 = - lam1
    fmin2 = lam2.copy()
    fmax2 = lam2.copy()

    fmin1[L1] = - np.inf
    fmax1[U1] = np.inf
    fmin2[U2] = - np.inf
    fmax2[L2] = np.inf

    fmin = np.maximum(fmin1, fmin2)
    fmax = np.minimum(fmax1, fmax2)
    f = np.clip(0, fmin, fmax)

    return f


def jump_box_repetitions(box_seq):

    i = 0
    keep = []
    while True:
        keep.append(i)
        b = box_seq[i]
        for j, c in enumerate(box_seq[i:][::-1]):
            if c == b:
                break
        i = len(box_seq) - j
        if i >= len(box_seq):
            break

    return box_seq[keep]
