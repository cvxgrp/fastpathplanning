import numpy as np
import cvxpy as cp
from time import time
from itertools import accumulate
from bisect import bisect
from scipy.special import binom
from fastpathplanning.bezier import BezierCurve, CompositeBezierCurve, l2_matrix

class Log:

    def __init__(self):
        print(self.write('Iter.') + \
              self.write('Cost') + \
              self.write('% Decr.') + \
              self.write('Kappa') + \
              self.write('Accept'))
        self.size = 60
        print('-' * self.size)

    def write(self, s, size=10):
        s = str(s)
        s0 = list('|' + ' ' * size + '|')
        s0[2:2 + len(s)] = s
        return ''.join(s0)

    def update(self, i, cost, decr, kappa, accept):
        if not np.isnan(decr):
            decr =  - round(decr * 100)
        print(self.write(i) + \
              self.write('{:.2e}'.format(cost)) + \
              self.write(f'{decr}') + \
              self.write('{:.1e}'.format(kappa)) + \
              self.write(accept))

    def terminate(self, cost, perc_cost, cost_tol):
        print('-' * self.size)
        if perc_cost < cost_tol:
            print('Expected cost decrease is {:.1e}'.format(perc_cost))
            print('...smaller than tolerance {:.1e}'.format(cost_tol))
        print('Final cost is {:.3e}'.format(cost))

def seq_conv_prog(L, U, trav_times, alpha, initial, final, verbose=False):

    # Check inputs.
    assert L.shape == U.shape
    assert max(initial) < max(alpha)
    assert max(final) < max(alpha)

    # Algorithm parameters.
    kappa = 1
    omega = 3
    cost_tol = 1e-2
    
    # Initialize optimization problems.
    T = sum(trav_times)
    projection_step, cvxpy_time_proj = projection_problem(L, U, alpha, initial, final)
    tangent_step, cvxpy_time_tan = tangent_problem(L, U, alpha, initial, final, T)
    cvxpy_time = cvxpy_time_proj + cvxpy_time_tan

    # Solve initial Bezier problem.
    best_trav_times = trav_times
    best_path, best_cost, best_points, cvxpy_time_proj = projection_step(best_trav_times)
    cvxpy_time += cvxpy_time_proj
    n_iters = 1
    if verbose:
        log = Log()
        log.update(n_iters, best_cost, np.nan, np.nan, True)

    # Iterate retiming and Bezier.
    convergence = False
    perc_cost = np.inf
    while not convergence:

        # Tangent step.
        new_trav_times, tangent_cost, kappa_max, cvxpy_time_tan = tangent_step(kappa, best_trav_times, best_points)
        perc_cost = (best_cost - tangent_cost) / best_cost
        if perc_cost < cost_tol:
            convergence = True

        # Projection step.
        new_path, new_cost, new_points, cvxpy_time_proj = projection_step(new_trav_times)
        cvxpy_time += cvxpy_time_tan + cvxpy_time_proj
        n_iters += 1
        
        # If tangent step improved the trajectory.
        decr = new_cost - best_cost
        accept = decr < 0
        if verbose:
            log.update(n_iters, new_cost, decr / best_cost, kappa, accept)
        if accept:
            best_trav_times = new_trav_times
            best_path = new_path
            best_cost = new_cost
            best_points = new_points
        kappa = kappa_max / omega
 
    if verbose:
        log.terminate(best_cost, perc_cost, cost_tol)

    return best_path, cvxpy_time

def control_points(dimension, D, n_boxes, n_points, type=cp.Variable):

    points = []
    for j in range(n_boxes):
        points_j = []
        for i in range(D + 1):
            points_j.append(type((n_points - i, dimension)))
        points.append(points_j)

    return points

def boundary_conditions(points, initial, final):

    constraints = []
    for i, value in initial.items():
        constraints.append(points[0][i][0] == value)
    for i, value in final.items():
        constraints.append(points[-1][i][-1] == value)

    return constraints

def box_containment(points, L, U):

    constraints = []
    for j, points_j in enumerate(points):
        n_points = points_j[0].shape[0]
        Lj = np.array([L[j]] * n_points)
        Uj = np.array([U[j]] * n_points)
        constraints.append(points_j[0] >= Lj)
        constraints.append(points_j[0] <= Uj)

    return constraints

def differentiability(points):

    constraints = []
    for points_jp1, points_j in zip(points[1:], points[:-1]):
        for points_jp1_i, points_j_i in zip(points_jp1, points_j):
            constraints.append(points_j_i[-1] == points_jp1_i[0])

    return constraints

def get_path(points, trav_times):

    beziers = []
    a = 0
    for j, points_j in enumerate(points):
        b = a + trav_times[j]
        beziers.append(BezierCurve(points_j[0].value, a, b))
        a = b

    return CompositeBezierCurve(beziers)

def get_points(points):

    opt_points = []
    for points_j in points:
        opt_points_j = []
        for points_ji in points_j:
            opt_points_j.append(points_ji.value)
        opt_points.append(opt_points_j)

    return opt_points

def projection_problem(L, U, alpha, initial, final):

    # Start clock.
    tic = time()

    # Problem size.
    n_boxes, d = L.shape
    D = max(alpha)
    n_points = (D + 1) * 2

    # Variables.
    points = control_points(d, D, n_boxes, n_points)

    # Parameters.
    trav_times = cp.Parameter(n_boxes, pos=True)

    # Constraints.
    constraints = boundary_conditions(points, initial, final)
    constraints += box_containment(points, L, U)
    constraints += differentiability(points)

    # Bezier dynamics.
    for j in range(n_boxes):
        for i in range(D):
            ci = trav_times[j] / (n_points - i - 1)
            constraints.append(points[j][i][1:] - points[j][i][:-1] == ci * points[j][i + 1])

    # Cost function.
    cost = 0
    for i, ai in alpha.items():
        if ai > 0:
            A = ai * l2_matrix(n_points - i - 1, d)
            for j in range(n_boxes):
                p = cp.vec(points[j][i], order='C')
                cost += trav_times[j] * cp.quad_form(p, A)

    # Construct problem.
    prob = cp.Problem(cp.Minimize(cost), constraints)
    assert prob.is_dcp(dpp=True)

    def projection_step(trav_times_val):

        # Solve problem.
        tic = time()
        trav_times.value = trav_times_val
        prob.solve(solver='CLARABEL', ignore_dpp=True)
        path = get_path(points, trav_times_val)
        opt_points = get_points(points)
        cvxpy_time = time() - tic - prob.solver_stats.solve_time

        return path, prob.value, opt_points, cvxpy_time

    cvxpy_time = time() - tic

    return projection_step, cvxpy_time

def tangent_problem(L, U, alpha, initial, final, T):

    # Start clock.
    tic = time()

    # Problem size.
    n_boxes, d = L.shape
    D = max(alpha)
    n_points = (D + 1) * 2

    # Variables.
    points = control_points(d, D, n_boxes, n_points)

    # Parameters.
    trust_region = cp.Parameter(pos=True)
    trust_region_inv = cp.Parameter(pos=True)
    nom_trav_times = cp.Parameter(n_boxes, pos=True)
    nom_points = control_points(d, D, n_boxes, n_points, cp.Parameter)
    nom_points_scaled = control_points(d, D, n_boxes, n_points, cp.Parameter)

    # Constraints.
    constraints = boundary_conditions(points, initial, final)
    constraints += box_containment(points, L, U)
    constraints += differentiability(points)

    # Traversal times.
    trav_times = cp.Variable(n_boxes)
    constraints.append(sum(trav_times) == T)
    constraints.append(trav_times * trust_region >= nom_trav_times)
    constraints.append(trav_times * trust_region_inv <= nom_trav_times)

    # Bezier dynamics.
    for j in range(n_boxes):
        for i in range(D):
            ci = 1 / (n_points - i - 1)
            lin = trav_times[j] * nom_points[j][i + 1] \
                  + nom_trav_times[j] * points[j][i + 1] \
                  - nom_points_scaled[j][i + 1]
            point_diff = points[j][i][1:] - points[j][i][:-1]
            constraints.append(point_diff == ci * lin)

    # Cost function.
    cost = 0
    for i, ai in alpha.items():
        if ai > 0:
            A = ai * l2_matrix(n_points - i - 1, d)
            A_chol = np.linalg.cholesky(A).T
            for j in range(n_boxes):
                point_diff = points[j][i - 1][1:] - points[j][i - 1][:-1]
                p = cp.vec(point_diff, order='C') * (n_points - i)
                cost += cp.quad_over_lin(A_chol @ p, trav_times[j])

    # Construct problem.
    prob = cp.Problem(cp.Minimize(cost), constraints)
    assert prob.is_dcp(dpp=True)

    def tangent_step(kappa, trav_times_val, points_val):

        # Start clock.
        tic = time()

        # Solve problem.
        trust_region.value = 1 + kappa
        trust_region_inv.value = 1 / trust_region.value
        nom_trav_times.value = trav_times_val
        for j in range(n_boxes):
            for i in range(D + 1):
                nom_points[j][i].value = points_val[j][i]
                nom_points_scaled[j][i].value = points_val[j][i] * trav_times_val[j]
        prob.solve(solver='CLARABEL', ignore_dpp=True)

        # Trust region update.
        opt_trav_times = trav_times.value
        ratios = np.divide(opt_trav_times, trav_times_val)
        kappa_max = max(max(ratios), 1 / min(ratios)) - 1
        cvxpy_time = time() - tic - prob.solver_stats.solve_time

        return opt_trav_times, prob.value, kappa_max, cvxpy_time

    cvxpy_time = time() - tic

    return tangent_step, cvxpy_time
