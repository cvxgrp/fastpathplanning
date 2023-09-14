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
              self.write('Decr.') + \
              self.write('Kappa') + \
              self.write('Accept'))
        self.size = 60
        print('-' * self.size)

    def write(self, s, size=10):
        s = str(s)
        s0 = list('|' + ' ' * size + '|')
        s0[2:2 + len(s)] = s
        return ''.join(s0)

    def update(self, i, cost, cost_decrease, kappa, accept):
        print(self.write(i) + \
              self.write('{:.2e}'.format(cost)) + \
              self.write('{:.1e}'.format(cost_decrease)) + \
              self.write('{:.1e}'.format(kappa)) + \
              self.write(accept))

    def terminate(self, n_iters, cost, runtime):
        print('-' * self.size)
        print(f'Smooth phase terminated in {n_iters} iterations')
        print('Final cost is {:.3e}'.format(cost))
        print('Solver time was {:.1e}s'.format(runtime))

def optimize_bezier_with_retiming(L, U, best_durations, alpha, initial, final, verbose=False, n_points=None):

    # Algorithm parameters.
    kappa = 1
    omega = 3
    kappa_min = 1e-2
    cost_tol = 1e-2

    # Initialize optimization problem.
    T = sum(best_durations)
    problem = problem_skeleton(L, U, alpha, initial, final, T, n_points)

    # Solve initial Bezier problem.
    path, sol_stats = optimize_shape(problem, best_durations)
    best_cost = sol_stats['cost']
    best_points = sol_stats['points']

    if verbose:
        log = Log()
        log.update(0, best_cost, np.nan, np.inf, True)

    bez_runtime = sol_stats['runtime']
    retiming_runtime = 0

    # Iterate retiming and Bezier.
    n_iters = 0
    while True:
        n_iters += 1
        
        # Retime.
        new_durations, kappa_max, sol_stats = optimize_shape_and_timing(problem,
            best_points, best_durations, kappa)
        retiming_runtime += sol_stats['runtime']
        retiming_cost = sol_stats['cost']

        # Improve Bezier curves.
        new_path, sol_stats = optimize_shape(problem, new_durations)
        new_cost = sol_stats['cost']
        bez_runtime += sol_stats['runtime']

        decr = new_cost - best_cost
        accept = decr < 0
        if verbose:
            log.update(n_iters, new_cost, decr, kappa, accept)

        # If retiming improved the trajectory.
        if accept:
            best_durations = new_durations
            best_path = new_path
            best_cost = new_cost
            best_points = sol_stats['points']
        if kappa < kappa_min:
            break
        perc_cost = np.abs(new_cost - retiming_cost) / new_cost
        if perc_cost < cost_tol:
            break
        kappa = kappa_max / omega
        
    runtime = bez_runtime + retiming_runtime
    if verbose:
        log.terminate(n_iters, best_cost, runtime)

    # Solution statistics.
    sol_stats = {}
    sol_stats['cost'] = best_cost
    sol_stats['n_iters'] = n_iters
    sol_stats['bez_runtime'] = bez_runtime
    sol_stats['retiming_runtime'] = retiming_runtime
    sol_stats['runtime'] = runtime

    return path, sol_stats

def problem_skeleton(L, U, alpha, initial, final, T, n_points=None):

    # Problem size.
    assert L.shape == U.shape
    n_boxes, d = L.shape
    D = max(alpha)
    assert max(initial) <= D
    assert max(final) <= D
    if n_points is None:
        n_points = (D + 1) * 2

    # Durations.
    durations = cp.Variable(n_boxes)
    duration_constraints = [sum(durations) == T]

    # Control points of the curves and their derivatives.
    points = {}
    for j in range(n_boxes):
        points[j] = {}
        for i in range(D + 1):
            size = (n_points - i, d)
            points[j][i] = cp.Variable(size)
    
    # Boundary conditions.
    point_constraints = []
    for i, value in initial.items():
        point_constraints.append(points[0][i][0] == value)
    for i, value in final.items():
        point_constraints.append(points[n_boxes - 1][i][-1] == value)

    # Box containment.
    for j in range(n_boxes):
        Lj = np.array([L[j]] * n_points)
        Uj = np.array([U[j]] * n_points)
        point_constraints.append(points[j][0] >= Lj)
        point_constraints.append(points[j][0] <= Uj)

    # Continuity and differentiability.
    for j in range(n_boxes):
        if j < n_boxes - 1:
            for i in range(D + 1):
                point_constraints.append(points[j][i][-1] == points[j + 1][i][0])

    # Cost function.
    point_costs = [0 for j in range(n_boxes)]
    duration_cost = 0
    for i, ai in alpha.items():
        if ai > 0:
            A = ai * l2_matrix(n_points - i - 1, d)
            A_chol = np.linalg.cholesky(A).T
            for j in range(n_boxes):
                p = cp.vec(points[j][i], order='C')
                point_costs[j] += cp.quad_form(p, A)
                p = cp.vec(points[j][i - 1][1:] - points[j][i - 1][:-1], order='C') * (n_points - i)
                duration_cost += cp.quad_over_lin(A_chol @ p, durations[j])

    problem = {}
    problem['points'] = points
    problem['durations'] = durations
    problem['point_costs'] = point_costs
    problem['duration_cost'] = duration_cost
    problem['point_constraints'] = point_constraints
    problem['duration_constraints'] = duration_constraints

    return problem

def optimize_shape(problem, nom_durations):

    points = problem['points']
    costs = problem['point_costs']
    constraints = problem['point_constraints']

    # Bezier dynamics.
    n_boxes = len(points)
    n_points = points[0][0].shape[0]
    D = len(points[0]) - 1
    additional_constraints = []
    for j in range(n_boxes):
        for i in range(D):
            ci = nom_durations[j] / (n_points - i - 1)
            additional_constraints.append(points[j][i][1:] - points[j][i][:-1] == ci * points[j][i + 1])

    # Cost function.
    cost = sum([costs[j] * nom_durations[j] for j in range(n_boxes)])

    # Solve problem.
    prob = cp.Problem(cp.Minimize(cost), constraints + additional_constraints)
    prob.solve(solver='CLARABEL')

    # Reconstruct path.
    beziers = []
    a = 0
    for j in range(len(points)):
        b = a + nom_durations[j]
        beziers.append(BezierCurve(points[j][0].value, a, b))
        a = b
    path = CompositeBezierCurve(beziers)

    # Solution statistics.
    sol_stats = {}
    sol_stats['cost'] = prob.value
    sol_stats['runtime'] = prob.solver_stats.solve_time

    # Optimal values of the control points.
    sol_stats['points'] = {}
    for j, points_j in points.items():
        sol_stats['points'][j] = {}
        for i, points_ji in points_j.items():
            sol_stats['points'][j][i] = points_ji.value

    return path, sol_stats

def optimize_shape_and_timing(problem, nom_points, nom_durations, kappa):

    points = problem['points']
    durations = problem['durations']
    cost = problem['duration_cost']
    constraints = problem['point_constraints'] + problem['duration_constraints']

    # Durations.
    additional_constraints = []
    for duration, nom_duration in zip(durations, nom_durations):
        additional_constraints.append(duration >= nom_duration / (1 + kappa))
        additional_constraints.append(duration <= nom_duration * (1 + kappa))

    # Bezier dynamics.
    n_boxes = len(points)
    n_points = points[0][0].shape[0]
    D = len(points[0]) - 1
    for j in range(n_boxes):
        for i in range(D):
            ci = 1 / (n_points - i - 1)
            linearization = - nom_durations[j] * nom_points[j][i + 1] \
                            + durations[j] * nom_points[j][i + 1] \
                            + nom_durations[j] * points[j][i + 1]
            additional_constraints.append(points[j][i][1:] - points[j][i][:-1] == ci * linearization)

    # Solve problem.
    prob = cp.Problem(cp.Minimize(cost), constraints + additional_constraints)
    prob.solve(solver='CLARABEL')

    new_durations = durations.value
    kappa_1 = max(np.divide(new_durations, nom_durations) - 1)
    kappa_2 = max(np.divide(nom_durations, new_durations) - 1)
    kappa_max = max(kappa_1, kappa_2)

    # Solution statistics.
    sol_stats = {}
    sol_stats['cost'] = prob.value
    sol_stats['runtime'] = prob.solver_stats.solve_time

    return new_durations, kappa_max, sol_stats
