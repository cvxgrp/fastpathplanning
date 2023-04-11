import numpy as np
import cvxpy as cp
from time import time
from itertools import accumulate
from bisect import bisect
from scipy.special import binom

class BezierCurve:

    def __init__(self, points, a=0, b=1):

        assert b > a

        self.points = points
        self.h = points.shape[0] - 1
        self.d = points.shape[1]
        self.a = a
        self.b = b
        self.duration = b - a

    def __call__(self, t):

        c = np.array([self.berstein(t, n) for n in range(self.h + 1)])
        return c.T.dot(self.points)

    def berstein(self, t, n):

        c1 = binom(self.h, n)
        c2 = (t - self.a) / self.duration 
        c3 = (self.b - t) / self.duration
        value = c1 * c2 ** n * c3 ** (self.h - n) 

        return value

    def start_point(self):

        return self.points[0]

    def end_point(self):

        return self.points[-1]
        
    def derivative(self):

        points = (self.points[1:] - self.points[:-1]) * (self.h / self.duration)

        return BezierCurve(points, self.a, self.b)

    def l2_squared(self):

        A = np.zeros((self.h + 1, self.h + 1))
        for m in range(self.h + 1):
            for n in range(self.h + 1):
                A[m, n] = binom(self.h, m) * binom(self.h, n) / binom(2 * self.h, m + n)
        A *= self.duration / (2 * self.h + 1)
        A = np.kron(A, np.eye(self.d))

        p = self.points.flatten()

        return p.dot(A.dot(p))

    def plot2d(self, samples=51, **kwargs):

        import matplotlib.pyplot as plt

        options = {'c':'b'}
        options.update(kwargs)
        t = np.linspace(self.a, self.b, samples)
        plt.plot(*self(t).T, **options)

    def scatter2d(self, **kwargs):

        import matplotlib.pyplot as plt

        options = {'fc':'orange', 'ec':'k', 'zorder':3}
        options.update(kwargs)
        plt.scatter(*self.points.T, **options)

    def plot_2dpolygon(self, **kwargs):

        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from scipy.spatial import ConvexHull

        options = {'fc':'lightcoral'}
        options.update(kwargs)
        hull = ConvexHull(self.points)
        ordered_points = hull.points[hull.vertices]
        poly = Polygon(ordered_points, **options)
        plt.gca().add_patch(poly)


class CompositeBezierCurve:

    def __init__(self, beziers):

        for bez1, bez2 in zip(beziers[:-1], beziers[1:]):
            assert bez1.b == bez2.a
            assert bez1.d == bez2.d

        self.beziers = beziers
        self.N = len(self.beziers)
        self.d = beziers[0].d
        self.a = beziers[0].a
        self.b = beziers[-1].b
        self.duration = self.b - self.a
        self.transition_times = [self.a] + [bez.b for bez in beziers]

    def find_segment(self, t):

        return min(bisect(self.transition_times, t) - 1, self.N - 1)

    def __call__(self, t):

        i = self.find_segment(t)

        return self.beziers[i](t)

    def start_point(self):

        return self.beziers[0].start_point()

    def end_point(self):

        return self.beziers[-1].end_point()

    def derivative(self):

        return CompositeBezierCurve([b.derivative() for b in self.beziers])

    def bound_on_integral(self, f):

        return sum(bez.bound_on_integral(f) for bez in self.beziers)

    def plot2d(self, **kwargs):

        for bez in self.beziers:
            bez.plot2d(**kwargs)

    def scatter2d(self, **kwargs):

        for bez in self.beziers:
            bez.scatter2d(**kwargs)

    def plot_2dpolygon(self, **kwargs):

        for bez in self.beziers:
            bez.plot_2dpolygon(**kwargs)

def optimize_bezier(L, U, durations, alpha, initial, final,
    n_points=None, **kwargs):

    # Problem size.
    n_boxes, d = L.shape
    D = max(alpha)
    assert max(initial) <= D
    assert max(final) <= D
    if n_points is None:
        n_points = (D + 1) * 2

    # Control points of the curves and their derivatives.
    points = {}
    for k in range(n_boxes):
        points[k] = {}
        for i in range(D + 1):
            size = (n_points - i, d)
            points[k][i] = cp.Variable(size)

    # Boundary conditions.
    constraints = []
    for i, value in initial.items():
        constraints.append(points[0][i][0] == value)
    for i, value in final.items():
        constraints.append(points[n_boxes - 1][i][-1] == value)

    # Loop through boxes.
    cost = 0
    for k in range(n_boxes):

        # Box containment.
        Lk = np.array([L[k]] * n_points)
        Uk = np.array([U[k]] * n_points)
        constraints.append(points[k][0] >= Lk)
        constraints.append(points[k][0] <= Uk)

        # Bezier dynamics.
        for i in range(D):
            h = n_points - i - 1
            ci = durations[k] / h
            constraints.append(points[k][i][1:] - points[k][i][:-1] == ci * points[k][i + 1])

        # Continuity and differentiability.
        if k < n_boxes - 1:
            for i in range(D + 1):
                constraints.append(points[k][i][-1] == points[k + 1][i][0])

        # Cost function.
        for i, ai in alpha.items():
            h = n_points - 1 - i
            A = np.zeros((h + 1, h + 1))
            for m in range(h + 1):
                for n in range(h + 1):
                    A[m, n] = binom(h, m) * binom(h, n) / binom(2 * h, m + n)
            A *= durations[k] / (2 * h + 1)
            A = np.kron(A, np.eye(d))
            p = cp.vec(points[k][i], order='C')
            cost += ai * cp.quad_form(p, A)

    # Solve problem.
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver='CLARABEL')

    # Reconstruct trajectory.
    beziers = []
    a = 0
    for k in range(n_boxes):
        b = a + durations[k]
        beziers.append(BezierCurve(points[k][0].value, a, b))
        a = b
    path = CompositeBezierCurve(beziers)

    # Reconstruct costs.
    cost_breakdown = {}
    for k in range(n_boxes):
        cost_breakdown[k] = {}
        bez = beziers[k]
        for i in range(1, D + 1):
            bez = bez.derivative()
            if i in alpha:
                cost_breakdown[k][i] = alpha[i] * bez.l2_squared()

    # Solution statistics.
    sol_stats = {}
    sol_stats['cost'] = prob.value
    sol_stats['runtime'] = prob.solver_stats.solve_time
    sol_stats['cost_breakdown'] = cost_breakdown

    return path, sol_stats


def retiming(kappa, costs, durations, **kwargs):

    # Decision variables.
    n_boxes = max(costs) + 1
    eta = cp.Variable(n_boxes)
    eta.value = np.ones(n_boxes)
    constr = [durations @ eta == sum(durations)]

    # Scale costs from previous trajectory.
    cost = 0
    for i, ci in costs.items():
        for j, cij in ci.items():
            power = 2 * j - 1
            cost += cij * cp.power(eta[i], - power)

    # Trust region.
    if not np.isinf(kappa):
        constr.append(eta[1:] - eta[:-1] <= kappa)
        constr.append(eta[:-1] - eta[1:] <= kappa)
        
    # Solve SOCP and get new durarations.
    prob = cp.Problem(cp.Minimize(cost), constr)
    prob.solve(solver='CLARABEL')
    new_durations = np.multiply(eta.value, durations)

    # New candidate for kappa.
    kappa_max = max(np.abs(eta.value[1:] - eta.value[:-1]))

    return new_durations, prob.solver_stats.solve_time, kappa_max


def optimize_bezier_with_retiming(L, U, durations, alpha, initial, final,
    omega=3, kappa_min=1e-2, verbose=False, **kwargs):

    # Solve initial Bezier problem.
    path, sol_stats = optimize_bezier(L, U, durations, alpha, initial, final, **kwargs)
    cost = sol_stats['cost']
    cost_breakdown = sol_stats['cost_breakdown']

    if verbose:
        print(f'Iter. 0: cost {np.round(cost, 3)}.')

    # Lists to populate.
    costs = [cost]
    paths = [path]
    durations_iter = [durations]
    bez_runtimes = [sol_stats['runtime']]
    retiming_runtimes = []

    # Iterate retiming and Bezier.
    kappa = np.inf
    n_iters = 0
    i = 1
    while True:
        n_iters += 1

        # Retime.
        new_durations, runtime, kappa_max = retiming(kappa, cost_breakdown,
            durations, **kwargs)
        durations_iter.append(new_durations)
        retiming_runtimes.append(runtime)

        # Improve Bezier curves.
        path_new, sol_stats = optimize_bezier(L, U, new_durations,
            alpha, initial, final, **kwargs)
        cost_new = sol_stats['cost']
        cost_breakdown_new = sol_stats['cost_breakdown']
        costs.append(cost_new)
        paths.append(path_new)
        bez_runtimes.append(sol_stats['runtime'])
        if verbose:
            print(f'Iter. {i}, cost {np.round(cost_new, 3)}, kappa {kappa}.')

        # If retiming improved the trajectory.
        if cost_new < cost:
            durations = new_durations
            path = path_new
            cost = cost_new
            cost_breakdown = cost_breakdown_new

        if kappa < kappa_min:
            break
        kappa = kappa_max / omega
        i += 1

    runtime = sum(bez_runtimes) + sum(retiming_runtimes)
    if verbose:
        print(f'Terminated in {i} iterations.')
        print(f'Final cost is {np.round(cost, 3)}.')
        print(f'Solver time was {np.round(runtime, 5)}.')

    # Solution statistics.
    sol_stats = {}
    sol_stats['cost'] = cost
    sol_stats['n_iters'] = n_iters
    sol_stats['costs'] = costs
    sol_stats['paths'] = paths
    sol_stats['durations_iter'] = durations_iter
    sol_stats['bez_runtimes'] = bez_runtimes
    sol_stats['retiming_runtimes'] = retiming_runtimes
    sol_stats['runtime'] = runtime
    
    return path, sol_stats
