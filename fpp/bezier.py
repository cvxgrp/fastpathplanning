import numpy as np
import cvxpy as cp

from time import time
from itertools import accumulate
from bisect import bisect

from scipy.special import binom

class Bezier:

    def __init__(self, points, duration):

        self.points = points
        self.n = points.shape[0] - 1
        self.duration = duration

    def berstein(self, t, i):

        a = binom(self.n, i)
        b = (t / self.duration) ** i
        c = (1 - t / self.duration) ** (self.n - i) 

        return a * b * c

    def evaluate(self, t):

        c = np.array([self.berstein(t, i) for i in range(self.n + 1)])
        return c.T.dot(self.points)
        
    def derivative(self):

        points = (self.points[1:] - self.points[:-1]) * (self.n / self.duration)

        return Bezier(points, self.duration)

    def integral_of_convex(self, f):

        return sum(f(p) for p in self.points) * self.duration / (self.n + 1)
    
    def plot2d(self, m=51, **kwargs):
        import matplotlib.pyplot as plt

        options = {'c':'b'}
        options.update(kwargs)
        t = np.linspace(0, self.duration, m)
        plt.plot(*self.evaluate(t).T, **options)

    def scatter2d(self, **kwargs):
        import matplotlib.pyplot as plt

        options = {'fc':'orange', 'ec':'k', 'zorder':3}
        options.update(kwargs)
        plt.scatter(*self.points.T, **options)


class BezierTrajectory:

    def __init__(self, beziers):

        self.beziers = beziers
        time_sums = list(accumulate([b.duration for b in beziers]))
        self.start_times = [0] + time_sums[:-1]
        self.duration = time_sums[-1]

    def find_segment(self, t):

        return bisect(self.start_times, t) - 1

    def evaluate(self, t):

        i = self.find_segment(t)

        return self.beziers[i].evaluate(t - self.start_times[i])

    def derivative(self):

        return BezierTrajectory([b.derivative() for b in self.beziers])

    def plot2d(self, **kwargs):
        import matplotlib.pyplot as plt

        for bezier in self.beziers:
            bezier.plot2d(**kwargs)

    def scatter2d(self, **kwargs):
        import matplotlib.pyplot as plt

        for bezier in self.beziers:
            bezier.scatter2d(**kwargs)

def optimize_bezier(l, u, h, lam, initial, final, n_derivatives=None, **kwargs):

    # Problem size.
    if n_derivatives is None:
        n_derivatives = max(max(lam), max(initial), max(final))
    n_points = (n_derivatives + 1) * 2
    n_boxes, d = l.shape

    # Decision variables.
    P = {} # Control points of the curves and their derivatives.
    for i in range(n_boxes):
        P[i] = {}
        for j in range(n_derivatives + 1):
            P[i][j] = cp.Variable((n_points - j, d))

    # Initial conditions.
    constraints = []
    for j, value in initial.items():
        constraints.append(P[0][j][0] == value)

    # Loop through boxes.
    cost = 0
    for i in range(n_boxes):

        # Box containment.
        li = np.array([l[i]] * n_points)
        ui = np.array([u[i]] * n_points)
        constraints.append(P[i][0] >= li)
        constraints.append(P[i][0] <= ui)

        # Bezier dynamics.
        for j in range(n_derivatives):
            steps = n_points - j - 1
            cj = h[i] / steps
            constraints.append(P[i][j][1:] - P[i][j][:-1] == cj * P[i][j + 1])

        # Continuity and differentiability.
        if i < n_boxes - 1:
            for j in range(n_derivatives + 1):
                constraints.append(P[i][j][-1] == P[i + 1][j][0])

        # Cost function.
        for j, lamj in lam.items():
            cj = lamj * h[i] / (n_points - j)
            cost += cj * cp.sum_squares(P[i][j].flatten())

    # Final conditions.
    for j, value in final.items():
        constraints.append(P[n_boxes - 1][j][-1] == value)

    # Solve problem.
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(**kwargs)

    # Reconstruct trajectory.
    beziers = []
    for i in range(n_boxes):
        beziers.append(Bezier(P[i][0].value, h[i]))
    traj = BezierTrajectory(beziers)

    # Reconstruct costs.
    costs = {}
    for i in range(n_boxes):
        costs[i] = {}
        for j, lamj in lam.items():
            cj = lamj * h[i] / (n_points - j)
            points_ij = P[i][j].value.flatten()
            costs[i][j] = cj * points_ij.dot(points_ij)

    return traj, prob.value, prob.solver_stats.solve_time, costs


def retiming(gamma, costs, h, **kwargs):

    # Decision variables are the ratios z of the old durations h and the new
    # durations h_new.
    # Optimizing directly over h_new would be numerically very unstable.
    n_boxes = max(costs) + 1
    z = cp.Variable(n_boxes)
    constr = [h @ z == sum(h)]

    # Scale costs from previous trajectory.
    cost = 0
    for i, ci in costs.items():
        for j, cij in ci.items():
            power = 2 * j - 1
            cost += cij * cp.power(z[i], - power)

    # Regualization term.
    cost += gamma * cp.sum_squares(z[1:] - z[:-1])
    # dz = z[1:] - z[:-1]
    # constr += [dz >= -1 / gamma, dz <= 1 / gamma]
        
    # Solve SOCP and get new durarations.
    prob = cp.Problem(cp.Minimize(cost), constr)
    prob.solve(**kwargs)
    h_new = np.multiply(z.value, h)

    return h_new, prob.solver_stats.solve_time


def optimize_bezier_with_retiming(l, u, h, lam, initial, final, n_derivatives=None,
    gamma_min=1e-3, gamma_max=1e4, gamma_step=10, max_iter=50, verbose=False, **kwargs):

    # Solve initiali Bezier problem.
    bez, cost, runtime, coeffs = optimize_bezier(l, u, h, lam, initial, final, **kwargs)
    if verbose:
        print(f'Iter. 0, cost {np.round(cost, 3)}')

    # Iterate retiming and Bezier.
    gamma = gamma_min
    for i in range(max_iter):

        # Retime.
        h_new, runtime_i = retiming(gamma, coeffs, h, **kwargs)
        runtime += runtime_i

        # Improve Bezier curves.
        bez_new, cost_new, runtime_i, coeffs_new = optimize_bezier(l, u, h_new, lam, initial, final, **kwargs)
        runtime += runtime_i
        if verbose:
            print(f'Iter. {i+1}, cost {np.round(cost_new, 3)}, gamma {gamma}')

        # If retiming improved the trajectory.
        if cost_new < cost:
            h = h_new
            bez = bez_new
            cost = cost_new
            coeffs = coeffs_new

        # If retiming did not improve the trajectory.
        else:
            if gamma > gamma_max:
                break
            gamma *= gamma_step
            

    if verbose:
        if i == max_iter - 1:
            print(f'Reached iteration limit.')
        print(f'\nTerminated in {i+1} iterations.')
        print(f'Final cost is {cost}.')
        print(f'Solver time was {runtime}.')
            
    return bez, cost, h, runtime
