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
        self.M = points.shape[0] - 1
        self.d = points.shape[1]
        self.a = a
        self.b = b
        self.duration = b - a

    def __call__(self, t):

        c = np.array([self.berstein(t, n) for n in range(self.M + 1)])
        return c.T.dot(self.points)

    def berstein(self, t, n):

        c1 = binom(self.M, n)
        c2 = (t - self.a) / self.duration 
        c3 = (self.b - t) / self.duration
        value = c1 * c2 ** n * c3 ** (self.M - n) 

        return value

    def start_point(self):

        return self.points[0]

    def end_point(self):

        return self.points[-1]
        
    def derivative(self):

        points = (self.points[1:] - self.points[:-1]) * (self.M / self.duration)

        return BezierCurve(points, self.a, self.b)

    def l2_squared(self):

        A = l2_matrix(self.M, self.d)
        p = self.points.flatten()

        return p.dot(A.dot(p)) * self.duration

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

def l2_matrix(M, d):

    A = np.zeros((M + 1, M + 1))
    for m in range(M + 1):
        for n in range(M + 1):
            A[m, n] = binom(M, m) * binom(M, n) / binom(2 * M, m + n)
    A /= (2 * M + 1)
    A_kron = np.kron(A, np.eye(d))

    return A_kron

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

    def l2_squared(self):

        return sum(bez.l2_squared() for bez in self.beziers)

    def plot2d(self, **kwargs):

        for bez in self.beziers:
            bez.plot2d(**kwargs)

    def scatter2d(self, **kwargs):

        for bez in self.beziers:
            bez.scatter2d(**kwargs)

    def plot_2dpolygon(self, **kwargs):

        for bez in self.beziers:
            bez.plot_2dpolygon(**kwargs)
