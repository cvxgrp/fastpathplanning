import numpy as np
import cvxpy as cp
from fastpathplanning.bezier import BezierCurve, CompositeBezierCurve, l2_matrix

def mixed_integer(L, U, alpha, p_init, p_term, T, N, der_init={}, der_term={},
                  n_points=None, **kwargs):

    initial = {0: p_init} | der_init
    final = {0: p_term} | der_term
    alpha = {i + 1: ai for i, ai in enumerate(alpha)}

    # Problem size.
    K, d = L.shape
    D = max(alpha)
    assert max(initial) <= D
    assert max(final) <= D
    if n_points is None:
        n_points = (D + 1) * 2

    # Control points of the curves and their derivatives.
    points = {}
    for j in range(N):
        points[j] = {}
        for i in range(D + 1):
            points[j][i] = cp.Variable((n_points - i, d))

    # Binary variables.
    constraints = []
    binaries = cp.Variable((N, K), boolean=True)
    for j in range(N):
        constraints.append(sum(binaries[j]) == 1)

    # Boundary conditions.
    for i, value in initial.items():
        constraints.append(points[0][i][0] == value)
    for i, value in final.items():
        constraints.append(points[N-1][i][-1] == value)

    # Box containment.
    for j in range(N):
        lj = cp.sum([np.array([L[k]] * n_points) * binaries[j, k] for k in range(K)])
        uj = cp.sum([np.array([U[k]] * n_points) * binaries[j, k] for k in range(K)])
        constraints.append(points[j][0] >= lj)
        constraints.append(points[j][0] <= uj)
        
    # Continuity and differentiability.
    for j in range(N - 1):
        for i in range(D + 1):
            constraints.append(points[j][i][-1] == points[j+1][i][0])

    # Bezier dynamics.
    for j in range(N):
        for i in range(D):
            h = n_points - i - 1
            ci = T / N / h
            constraints.append(points[j][i][1:] - points[j][i][:-1] == ci * points[j][i+1])

    # Cost function.
    cost = 0
    for i, ai in alpha.items():
        if ai > 0:
            A = l2_matrix(n_points - i - 1, d)
            for j in range(N):
                p = cp.vec(points[j][i], order='C')
                cost += ai * cp.quad_form(p, A) * T / N

    # Solve problem.
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(**kwargs)

    # Reconstruct trajectory.
    beziers = []
    for j in range(N):
        a = j * T / N
        b = (j + 1) * T / N
        beziers.append(BezierCurve(points[j][0].value, a, b))
    path = CompositeBezierCurve(beziers)

    return path
