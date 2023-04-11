import numpy as np
from fastpathplanning.boxes import Box, BoxCollection
from fastpathplanning.polygonal import iterative_planner
from fastpathplanning.smooth import optimize_bezier_with_retiming

class SafeSet:

    def __init__(self, L, U, verbose=True):

        if verbose:
            print(f'Preprocessing started.')

        assert L.shape == U.shape
        boxes = [Box(l, u) for l, u in zip(L, U)]
        self.B = BoxCollection(boxes, verbose)
        self.G = self.B.line_graph(verbose)

    def plot2d(self, **kwargs):

        self.B.plot2d(**kwargs)

def plan(S, p_init, p_term, T, alpha, der_init={}, der_term={}, verbose=True):

    if verbose:
        print('Polygonal phase started.')

    discrete_planner, runtime = S.G.shortest_path(p_term)
    box_seq, length, runtime = discrete_planner(p_init)
    box_seq, traj, length, solver_time = iterative_planner(S.B, p_init, p_term, box_seq, verbose)

    if verbose:
        print('\nSmooth phase started.')

    # Fix box sequence.
    L = np.array([S.B.boxes[i].l for i in box_seq])
    U = np.array([S.B.boxes[i].u for i in box_seq])

    # Cost coefficients.
    alpha = {i + 1: ai for i, ai in enumerate(alpha)}

    # Boundary conditions.
    initial = {0: p_init} | der_init
    final = {0: p_term} | der_term

    # Initialize transition times.
    durations = np.linalg.norm(traj[1:] - traj[:-1], axis=1)
    durations *= T / sum(durations)

    path, sol_stats = optimize_bezier_with_retiming(L, U, durations, alpha, initial, final, verbose=True)

    return path
