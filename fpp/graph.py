import numpy as np
import scipy as sp
import cvxpy as cp
import networkx as nx

from time import time
from itertools import product

class LineGraph(nx.Graph):

    def __init__(self, B, points, *args, **kwargs):

        # Initialize and store boxes.
        super().__init__(*args, **kwargs)
        self.B = B

        # Compute line graph using networkx.
        inters_graph = nx.Graph()
        inters_graph.add_nodes_from(B.inters.keys())
        for k, k_inters in B.inters.items():
            k_inters_unique = [l for l in k_inters if l > k]
            inters_graph.add_edges_from(product([k], k_inters_unique))
        line_graph = nx.line_graph(inters_graph)
        self.add_nodes_from(line_graph.nodes)
        self.add_edges_from(line_graph.edges)
        self.v2i = {v: i for i, v in enumerate(self.nodes)}
        self.i2v = {i: v for i, v in enumerate(self.nodes)}

        # Pair each vertex with the corresponding box intersection.
        for v in self.nodes:
            boxk = B.boxes[v[0]]
            boxl = B.boxes[v[1]]
            self.nodes[v]['box'] = boxk.intersect(boxl)

        # Place representative point in each box intersection.
        if points == 'centers':
            self.points_in_centers()
        elif points == 'optimize':
            self.optimize_points()
        elif points == 'weiszfeld':
            self.weiszfeld_points()
        else:
            raise ValueError

        # Assign fixed length to each edge of the line graph.
        for e in self.edges:
            pu = self.nodes[e[0]]['point']
            pv = self.nodes[e[1]]['point']
            self.edges[e]['weight'] = np.linalg.norm(pv - pu)

        # Store adjacency matrix for scipy's shortest-path algorithms.
        self.adj_mat = nx.to_scipy_sparse_array(self)
        
    def points_in_centers(self):

        for v in self.nodes:
            self.nodes[v]['point'] = self.nodes[v]['box'].c

    def optimize_points(self):

        x = {}
        constraints = []
        for v in self.nodes:
            box = self.nodes[v]['box']
            x[v] = cp.Variable(box.d)
            constraints.append(x[v] >= box.l)
            constraints.append(x[v] <= box.u)
        cost = sum(cp.norm(x[e[1]] - x[e[0]], 2) for e in self.edges)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver='ECOS')

        for v in self.nodes:
            self.nodes[v]['point'] = x[v].value

    def weiszfeld_points(self, reps=1e-3, rtol=1e-3, max_iter=50, iter_gap=3):
        '''TODO: improve the following as in "The multivariate L1-median and
        associated data depth."'''

        P = np.array([self.nodes[v]['box'].c for v in self.nodes])
        L = np.array([self.nodes[v]['box'].l for v in self.nodes])
        U = np.array([self.nodes[v]['box'].u for v in self.nodes])

        def evaluate(P):
            gaps = [P[self.v2i[e[1]]] - P[self.v2i[e[0]]] for e in self.edges]
            distances = np.linalg.norm(gaps, axis=1)
            return sum(distances)
        def project(P):
            return np.minimum(np.maximum(P, L), U)
        eps = evaluate(P) / P.shape[0] * reps
        def smooth_norm(x):
            return (x.dot(x) + eps) ** .5
        adj = {u: [self.v2i[v] for v in self.adj[u]] for u in self.nodes}
        def update(P):
            Q = np.zeros(P.shape)
            for i, u in enumerate(self.nodes):
                if len(adj[u]) > 0:
                    neighbors = np.array([P[j] for j in adj[u]])
                    Pi = P[i]
                    distances = np.array([smooth_norm(Pi - Pj) for Pj in neighbors])
                    num = sum((neighbors.T / distances).T)
                    den = sum(1 / distances)
                    Q[i] = num / den
            return Q
        
        cost = evaluate(P)
        for k in range(max_iter):
            P = project(update(P))
            if k % iter_gap == 0:
                new_cost = evaluate(P)
                if new_cost > cost * (1 - rtol):
                    break
                cost = new_cost

        for i, v in enumerate(self.nodes):
            self.nodes[v]['point'] = P[i]

    @staticmethod
    def node(k, l):

        return (k, l) if k < l else (l, k)

    def shortest_path(self, goal):

        tic = time()

        rows = []
        data = []
        for k in self.B.contain(goal):
            for l in self.B.inters[k]:
                v = self.node(k, l)
                i = self.v2i[v]
                rows.append(i)
                pv = self.nodes[v]['point']
                data.append(np.linalg.norm(goal - pv))
        cols = [0] * len(rows)
        shape = (len(self.nodes), 1)
        adj_col = sp.sparse.csr_matrix((data, (rows, cols)), shape)
        adj_mat = sp.sparse.bmat([[self.adj_mat, adj_col], [adj_col.T, None]])

        dist, succ = sp.sparse.csgraph.dijkstra(
            csgraph=adj_mat,
            directed=False,
            return_predecessors=True,
            indices=-1
        )

        planner = lambda start: self._planner_all_to_one(start, dist, succ)

        return planner, time() - tic

    def _planner_all_to_one(self, start, dist, succ):

        tic = time()

        length = np.inf
        for k in self.B.contain(start):
            for l in self.B.inters[k]:
                v = self.node(k, l)
                i = self.v2i[v]
                dist_vg = dist[i]
                if np.isinf(dist_vg):
                    return None, None, time() - tic
                dist_sv = np.linalg.norm(self.nodes[v]['point'] - start)
                dist_sg = dist_sv + dist_vg
                if dist_sg < length:
                    length = dist_sg
                    first_box = k
                    first_vertex = i

        box_sequence = self._succ_to_box_sequence(succ, first_box, first_vertex)

        return box_sequence, length, time() - tic

    def all_pairs_shortest_path(self):

        tic = time()

        dist, succ = sp.sparse.csgraph.floyd_warshall(
            csgraph=self.adj_mat,
            directed=False,
            return_predecessors=True
        )
        planner = lambda start, goal: self._planner_all_pairs(start, goal, dist, succ)

        runtime = time() - tic

        return planner, runtime

    def _planner_all_pairs(self, start, goal, dist, succ):

        tic = time()

        length = np.inf
        for k in self.B.contain(start):
            for l in self.B.inters[k]:
                for p in self.B.contain(goal):
                    for q in self.B.inters[p]:
                        u = self.node(k, l)
                        v = self.node(p, q)
                        i = self.v2i[u]
                        j = self.v2i[v]
                        dist_uv = dist[i, j]
                        if np.isinf(dist_uv):
                            return None, None, time() - tic
                        dist_su = np.linalg.norm(self.nodes[u]['point'] - start)
                        dist_vg = np.linalg.norm(self.nodes[v]['point'] - goal)
                        dist_sg = dist_su + dist_uv + dist_vg
                        if dist_sg < length:
                            length = dist_sg
                            first_box = k
                            last_box = p
                            first_vertex = i
                            last_vertex = j

        box_sequence = self._succ_to_box_sequence(succ[last_vertex], first_box, first_vertex)
        box_sequence.append(last_box)

        return box_sequence, length, time() - tic

    def _succ_to_box_sequence(self, succ, first_box, first_vertex):

        box_sequence = [first_box]
        i = first_vertex
        while succ[i] >= 0:
            v = self.i2v[i]
            j = succ[i]
            if succ[j] >= 0:
                w = self.i2v[j]
                while box_sequence[-1] in w:
                    i = j
                    v = w
                    j = succ[i]
                    w = self.i2v[j]
            if v[0] == box_sequence[-1]:
                box_sequence.append(v[1])
            else:
                box_sequence.append(v[0])
            i = succ[i]

        return box_sequence

    def plot(self, **kwargs):
        import matplotlib.pyplot as plt

        options = {'marker':'o', 'markerfacecolor':'w', 'markeredgecolor':'k', 'c':'k'}
        options.update(kwargs)

        if 'label' in options:
            plt.plot([np.nan], [np.nan], **options)
            options.pop('label')

        for e in self.edges:
            start = self.nodes[e[0]]['point']
            stop = self.nodes[e[1]]['point']
            P = np.array([start, stop])
            plt.plot(*P.T, **options)
