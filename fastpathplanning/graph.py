import numpy as np
import scipy as sp
import cvxpy as cp
import networkx as nx
from time import time
from itertools import product

class LineGraph(nx.Graph):

    def __init__(self, B, verbose=True, solver='CLARABEL', *args, **kwargs):

        if verbose:
            print('Computing line graph')

        # Initialize and store boxes.
        super().__init__(*args, **kwargs)
        self.B = B
        self.solver = solver

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

        if verbose:
            print(f'...line graph has {self.number_of_nodes()} vertices ' \
                f'and {self.number_of_edges()} edges')

        if verbose:
            print('Optimizing representative points')

        # Pair each vertex with the corresponding box intersection.
        for v in self.nodes:
            boxk = B.boxes[v[0]]
            boxl = B.boxes[v[1]]
            self.nodes[v]['box'] = boxk.intersect(boxl)

        # Place representative point in each box intersection.
        self.optimize_points()

        # Assign fixed length to each edge of the line graph.
        for e in self.edges:
            pu = self.nodes[e[0]]['point']
            pv = self.nodes[e[1]]['point']
            self.edges[e]['weight'] = np.linalg.norm(pv - pu)

        # Store adjacency matrix for scipy's shortest-path algorithms.
        self.adj_mat = nx.to_scipy_sparse_array(self)

        if verbose:
            print('...done')
        
    def optimize_points(self):
        tic = time()

        x = cp.Variable((self.number_of_nodes(), self.B.d))
        x.value = np.array([self.nodes[v]['box'].c for v in self.nodes])

        l = np.vstack([self.nodes[v]['box'].l for v in self.nodes])
        u = np.vstack([self.nodes[v]['box'].u for v in self.nodes])
        constraints = [x >= l, x <= u]

        A = nx.incidence_matrix(self, oriented=True)
        y = A.T.dot(x)
        cost = cp.sum(cp.norm(y, 2, axis=1))

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=self.solver)

        for i, v in enumerate(self.nodes):
            self.nodes[v]['point'] = x[i].value

        self.cvxpy_time = time() - tic - prob.solver_stats.solve_time

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

        plot_graph(self, **kwargs)

def plot_graph(G, **kwargs):

    import matplotlib.pyplot as plt

    options = {'marker':'o', 'markerfacecolor':'w', 'markeredgecolor':'k', 'c':'k'}
    options.update(kwargs)

    if 'label' in options:
        plt.plot([np.nan], [np.nan], **options)
        options.pop('label')

    for e in G.edges:
        start = G.nodes[e[0]]['point']
        stop = G.nodes[e[1]]['point']
        P = np.array([start, stop])
        plt.plot(*P.T, **options)
