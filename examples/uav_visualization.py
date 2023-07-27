import numpy as np

from meshcat import Visualizer
from meshcat.geometry import Box, Sphere, Cylinder, MeshLambertMaterial
from meshcat.transformations import translation_matrix, rotation_matrix
from matplotlib.colors import to_hex as _to_hex


to_hex = lambda rgb: '0x' + _to_hex(rgb)[1:]


class EnvironmentVisualizer(Visualizer):

    def __init__(self):

        super().__init__()
        self['/Background'].set_property('visible', False)

    def cube(vis, name, c, r, color=(1,0,0), opacity=1):

        return _cube(vis, name, c, r, color, opacity)

    def box(self, name, l, u, color=(1,0,0), opacity=1):

        return _box(self, name, l, u, color, opacity)

    def sphere(self, name, c, r, color=(1,0,0), opacity=1):

        return _sphere(self, name, c, r, color, opacity)

    def cylinder(self, name, c1, c2, r, color=(1,0,0), opacity=1):

        return _cylinder(self, name, c1, c2, r, color, opacity)

    def capsule(self, name, c1, c2, r, color=(1,0,0), opacity=1):

        return _capsule(self, name, c1, c2, r, color, opacity)

    def trajectory(self, name, traj, r=.005, color=(0,0,1)):

        return _trajectory(self, name, traj, r, color)

def _cube(vis, name, c, r, color, opacity):
    
    c = np.array(c, dtype=float)
    color = to_hex(color)
    material = MeshLambertMaterial(color=color, opacity=opacity)
    cube = vis[name]
    cube.set_object(Box(2 * r * np.ones(3)), material)

    cube.set_transform(translation_matrix(c))

    return cube

def _box(vis, name, l, u, color, opacity):
    
    l = np.array(l, dtype=float)
    u = np.array(u, dtype=float)
    color = to_hex(color)
    material = MeshLambertMaterial(color=color, opacity=opacity)
    box = vis[name]
    box.set_object(Box(u - l), material)

    c = (u - l) / 2
    box.set_transform(translation_matrix(l + c))
    
    return box

def _sphere(vis, name, c, r, color, opacity):
    
    c = np.array(c, dtype=float)
    color = to_hex(color)
    material = MeshLambertMaterial(color=color, opacity=opacity)
    sphere = vis[name]
    sphere.set_object(Sphere(r), material)

    sphere.set_transform(translation_matrix(c))
    
    return sphere

def _cylinder(vis, name, c1, c2, r, color, opacity):
    
    c1 = np.array(c1, dtype=float)
    c2 = np.array(c2, dtype=float)
    l = np.linalg.norm(c1 - c2)
    color = to_hex(color)
    material = MeshLambertMaterial(color=color, opacity=opacity)
    cylinder = vis[name]
    cylinder.set_object(Cylinder(l, r), material)

    c = (c2 + c1) / 2
    d = (c2 - c1) / l
    angle = np.arccos(d[1])
    if np.isclose(angle % np.pi, 0):
        ax = np.array([1, 0, 0])
    else:
        ax = np.array([d[2], 0, -d[0]])
    R = rotation_matrix(angle, ax)
    T = translation_matrix(c)
    cylinder.set_transform(T.dot(R))

    return cylinder

def _capsule(vis, name, c1, c2, r, color, opacity):
    
    cylinder = _cylinder(vis, name + '_cylinder', c1, c2, r, color, opacity)
    sphere1 = _sphere(vis, name + '_sphere1', c1, r, color, opacity)
    sphere2 = _sphere(vis, name + '_sphere2', c2, r, color, opacity)
    
    return cylinder, sphere1, sphere2

def _trajectory(vis, name, traj, r, color):

    capsules = []
    for i, (x, y) in enumerate(zip(traj[:-1], traj[1:])):
        capsules.append(vis.capsule(name + f'_{i}', x, y, r, color))

    return capsules


class Village(EnvironmentVisualizer):

    def __init__(self):

        super().__init__()
        self['/Grid'].set_property('visible', False)

    def ground(self, side, color=(1, 1, 1)):
        
        l = [-.5, -.5, -.02]
        u = [side + .5, side + .5, -.01]
        ground = self.box('ground', l, u, color)
        
        return ground

    def platform(self, name, x, y, r, color=(1, 1, 1)):

        c1 = np.array([x, y, -.01])
        c2 = np.array([x, y, -.005])
        platform = self.cylinder(name, c1, c2, r, color)
        
        return platform

    def bush(self, name, c, r, h):
        
        c = np.array(c, dtype=float)

        l = [c[0] - r, c[1] - r, 0]
        u = [c[0] + r, c[1] + r, h]
        color = (0, .5, 0)
        bush = self.box(name, l, u, color)
        
        return bush

    def tree(self, name, c, r, r_trunk):
        
        c = np.array(c, dtype=float)

        l = [c[0] - r_trunk, c[1] - r_trunk, 0]
        u = [c[0] + r_trunk, c[1] + r_trunk, c[2] - r]
        color = (.7, .35, 0)
        trunk = self.box(name + '_trunk', l, u, color)

        color = (.2, .8, .2)
        foliage = self.cube(name + '_foliage', c, r, color)
        
        return trunk, foliage

    def building(self, name, c, r, n):

        c = np.array(c, dtype=float)
        r = np.array(r, dtype=float)
        n = np.array(n, dtype=int)

        h = 2 * r[2]
        l = [c[0] - r[0], c[1] - r[1], 0]
        u = [c[0] + r[0], c[1] + r[1], h]
        color = (.8, .8, .8)
        body = self.box(name + '_body', l, u, color)

        roof_ratio = 1 / 50
        l[2] = h 
        u[2] = h * (1 + roof_ratio)
        color = (.7, .0, .0)
        roof = self.box(name + '_roof', l, u, color)

        windows = []
        eps = 1e-2
        wcolor = (.3, .6, 1)
        fcolor = (0, 0, 0)
        d = 2 * r / n # Distance between windows in all directions.
        wr = d / 3.5 # Window radius in all directions.
        f = wr / 5 # Frame size in all directions.

        dw = np.array([wr[0], r[1] + 2 * eps, wr[2]])
        df = [f[0], - eps, f[2]]
        for i in range(n[0]):
            for j in range(n[2]):
                cij = np.array([c[0] - r[0] + d[0] * (i + .5), c[1], d[2] * (j + .5)])
                lij = cij - dw
                uij = cij + dw
                windows.append(self.box(name + f'_window_x_{i}_{j}', lij, uij, wcolor))
                lij -= df
                uij += df
                windows.append(self.box(name + f'_frame_x_{i}_{j}', lij, uij, fcolor))

        dw = np.array([r[0] + 2 * eps, wr[1], wr[2]])
        df = [- eps, f[1], f[2]]
        for i in range(n[1]):
            for j in range(n[2]):
                cij = np.array([c[0], c[1] - r[1] + d[1] * (i + .5), d[2] * (j + .5)])
                lij = cij - dw
                uij = cij + dw
                windows.append(self.box(name + f'_window_y_{i}_{j}', lij, uij, wcolor))

                lij -= df
                uij += df
                windows.append(self.box(name + f'_frame_y_{i}_{j}', lij, uij, fcolor))
        
        return body, roof, windows

    def uav(self, name='uav', c=[0, 0, 0]):
        
        c = np.array(c, dtype=float)
        size = np.array([.03, .03, .005])

        l = c - size
        u = c + size
        color = (1, 0, 0)
        opacity = 1
        uav = _box(self, name, l, u, color, opacity)

        r = .025
        d1 = np.array([0, 0, .02])
        d2 = np.array([0, 0, .025])
        color_prop = (.5, .5, .5)
        for i, b in enumerate([[1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0]]):
            ci = .05 * np.array(b)
            di = .025 * np.array(b)
            propi = _cylinder(uav, f'prop{i}', ci + d1, ci + d2, r, color_prop, opacity)
            beami = _cylinder(uav, f'beam{i}', di, ci + d1, r/5, color, opacity)
        
        return uav
