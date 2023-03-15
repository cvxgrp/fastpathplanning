import fpp
import numpy as np

L = np.random.randn(100, 3)
U = L + 2 * np.random.rand(100, 3)

safeset = fpp.SafeSet(lower=L, upper=U, verbose=True)
"""
> Computing the intersection of XXX boxes...
> Found XXX intersections of boxes in XXX seconds.
> Optimizing representative points of intersections...
> Done in XXX seconds.
"""

# safeset.plot() maybe
safeset.save_to_file('model.fpp')

safeset_from_file = fpp.load_file('model.fpp')
duration = 100
prob = fpp.Problem(cost_weights=np.array([0, 1.0, 0, 100.0]), # Value minimal jerk 100x more than speed
                   duration=duration, # Take 100 seconds
                   safeset=safeset_from_file)

path = prob.solve(initial_point=np.array([0, 0, 0]),
                  terminal_point=np.array([1, 1, 1]),
                  initial_derivatives=None, # Could require that we start with a particular velocity/accel
                  terminal_derivatives=None, # Could require that we end with a particular velocity/accel
                  )

sample_times = np.linspace(0, duration)
sampled_positions = path(sample_times)
velocity = path.deriv(1)
sampled_velocities = velocity(sample_times)
