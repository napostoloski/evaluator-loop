"""evloop: an AI-enabled loop for locating the roots of evaluator bias.

Prototype for the Organization Science special issue proposal "Opening the Evaluator's
Black Box". See README.md.
"""
import os

# The matrices here are small (about 156 x 600). Multi-threaded BLAS makes them slower,
# sometimes by orders of magnitude on shared machines, so we pin BLAS to one thread.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
try:  # also applies if numpy was imported before evloop
    from threadpoolctl import threadpool_limits as _limits
    _limits(1)
except Exception:  # pragma: no cover
    pass

__version__ = "0.1.0"
