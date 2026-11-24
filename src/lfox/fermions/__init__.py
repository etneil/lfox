"""Fermions: gamma matrices and Dirac operators.

A spinor is a plain `LatticeField` with a "spin" axis.  See fermions_design.md.
"""

from lfox.fermions.dirac import wilson
from lfox.fermions.gamma import gamma5, gammas

__all__ = ["gamma5", "gammas", "wilson"]
