"""Fermions: gamma matrices, Dirac operators, and pseudofermions.

A spinor is a plain `LatticeField` with a "spin" axis.  See fermions_design.md.
"""

from lfox.fermions.dirac import wilson
from lfox.fermions.gamma import gamma5, gammas
from lfox.fermions.pseudofermion import PseudoFermion

__all__ = ["PseudoFermion", "gamma5", "gammas", "wilson"]
