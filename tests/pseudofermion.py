from lfox.fermions import PseudoFermion, wilson


class YukawaFermion(PseudoFermion):
    """Two degenerate flavors of a Yukawa-Wilson fermion, by one pseudofermion:

        M(phi) psi = wilson(psi, kappa) + g phi psi

    Everything else -- the action, the heatbath, the force -- is the library's.
    """

    kappa: float
    g: float

    def fermion_matrix(self, phi):
        return lambda psi: wilson(psi, self.kappa) + self.g * phi * psi
