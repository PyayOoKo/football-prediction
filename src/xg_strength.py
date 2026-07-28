from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class XGStrengthModel:
    """Team attack/defence model trained on real xG data.

    Parameters are fitted via quasi-Poisson regression (minimising
    lambda - xG * ln(lambda) which is equivalent to Poisson negative
    log-likelihood without the ln(Gamma(xG+1)) constant term).

    expected_goals(home, away) => (lambda_home, lambda_away)
    """

    alpha: dict[str, float]
    beta: dict[str, float]
    gamma: float
    team_list: list[str]
    fitted: bool = False
    n_matches: int = 0

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        if not self.fitted:
            raise RuntimeError("Model not fitted")
        alpha_h = self.alpha.get(home_team, 0.0)
        beta_a = self.beta.get(away_team, 0.0)
        alpha_a = self.alpha.get(away_team, 0.0)
        beta_h = self.beta.get(home_team, 0.0)
        lam = float(np.exp(alpha_h + beta_a + self.gamma))
        mu = float(np.exp(alpha_a + beta_h))
        return lam, mu
