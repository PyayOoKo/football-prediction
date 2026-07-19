"""Probability calibration — calibration methods and wrappers.

See the sub-modules for detailed documentation:

* ``utils`` — shared validation / normalisation helpers
* ``calibrators`` — calibrator classes (Platt, Isotonic, HybridTail, Temperature)
* ``wrappers`` — model wrappers + convenience functions
* ``reporting`` — calibration curve and report utilities
"""

# Re-export everything for backward compatibility.
# All existing ``from src.calibration import X`` statements continue to work.

from src.calibration.utils import (
    _renormalise_probs,
    _validate_probs_input,
    renormalise_probs,
    validate_probs_input,
)

from src.calibration.calibrators import (
    HybridTailCalibrator,
    IsotonicRegressionCalibrator,
    PlattScalingCalibrator,
    TemperatureScalingCalibrator,
)

from src.calibration.wrappers import (
    CalibratedModel,
    CalibratedStatsModel,
    CalibratedTemperatureWrapper,
    _fit_calibrators,
    calibrate_model,
)

from src.calibration.reporting import (
    calibration_curve,
    calibration_report,
)

__all__ = [
    # Utils
    "validate_probs_input",
    "_validate_probs_input",
    "renormalise_probs",
    "_renormalise_probs",
    # Calibrators
    "IsotonicRegressionCalibrator",
    "PlattScalingCalibrator",
    "HybridTailCalibrator",
    "TemperatureScalingCalibrator",
    # Wrappers
    "CalibratedTemperatureWrapper",
    "CalibratedStatsModel",
    "CalibratedModel",
    "_fit_calibrators",
    "calibrate_model",
    # Reporting
    "calibration_curve",
    "calibration_report",
]
