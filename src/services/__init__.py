"""
Services package — business logic orchestration layer.

Services sit between the API/web layer and the data access layer.
They encapsulate domain logic, coordinate multiple repositories
and external APIs, and are the primary unit for integration tests.

Modules
-------
prediction_service
    Coordinates model inference, feature building, and result storage.
training_service
    Manages model training lifecycle (scheduling, versioning, evaluation).
betting_service
    Value betting calculations, Kelly criterion, bankroll management.
"""

from src.services.prediction_service import PredictionService
from src.services.training_service import TrainingService

__all__ = [
    "PredictionService",
    "TrainingService",
]
