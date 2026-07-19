"""
Comprehensive tests for API authentication, model metadata, and feature alignment.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi import Depends, FastAPI, Request, status
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

# ═══════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════


@pytest.fixture
def mock_request() -> MagicMock:
    """Create a minimal mock FastAPI Request."""
    request = MagicMock(spec=Request)
    request.headers = {}
    request.query_params = {}
    request.state.request_id = "test-1234"
    request.client = MagicMock()
    request.client.host = "127.0.0.1"
    return request


@pytest.fixture
def mock_bearer_credentials() -> MagicMock:
    """Create mock HTTPAuthorizationCredentials with a bearer token."""
    creds = MagicMock()
    creds.scheme = "Bearer"
    creds.credentials = "test-api-key-12345"
    return creds


# ═══════════════════════════════════════════════════════════
#  Scenario 11: Module import never crashes
# ═══════════════════════════════════════════════════════════


class TestModuleImport:
    """Verify api.auth module imports without NameError or TypeError."""

    def test_import_auth_no_error(self) -> None:
        """Module import should not raise NameError or TypeError."""
        # Use isolated env vars to ensure clean import

        # Clear any cached module
        sys.modules.pop("api.auth", None)

        # Temporarily set up clean env
        with patch.dict(
            os.environ,
            {
                "APP_ENV": "development",
                "PREDICTION_API_KEY": "test-key-for-import",
                "API_AUTH_DISABLED": "",
            },
            clear=False,
        ):
            try:
                from api import auth

                assert auth is not None
                assert hasattr(auth, "verify_api_key")
                assert hasattr(auth, "optional_auth")
                assert hasattr(auth, "rate_limiter")
            except (NameError, TypeError) as exc:
                pytest.fail(f"Module import raised {type(exc).__name__}: {exc}")


# ═══════════════════════════════════════════════════════════
#  Scenario 1 & 2: No key configured
# ═══════════════════════════════════════════════════════════


class TestNoKeyConfigured:
    """Auth behavior when PREDICTION_API_KEY is not set."""

    @pytest.mark.parametrize(
        "env_vars,expected_descriptor",
        [
            # Scenario 1: Dev mode — auth returns dev-mode (lenient)
            ({"APP_ENV": "development"}, "dev-mode"),
        ],
    )
    @pytest.mark.asyncio
    async def test_no_key_dev(
        self,
        mock_request: MagicMock,
        env_vars: dict[str, str],
        expected_descriptor: str,
    ) -> None:
        """When API_KEY is not set in development, auth should return dev-mode."""
        with patch.dict(os.environ, env_vars, clear=False):
            import importlib

            from api import auth

            importlib.reload(auth)

            result = await auth.verify_api_key(mock_request, None)
            assert result == expected_descriptor

    # Scenario 2: No key configured in production — server misconfiguration
    @pytest.mark.asyncio
    async def test_no_key_prod(self, mock_request: MagicMock) -> None:
        """When API_KEY is not set in production, verify_api_key should raise 503."""
        from fastapi import HTTPException, status

        # Must explicitly clear all key env vars to prevent leaks from real env
        with patch.dict(
            os.environ,
            {
                "APP_ENV": "production",
                "PREDICTION_API_KEY": "",
                "THE_ODDS_API_KEY": "",
            },
            clear=False,
        ):
            import importlib

            from api import auth

            importlib.reload(auth)

            with pytest.raises(HTTPException) as exc_info:
                await auth.verify_api_key(mock_request, None)
            assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            # 503 = server misconfiguration, not client error
            assert "not configured" in str(exc_info.value.detail).lower()


# ═══════════════════════════════════════════════════════════
#  Scenarios 3-7: Credential handling (with valid API_KEY)
# ═══════════════════════════════════════════════════════════


class TestCredentialHandling:
    """Various credential sources when a valid API_KEY is configured."""

    TEST_API_KEY = "test-api-key-12345"

    @pytest.fixture(autouse=True)
    def _setup_auth(self):
        """Patch env vars and reload auth module for each test."""
        with patch.dict(
            os.environ,
            {
                "APP_ENV": "production",
                "PREDICTION_API_KEY": self.TEST_API_KEY,
            },
            clear=False,
        ):
            import importlib

            from api import auth

            importlib.reload(auth)
            yield

    # Scenario 3: Missing Authorization header
    @pytest.mark.asyncio
    async def test_missing_header(self, mock_request: MagicMock) -> None:
        """No Authorization header and no other source should raise 401."""
        from fastapi import HTTPException

        from api.auth import verify_api_key

        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(mock_request, None)
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    # Scenario 4: Invalid Bearer token
    @pytest.mark.asyncio
    async def test_invalid_bearer(
        self, mock_request: MagicMock, mock_bearer_credentials: MagicMock
    ) -> None:
        """Wrong bearer token should raise 403."""
        mock_bearer_credentials.credentials = "wrong-key"
        from fastapi import HTTPException

        from api.auth import verify_api_key

        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(mock_request, mock_bearer_credentials)
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN

    # Scenario 5: Valid Bearer token
    @pytest.mark.asyncio
    async def test_valid_bearer(
        self, mock_request: MagicMock, mock_bearer_credentials: MagicMock
    ) -> None:
        """Valid bearer token should return the API key."""
        mock_bearer_credentials.credentials = self.TEST_API_KEY
        from api.auth import verify_api_key

        result = await verify_api_key(mock_request, mock_bearer_credentials)
        assert result == self.TEST_API_KEY

    # Scenario 6: Valid X-API-Key header
    @pytest.mark.asyncio
    async def test_valid_x_api_key_header(self, mock_request: MagicMock) -> None:
        """X-API-Key header should authenticate when no Bearer token."""
        mock_request.headers = {"x-api-key": self.TEST_API_KEY}
        from api.auth import verify_api_key

        result = await verify_api_key(mock_request, None)
        assert result == self.TEST_API_KEY

    # Scenario 7: Query parameter (backward compatibility)
    @pytest.mark.asyncio
    async def test_query_param(self, mock_request: MagicMock) -> None:
        """Query parameter ?api_key= should authenticate (backward compat)."""
        mock_request.query_params = {"api_key": self.TEST_API_KEY}
        from api.auth import verify_api_key

        result = await verify_api_key(mock_request, None)
        assert result == self.TEST_API_KEY

    # Invalid X-API-Key
    @pytest.mark.asyncio
    async def test_invalid_x_api_key_header(self, mock_request: MagicMock) -> None:
        """Wrong X-API-Key should raise 403."""
        mock_request.headers = {"x-api-key": "wrong-key-for-sure"}
        from fastapi import HTTPException

        from api.auth import verify_api_key

        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(mock_request, None)
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN

    # Empty key after trimming
    @pytest.mark.asyncio
    async def test_empty_key_string(self, mock_request: MagicMock) -> None:
        """Whitespace-only key string should raise 403 (treated as invalid)."""
        mock_request.query_params = {"api_key": " "}
        from fastapi import HTTPException

        from api.auth import verify_api_key

        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(mock_request, None)
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN


# ═══════════════════════════════════════════════════════════
#  Scenarios 8-10: Endpoint protection (via TestClient)
# ═══════════════════════════════════════════════════════════


class TestEndpointProtection:
    """End-to-end endpoint protection tests with TestClient."""

    TEST_API_KEY = "test-api-key-12345"

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Patch env vars, create minimal test app with the real auth.

        The ``patch.dict`` context **must** stay active via ``yield`` so
        that ``verify_api_key`` still sees the patched env when the
        TestClient processes a request.
        """
        with patch.dict(
            os.environ,
            {
                "APP_ENV": "production",
                "PREDICTION_API_KEY": self.TEST_API_KEY,
            },
            clear=False,
        ):
            import importlib

            from api import auth

            importlib.reload(auth)
            from api.auth import verify_api_key

            # Build a minimal test app with the real auth dependencies
            self.app = FastAPI()

            # Public health endpoint (no auth)
            @self.app.get("/health")
            async def health():
                return {"status": "healthy"}

            # Protected /models endpoint
            @self.app.get("/models")
            async def list_models(auth: str = Depends(verify_api_key)):
                return {"models": [], "total": 0}

            # Protected /predict endpoint
            @self.app.post("/predict")
            async def predict(auth: str = Depends(verify_api_key)):
                return {"status": "success", "predictions": []}

            self.client = TestClient(self.app)
            yield

    # Scenario 8: Protected /predict — requires auth
    def test_predict_no_auth(self) -> None:
        """POST /predict without API key should return 401."""
        response = self.client.post("/predict", json={"fixtures": []})
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_predict_invalid_auth(self) -> None:
        """POST /predict with wrong API key should return 403."""
        response = self.client.post(
            "/predict",
            json={"fixtures": []},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_predict_valid_auth_bearer(self) -> None:
        """POST /predict with valid Bearer token should succeed."""
        response = self.client.post(
            "/predict",
            json={"fixtures": []},
            headers={"Authorization": f"Bearer {self.TEST_API_KEY}"},
        )
        assert response.status_code == 200

    def test_predict_valid_auth_x_api_key(self) -> None:
        """POST /predict with valid X-API-Key header should succeed."""
        response = self.client.post(
            "/predict",
            json={"fixtures": []},
            headers={"X-API-Key": self.TEST_API_KEY},
        )
        assert response.status_code == 200

    # Scenario 9: Protected /models — requires auth
    def test_models_no_auth(self) -> None:
        """GET /models without API key should return 401."""
        response = self.client.get("/models")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_models_invalid_auth(self) -> None:
        """GET /models with wrong API key should return 403."""
        response = self.client.get(
            "/models",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_models_valid_auth(self) -> None:
        """GET /models with valid API key should succeed."""
        response = self.client.get(
            "/models",
            headers={"Authorization": f"Bearer {self.TEST_API_KEY}"},
        )
        assert response.status_code == 200

    # Scenario 10: Public /health — no auth required
    def test_health_no_auth(self) -> None:
        """GET /health should work without any authentication."""
        response = self.client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_health_with_auth(self) -> None:
        """GET /health with API key should also work."""
        response = self.client.get(
            "/health",
            headers={"Authorization": f"Bearer {self.TEST_API_KEY}"},
        )
        assert response.status_code == 200


# ═══════════════════════════════════════════════════════════
#  optional_auth tests
# ═══════════════════════════════════════════════════════════


class TestOptionalAuth:
    """optional_auth should return None instead of raising."""

    TEST_API_KEY = "test-api-key-12345"

    @pytest.fixture(autouse=True)
    def _setup_auth(self):
        with patch.dict(
            os.environ,
            {
                "APP_ENV": "production",
                "PREDICTION_API_KEY": self.TEST_API_KEY,
            },
            clear=False,
        ):
            import importlib

            from api import auth

            importlib.reload(auth)
            yield

    @pytest.mark.asyncio
    async def test_optional_auth_missing(self, mock_request: MagicMock) -> None:
        """optional_auth returns None when no credentials."""
        from api.auth import optional_auth

        result = await optional_auth(mock_request, None)
        assert result is None

    @pytest.mark.asyncio
    async def test_optional_auth_invalid(
        self, mock_request: MagicMock, mock_bearer_credentials: MagicMock
    ) -> None:
        """optional_auth returns None for invalid bearer token."""
        mock_bearer_credentials.credentials = "wrong-key"
        from api.auth import optional_auth

        result = await optional_auth(mock_request, mock_bearer_credentials)
        assert result is None

    @pytest.mark.asyncio
    async def test_optional_auth_valid(
        self, mock_request: MagicMock, mock_bearer_credentials: MagicMock
    ) -> None:
        """optional_auth returns the key for valid bearer token."""
        mock_bearer_credentials.credentials = self.TEST_API_KEY
        from api.auth import optional_auth

        result = await optional_auth(mock_request, mock_bearer_credentials)
        assert result == self.TEST_API_KEY


# ═══════════════════════════════════════════════════════════
#  Model Metadata Tests
# ═══════════════════════════════════════════════════════════


def _make_dummy_model(
    n_features: int = 10, classes: tuple[int, ...] = (0, 1, 2)
) -> DummyClassifier:
    """Create a fitted DummyClassifier for testing."""
    model = DummyClassifier(strategy="stratified", random_state=42)
    X_dummy = np.zeros((10, n_features))
    y_dummy = np.random.default_rng(42).integers(0, 3, size=10)
    model.fit(X_dummy, y_dummy)
    return model


# ── Helper classes for corrupt/incompatible model tests ─────
# These MUST be defined at module level so joblib/pickle can serialise them


class _PartialPredictModel:
    """Model stub with predict() but no predict_proba()."""

    def predict(self, X):  # type: ignore[no-untyped-def]
        return np.array([0, 1, 2])


class _BadPredictModel:
    """Model stub where predict is a non-callable string."""

    predict = "string not callable"  # type: ignore[assignment]

    def predict_proba(self, X):  # type: ignore[no-untyped-def]
        return np.array([[0.3, 0.3, 0.4]])


class _BadProbaModel:
    """Model stub where predict_proba is a non-callable string."""

    def predict(self, X):  # type: ignore[no-untyped-def]
        return np.array([0])

    predict_proba = "string not callable"  # type: ignore[assignment]


class TestModelMetadata:
    """Tests for model metadata capture in _try_load."""

    def test_try_load_captures_model_type(self) -> None:
        """Loaded model should have model_type set to class name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_dummy_model()
            path = Path(tmpdir) / "test_model.joblib"
            joblib.dump(model, path)

            from api.main import _try_load, state

            state.model = None
            state.model_name = "none"
            state.model_type = "none"
            _try_load(path)

            assert state.model is not None
            assert state.model_name == "test_model.joblib"
            assert state.model_type == "DummyClassifier"
            assert state.model_feature_count == 10

    def test_try_load_rejects_invalid(self) -> None:
        """Model without predict/predict_proba should be skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid = {"not": "a model"}
            path = Path(tmpdir) / "invalid.joblib"
            joblib.dump(invalid, path)

            from api.main import _try_load, state

            state.model = None
            _try_load(path)

            assert state.model is None

    def test_try_load_rejects_corrupt(self) -> None:
        """Corrupt .joblib file should not crash _try_load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "corrupt.joblib"
            path.write_text("not valid joblib data")

            from api.main import _try_load, state

            state.model = None
            _try_load(path)

            assert state.model is None

    def test_try_load_rejects_missing_predict_proba(self) -> None:
        """Model with predict() but no predict_proba() should be skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _PartialPredictModel()
            path = Path(tmpdir) / "partial_model.joblib"
            joblib.dump(model, path)

            from api.main import _try_load, state

            state.model = None
            _try_load(path)

            assert state.model is None

    def test_try_load_rejects_non_callable_predict(self) -> None:
        """Model where predict attribute exists but is not callable should be skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _BadPredictModel()
            path = Path(tmpdir) / "bad_predict_model.joblib"
            joblib.dump(model, path)

            from api.main import _try_load, state

            state.model = None
            _try_load(path)

            assert state.model is None

    def test_try_load_rejects_non_callable_predict_proba(self) -> None:
        """Model where predict_proba attribute is not callable should be skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _BadProbaModel()
            path = Path(tmpdir) / "bad_proba_model.joblib"
            joblib.dump(model, path)

            from api.main import _try_load, state

            state.model = None
            _try_load(path)

            assert state.model is None

    def test_try_load_accepts_valid_model(self) -> None:
        """Valid model with both predict() and predict_proba() callable should load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_dummy_model()
            path = Path(tmpdir) / "valid_model.joblib"
            joblib.dump(model, path)

            from api.main import _try_load, state

            state.model = None
            _try_load(path)

            assert state.model is not None
            assert state.model_name == "valid_model.joblib"

    def test_model_path_stored(self) -> None:
        """Absolute model path should be stored in state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_dummy_model()
            path = Path(tmpdir) / "my_model.joblib"
            joblib.dump(model, path)

            from api.main import _try_load, state

            state.model = None
            state.model_path = ""
            _try_load(path)

            assert state.model_path == str(path.absolute())
            assert "my_model.joblib" in state.model_path

    def test_model_trained_at_populated(self) -> None:
        """model_trained_at should be set to file mtime."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_dummy_model()
            path = Path(tmpdir) / "timestamped.joblib"
            joblib.dump(model, path)

            from api.main import _try_load, state

            state.model = None
            state.model_trained_at = None
            _try_load(path)

            assert state.model_trained_at is not None
            assert "T" in state.model_trained_at  # ISO format includes T


class TestFeatureAlignment:
    """Tests for _validate_feature_alignment."""

    def test_alignment_passes_when_count_matches(self) -> None:
        """No error when feature count matches model expectations."""
        from fastapi import HTTPException

        from api.main import _validate_feature_alignment, state

        state.model = _make_dummy_model(n_features=5)
        state.model_feature_count = 5
        state.model_feature_names = []

        feature_row = pd.DataFrame(
            np.random.rand(1, 5), columns=["a", "b", "c", "d", "e"]
        )
        try:
            _validate_feature_alignment(feature_row)
        except HTTPException:
            pytest.fail("Should not raise HTTPException on matching count")

    def test_alignment_raises_on_count_mismatch(self) -> None:
        """HTTPException(503) on feature count mismatch."""
        from fastapi import HTTPException, status

        from api.main import _validate_feature_alignment, state

        state.model = _make_dummy_model(n_features=5)
        state.model_feature_count = 5
        state.model_feature_names = []
        state.model_name = "test_model.joblib"

        feature_row = pd.DataFrame(
            np.random.rand(1, 10), columns=[f"c{i}" for i in range(10)]
        )

        with pytest.raises(HTTPException) as exc_info:
            _validate_feature_alignment(feature_row)
        assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "5" in str(exc_info.value.detail)
        assert "10" in str(exc_info.value.detail)

    def test_alignment_raises_on_missing_columns(self) -> None:
        """HTTPException(503) when expected columns are missing (count matches)."""
        from fastapi import HTTPException, status

        from api.main import _validate_feature_alignment, state

        state.model = _make_dummy_model(n_features=5)
        state.model_feature_count = 5
        state.model_feature_names = ["a", "b", "c", "d", "e"]
        state.model_name = "test.joblib"

        # 5 columns (matching count) but wrong names: 'a' and 'b' match, 'x','y','z' don't
        feature_row = pd.DataFrame(
            np.random.rand(1, 5), columns=["a", "b", "x", "y", "z"]
        )

        with pytest.raises(HTTPException) as exc_info:
            _validate_feature_alignment(feature_row)
        assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "missing" in str(exc_info.value.detail).lower()

    def test_alignment_skipped_when_no_model(self) -> None:
        """No error when model is None (app starting up)."""
        from api.main import _validate_feature_alignment, state

        state.model = None
        feature_row = pd.DataFrame()
        try:
            _validate_feature_alignment(feature_row)
        except Exception:
            pytest.fail("Should not raise when model is None")

    def test_alignment_skipped_when_feature_count_zero(self) -> None:
        """No error when model_feature_count is 0 (unknown)."""
        from api.main import _validate_feature_alignment, state

        state.model = _make_dummy_model(n_features=0)
        state.model_feature_count = 0
        state.model_feature_names = []

        feature_row = pd.DataFrame(np.random.rand(1, 5))
        try:
            _validate_feature_alignment(feature_row)
        except Exception:
            pytest.fail("Should not raise when feature count is 0")


class TestHealthEndpointMeta:
    """Tests that health endpoint exposes model metadata."""

    TEST_API_KEY = "test-api-key-12345"

    @pytest.fixture(autouse=True)
    def _setup(self):
        with patch.dict(
            os.environ,
            {
                "APP_ENV": "production",
                "PREDICTION_API_KEY": self.TEST_API_KEY,
            },
            clear=False,
        ):
            import importlib

            from api import auth

            importlib.reload(auth)

            from api.main import app, state

            # Reset state for clean test
            state.model = None
            state.model_name = "none"
            state.model_type = "none"
            state.model_path = ""
            state.model_trained_at = None
            state.model_feature_count = 0

            from fastapi.testclient import TestClient

            self.client = TestClient(app)
            yield
            state.model = None

    def test_health_no_model(self) -> None:
        """Health shows model_loaded=False and metadata defaults when no model loaded."""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is False
        assert data["model_name"] == "none"
        assert data["model_type"] == "none"
        assert data["model_features"] == 0
        assert data["model_trained_at"] is None
        assert data["status"] == "healthy"

    def test_health_with_model(self) -> None:
        """Health shows model metadata when a model is loaded."""
        from api.main import _try_load

        model = _make_dummy_model(n_features=10)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "my_model.joblib"
            joblib.dump(model, path)
            _try_load(path)

            response = self.client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["model_loaded"] is True
            assert data["model_name"] == "my_model.joblib"
            assert data["model_type"] == "DummyClassifier"
            assert data["model_features"] == 10
            assert data["model_trained_at"] is not None


class TestModelEndpointMeta:
    """Tests that /models endpoint exposes model metadata."""

    TEST_API_KEY = "test-api-key-12345"

    @pytest.fixture(autouse=True)
    def _setup(self):
        with patch.dict(
            os.environ,
            {
                "APP_ENV": "production",
                "PREDICTION_API_KEY": self.TEST_API_KEY,
            },
            clear=False,
        ):
            import importlib

            from api import auth

            importlib.reload(auth)

            from api.main import app, state

            state.model = None
            state.model_name = "none"
            state.model_type = "none"
            state.model_path = ""
            state.model_trained_at = None
            state.model_feature_count = 0
            state.model_feature_names = []

            from fastapi.testclient import TestClient

            self.client = TestClient(app)
            yield
            state.model = None

    def test_models_no_model(self) -> None:
        """Empty model list when no model loaded."""
        response = self.client.get(
            "/models",
            headers={"Authorization": f"Bearer {self.TEST_API_KEY}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["models"] == []

    def test_models_with_model(self) -> None:
        """Model list shows metadata for loaded model."""
        from api.main import _try_load

        model = _make_dummy_model(n_features=8)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "xgboost_model.joblib"
            joblib.dump(model, path)
            _try_load(path)

            response = self.client.get(
                "/models",
                headers={"Authorization": f"Bearer {self.TEST_API_KEY}"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 1
            model_info = data["models"][0]
            assert model_info["name"] == "xgboost_model.joblib"
            assert model_info["model_type"] == "DummyClassifier"
            assert model_info["features"] == 8
            assert model_info["model_path"] != ""
            assert model_info["trained_at"] is not None
