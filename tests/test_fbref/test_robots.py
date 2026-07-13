"""
Unit tests for RobotsChecker.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.data_collection.sources.fbref.robots import (
    DEFAULT_DELAY,
    RobotsChecker,
    RobotsPolicy,
)


class TestRobotsPolicy:
    def test_allows_main_paths(self) -> None:
        policy = RobotsPolicy()
        assert policy.is_path_allowed("/en/comps/9/Premier-League-Stats") is True
        assert policy.is_path_allowed("/en/squads/b8fd03ef/2024-2025/Man-City") is True

    def test_disallows_specific_paths(self) -> None:
        policy = RobotsPolicy(
            disallowed_prefixes=["/my/", "/feedback/", "/fbref/"],
        )
        assert policy.is_path_allowed("/my/account") is False
        assert policy.is_path_allowed("/feedback/submit") is False
        assert policy.is_path_allowed("/en/comps/9/") is True

    def test_disallowed_overall(self) -> None:
        policy = RobotsPolicy(is_allowed=False)
        assert policy.is_path_allowed("/any/path") is False

    def test_stale_policy(self) -> None:
        import time
        policy = RobotsPolicy(fetched_at=time.time() - 4000, ttl=3600)
        assert policy.is_stale is True
        policy.fetched_at = time.time()
        assert policy.is_stale is False


class TestRobotsChecker:
    def test_default_delay(self) -> None:
        checker = RobotsChecker()
        policy = checker.get_policy()
        assert policy.crawl_delay >= DEFAULT_DELAY

    def test_known_disallowed_paths(self) -> None:
        """Even without fetching robots.txt, known paths are disallowed."""
        checker = RobotsChecker()
        # Known FBref disallowed paths
        assert checker.check_path(path="/feedback/") is False
        assert checker.check_path(path="/my/account") is False
        assert checker.check_path(path="/en/comps/9/") is True

    def test_rate_limiting(self) -> None:
        """Rate limiter blocks for the configured delay."""
        checker = RobotsChecker(default_delay=0.05)
        waited = checker.wait_if_needed()
        assert waited == 0.0  # No previous request

        checker.record_request()
        waited = checker.wait_if_needed()
        assert waited > 0.0  # Should have waited some time

    def test_parse_robots_txt_standard(self) -> None:
        """Parse a well-formed robots.txt."""
        robots_text = """\
User-agent: *
Disallow: /fbref/
Disallow: /feedback/
Disallow: /my/
Crawl-delay: 5

User-agent: Googlebot
Disallow: /private/
"""
        policy = RobotsChecker._parse_robots_txt(robots_text)
        assert "/fbref/" in policy.disallowed_prefixes
        assert "/feedback/" in policy.disallowed_prefixes
        assert policy.crawl_delay == 5.0

    def test_parse_robots_txt_no_rules(self) -> None:
        """Empty robots.txt should use FBref known defaults."""
        robots_text = "User-agent: *\nAllow: /\n"
        policy = RobotsChecker._parse_robots_txt(robots_text)
        # Should have the known FBref paths as fallback
        assert len(policy.disallowed_prefixes) > 0
        assert "/fbref/" in policy.disallowed_prefixes

    def test_close(self) -> None:
        checker = RobotsChecker()
        checker.close()
        # No error on double close
        checker.close()
