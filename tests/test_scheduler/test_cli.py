"""
Tests for the scheduler CLI — argument parsing and command handlers.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.scheduler.cli import (
    build_parser,
    cmd_list,
    cmd_run,
    cmd_status,
    format_report,
    format_report_dict,
    main,
)


class TestArgumentParser:
    def test_parser_builds(self) -> None:
        parser = build_parser()
        assert parser.prog == "python -m src.scheduler.cli"

    def test_run_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run"])
        assert args.command == "run"
        assert args.tasks == ""
        assert args.quiet is False
        assert args.no_report is False
        assert args.abort is True

    def test_run_with_tasks(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--tasks", "download,backup"])
        assert args.tasks == "download,backup"

    def test_run_quiet(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--quiet"])
        assert args.quiet is True

    def test_install_windows(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["install", "--platform", "windows"])
        assert args.platform == "windows"

    def test_install_cron(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["install", "--platform", "cron"])
        assert args.platform == "cron"

    def test_install_dry_run(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["install", "--dry-run"])
        assert args.dry_run is True

    def test_install_custom_schedule(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["install", "--schedule", "hourly", "--time", "00:00"])
        assert args.schedule == "hourly"
        assert args.time == "00:00"

    def test_generate_windows(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["generate", "--platform", "windows"])
        assert args.platform == "windows"

    def test_generate_with_output(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["generate", "--platform", "cron", "--output", "mycron.txt"])
        assert args.output == "mycron.txt"

    def test_status_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["status"])
        assert args.command == "status"

    def test_list_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["list"])
        assert args.command == "list"

    def test_no_command_shows_help(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None


class TestCmdRun:
    def test_run_success(self) -> None:
        with patch("src.scheduler.cli.TaskEngine") as MockEngine:
            mock_engine = MagicMock()
            mock_report = MagicMock()
            mock_report.success = True
            mock_report.to_dict.return_value = {"succeeded": 3, "failed": 0, "skipped": 0}
            mock_engine.run_all.return_value = mock_report
            MockEngine.return_value = mock_engine

            parser = build_parser()
            args = parser.parse_args(["run", "--no-report", "--quiet"])
            exit_code = cmd_run(args)
            assert exit_code == 0

    def test_run_failure(self) -> None:
        with patch("src.scheduler.cli.TaskEngine") as MockEngine:
            mock_engine = MagicMock()
            mock_report = MagicMock()
            mock_report.success = False
            mock_report.to_dict.return_value = {"succeeded": 0, "failed": 2, "skipped": 0}
            mock_engine.run_all.return_value = mock_report
            MockEngine.return_value = mock_engine

            parser = build_parser()
            args = parser.parse_args(["run", "--no-report", "--quiet"])
            exit_code = cmd_run(args)
            assert exit_code == 1

    def test_run_with_specific_tasks(self) -> None:
        with patch("src.scheduler.cli.TaskEngine") as MockEngine:
            mock_engine = MagicMock()
            mock_report = MagicMock()
            mock_report.success = True
            mock_report.to_dict.return_value = {"succeeded": 1, "failed": 0, "skipped": 0}
            mock_engine.run_all.return_value = mock_report
            MockEngine.return_value = mock_engine

            parser = build_parser()
            args = parser.parse_args(["run", "--tasks", "download_fixtures,validate_data", "--no-report", "--quiet"])
            exit_code = cmd_run(args)
            assert exit_code == 0

    def test_run_with_abort(self) -> None:
        with patch("src.scheduler.cli.TaskEngine") as MockEngine:
            mock_engine = MagicMock()
            mock_report = MagicMock()
            mock_report.success = True
            mock_report.to_dict.return_value = {}
            mock_engine.run_all.return_value = mock_report
            MockEngine.return_value = mock_engine

            parser = build_parser()
            args = parser.parse_args(["run", "--no-report", "--quiet", "--abort"])
            exit_code = cmd_run(args)
            assert exit_code == 0


class TestCmdList:
    def test_list_returns_zero(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["list"])
        exit_code = cmd_list(args)
        assert exit_code == 0

    def test_list_prints_tasks(self, capsys) -> None:
        parser = build_parser()
        args = parser.parse_args(["list"])
        cmd_list(args)
        captured = capsys.readouterr()
        assert "download_fixtures" in captured.out
        assert "validate_data" in captured.out
        assert "Pipeline:" in captured.out


class TestCmdStatus:
    def test_status_no_reports(self) -> None:
        with patch("pathlib.Path.exists", return_value=False):
            parser = build_parser()
            args = parser.parse_args(["status"])
            exit_code = cmd_status(args)
            assert exit_code == 0

    def test_status_with_report(self, capsys) -> None:
        import json
        import tempfile

        # Create a mock report directory with a report file
        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = Path(tmpdir)
            report_file = report_dir / "run_20240101_120000.json"
            report_file.write_text(json.dumps({
                "pipeline_name": "test",
                "started_at": "2024-01-01T12:00:00",
                "duration_seconds": 10.5,
                "succeeded": 3,
                "failed": 0,
                "skipped": 0,
            }))

            with patch("src.scheduler.cli.Path") as MockPath:
                mock_path = MagicMock()
                mock_path.exists.return_value = True
                mock_path.glob.return_value = [report_file]
                MockPath.return_value = mock_path

                with patch("src.scheduler.cli.Path.__init__", return_value=None):
                    with patch("src.scheduler.cli.ScheduleConfig") as MockCfg:
                        mock_cfg = MagicMock()
                        mock_cfg.report_dir = tmpdir
                        MockCfg.default.return_value = mock_cfg

                        with patch("src.scheduler.cli.sys") as mock_sys:
                            mock_sys.platform = "linux"  # Avoid Windows task check
                            parser = build_parser()
                            args = parser.parse_args(["status"])
                            exit_code = cmd_status(args)
                            assert exit_code == 0


class TestFormatReport:
    def test_format_report_empty(self) -> None:
        text = format_report_dict({})
        assert "Pipeline:" in text
        assert "Started:" in text

    def test_format_report_with_errors(self) -> None:
        text = format_report_dict({
            "pipeline_name": "test",
            "started_at": "now",
            "duration_seconds": 5.0,
            "succeeded": 2,
            "failed": 1,
            "skipped": 0,
            "errors": ["Task A failed", "Task B warning"],
        })
        assert "2 succeeded" in text
        assert "1 failed" in text
        assert "Task A failed" in text


class TestMain:
    def test_main_no_args(self) -> None:
        exit_code = main([])
        assert exit_code == 1

    def test_main_unknown_command(self) -> None:
        exit_code = main(["unknown_command"])
        assert exit_code == 1

    def test_main_list(self) -> None:
        exit_code = main(["list"])
        assert exit_code == 0

    def test_main_run(self) -> None:
        with patch("src.scheduler.cli.TaskEngine") as MockEngine:
            mock_engine = MagicMock()
            mock_report = MagicMock()
            mock_report.success = True
            mock_report.to_dict.return_value = {}
            mock_engine.run_all.return_value = mock_report
            MockEngine.return_value = mock_engine

            exit_code = main(["run", "--no-report", "--quiet"])
            assert exit_code == 0

    def test_main_keyboard_interrupt(self) -> None:
        with patch("src.scheduler.cli.cmd_run", side_effect=KeyboardInterrupt):
            exit_code = main(["run", "--no-report", "--quiet"])
            assert exit_code == 1

    def test_main_exception(self) -> None:
        with patch("src.scheduler.cli.cmd_run", side_effect=ValueError("Boom")):
            exit_code = main(["run", "--no-report", "--quiet"])
            assert exit_code == 1
