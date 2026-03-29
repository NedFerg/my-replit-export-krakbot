"""
Unit tests for CSV merge-conflict recovery in the broker trade log.

These tests verify that _recover_csv_conflict_markers() correctly detects
git merge conflict markers in CSV log files and rotates the corrupted file
to a timestamped backup (``<path>.bak_YYYYMMDD_HHMMSS``) so a fresh log
can be started — without losing the ability to record new trades and
without overwriting a previous backup from an earlier incident.

Background
----------
If a ``git merge`` or ``git pull`` runs while the bot is offline and
paper_trades.csv (or live_trades.csv) is tracked by git, conflict markers
like::

    <<<<<<< HEAD
    2026-03-21T17:10:07Z,SOL,buy,...
    =======
    2026-03-21T17:10:07Z,ETH,buy,...
    >>>>>>> 6b4141565ff99c0d7ac34311105de14b02f71230

can be injected into the CSV.  The bot must detect these at startup,
rotate the bad file to a timestamped backup, and continue normally.
"""
import glob
import os
import sys
import tempfile
import unittest

# Allow importing from project/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))

from broker.broker import _recover_csv_conflict_markers


CLEAN_CSV = (
    "timestamp,asset,side,size_coins,fill_price,notional_usd,"
    "fee_usd,realized_pnl_usd,position_after_trade\n"
    "2026-03-21T17:10:07Z,SOL,buy,0.22353862,89.514735,20.01,"
    "0.08004,0.0,0.22353862\n"
)

CONFLICTED_CSV = (
    "timestamp,asset,side,size_coins,fill_price,notional_usd,"
    "fee_usd,realized_pnl_usd,position_after_trade\n"
    "<<<<<<< HEAD\n"
    "2026-03-21T17:10:07Z,SOL,buy,0.22353862,89.514735,20.01,"
    "0.08004,0.0,0.22353862\n"
    "=======\n"
    "2026-03-21T17:10:07Z,ETH,buy,0.00933032,2144.621775,20.01,"
    "0.08004,0.0,0.00933032\n"
    ">>>>>>> 6b4141565ff99c0d7ac34311105de14b02f71230\n"
)


class TestRecoverCsvConflictMarkers(unittest.TestCase):
    """Tests for _recover_csv_conflict_markers()."""

    def _write_file(self, tmp_dir: str, filename: str, content: str) -> str:
        path = os.path.join(tmp_dir, filename)
        with open(path, "w") as fh:
            fh.write(content)
        return path

    # ------------------------------------------------------------------
    # Non-conflicted file — must not be touched
    # ------------------------------------------------------------------

    def test_clean_file_returns_false(self):
        """A clean CSV returns a falsy value and leaves the file unchanged."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_file(tmp, "paper_trades.csv", CLEAN_CSV)
            result = _recover_csv_conflict_markers(path)
            self.assertFalse(result)
            self.assertTrue(os.path.exists(path), "clean file must not be renamed")
            bak_files = glob.glob(path + ".bak_*")
            self.assertEqual(len(bak_files), 0, "no backup files must be created for clean file")

    # ------------------------------------------------------------------
    # Non-existent file — must not crash
    # ------------------------------------------------------------------

    def test_missing_file_returns_false(self):
        """A missing file returns False without raising."""
        result = _recover_csv_conflict_markers("/nonexistent/path/paper_trades.csv")
        self.assertFalse(result)

    # ------------------------------------------------------------------
    # Conflicted file — must be rotated to .bak
    # ------------------------------------------------------------------

    def test_conflicted_file_returns_true(self):
        """A file with conflict markers returns a truthy backup path."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_file(tmp, "paper_trades.csv", CONFLICTED_CSV)
            result = _recover_csv_conflict_markers(path)
            self.assertTrue(result)

    def test_conflicted_file_is_rotated_to_timestamped_bak(self):
        """The conflicted file is renamed to a timestamped backup."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_file(tmp, "paper_trades.csv", CONFLICTED_CSV)
            bak_path = _recover_csv_conflict_markers(path)
            self.assertIsNotNone(bak_path, "backup path must be returned")
            self.assertTrue(os.path.exists(bak_path), "timestamped .bak file must be created")
            self.assertFalse(os.path.exists(path), "original file must be removed")
            # Backup name must contain a timestamp suffix (.bak_YYYYMMDD_HHMMSS)
            self.assertRegex(
                os.path.basename(bak_path),
                r"\.bak_\d{8}_\d{6}$",
                "backup filename must end with .bak_YYYYMMDD_HHMMSS",
            )

    def test_bak_preserves_original_content(self):
        """The backup file contains the exact original (conflicted) content."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_file(tmp, "paper_trades.csv", CONFLICTED_CSV)
            bak_path = _recover_csv_conflict_markers(path)
            with open(bak_path) as fh:
                content = fh.read()
            self.assertEqual(content, CONFLICTED_CSV)

    # ------------------------------------------------------------------
    # Variant: only '=======' marker (degenerate conflict)
    # ------------------------------------------------------------------

    def test_equals_only_marker_detected(self):
        """A file containing only '=======' is still flagged."""
        content = "header\n=======\nrow\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_file(tmp, "paper_trades.csv", content)
            result = _recover_csv_conflict_markers(path)
            self.assertTrue(result)

    # ------------------------------------------------------------------
    # Variant: marker is preceded by whitespace
    # ------------------------------------------------------------------

    def test_whitespace_prefix_marker_detected(self):
        """Conflict markers with leading whitespace are still detected."""
        content = "header\n  <<<<<<< HEAD\nrow\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_file(tmp, "paper_trades.csv", content)
            result = _recover_csv_conflict_markers(path)
            self.assertTrue(result)

    # ------------------------------------------------------------------
    # Docs / future-contributor note
    # ------------------------------------------------------------------
    # To test the full PaperBroker startup recovery path end-to-end,
    # instantiate PaperBroker with a pre-corrupted LOG_PATH (override
    # the class attribute or environment variable) and verify that:
    #   1. paper_trades.csv.bak_<timestamp> exists after __init__
    #   2. paper_trades.csv is fresh (header only)
    #   3. Subsequent _paper_fill() calls write rows correctly
    # That integration test requires mocking the Kraken HTTP layer and
    # is out of scope here; the unit tests above cover the recovery helper
    # in isolation.


if __name__ == "__main__":
    unittest.main()
