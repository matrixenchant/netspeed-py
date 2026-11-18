"""Tests for netspeed. Run with: python -m unittest -v

No network needed: HTTP tests use a local server on 127.0.0.1.
"""

import contextlib
import http.server
import io
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import netspeed

BODY = b"x" * 200_000
HTML = b"<!doctype html><title>hi</title>"


class Handler(http.server.BaseHTTPRequestHandler):
    """Test endpoints, one per server behaviour we need to cover."""

    def do_GET(self):
        self.server.seen.append((self.path, dict(self.headers)))
        route = self.path.split("?")[0]
        if route == "/ok":
            self._send(BODY, "image/png")
        elif route == "/html":
            self._send(HTML, "text/html; charset=utf-8")
        elif route == "/no-type":
            self._send(BODY, None)
        elif route == "/no-length":
            # HTTP/1.0 response without Content-Length: the body ends when the connection closes
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            self.wfile.write(BODY)
        elif route == "/broken":
            # promise more than we send, then drop the connection mid-body
            self.send_response(200)
            self.send_header("Content-Length", str(len(BODY)))
            self.end_headers()
            self.wfile.write(BODY[:1000])
        elif route == "/slow":
            self.send_response(200)
            self.send_header("Content-Length", str(5 * 1000))
            self.end_headers()
            for _ in range(5):
                self.wfile.write(b"y" * 1000)
                self.wfile.flush()
                time.sleep(0.3)
        else:
            self.send_error(404)

    def _send(self, body, content_type):
        self.send_response(200)
        if content_type:
            self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep test output clean
        pass


class QuietServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        pass  # the client hanging up on /slow and /broken is expected


class LocalServerMixin:
    @classmethod
    def setUpClass(cls):
        cls.server = QuietServer(("127.0.0.1", 0), Handler)
        cls.server.seen = []
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.server.seen.clear()


class FakeStdout(io.StringIO):
    """In-memory stdout that can pretend to be a terminal with a given encoding."""

    def __init__(self, tty=False, encoding="utf-8"):
        super().__init__()
        self._tty = tty
        self._encoding = encoding

    @property
    def encoding(self):
        return self._encoding

    def isatty(self):
        return self._tty


def run_main(argv, stdout=None):
    """Run netspeed.main, return (exit code, stdout text, stderr text)."""
    out = stdout or FakeStdout()
    err = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = netspeed.main(argv)
    return code, out.getvalue(), err.getvalue()


# --------------------------------------------------------------------------- pure functions


class MbitPerSecondTest(unittest.TestCase):
    def test_converts_bytes_to_megabits(self):
        self.assertEqual(netspeed.mbit_per_s(1_000_000, 1.0), 8.0)
        self.assertEqual(netspeed.mbit_per_s(125_000, 0.5), 2.0)

    def test_zero_or_negative_time_gives_zero(self):
        self.assertEqual(netspeed.mbit_per_s(100, 0.0), 0.0)
        self.assertEqual(netspeed.mbit_per_s(100, -1.0), 0.0)


class FormatBytesTest(unittest.TestCase):
    def test_decimal_units(self):
        self.assertEqual(netspeed.format_bytes(0), "0.00 B")
        self.assertEqual(netspeed.format_bytes(999), "999.00 B")
        self.assertEqual(netspeed.format_bytes(1_000), "1.00 KB")
        self.assertEqual(netspeed.format_bytes(10_485_760), "10.49 MB")
        self.assertEqual(netspeed.format_bytes(2_500_000_000), "2.50 GB")
        self.assertEqual(netspeed.format_bytes(3_000_000_000_000), "3.00 TB")


class CacheBusterTest(unittest.TestCase):
    def test_adds_query(self):
        self.assertEqual(netspeed.add_cache_buster("http://h/a.jpg", "1"), "http://h/a.jpg?_=1")

    def test_keeps_existing_query_and_fragment(self):
        self.assertEqual(
            netspeed.add_cache_buster("http://h/a.jpg?w=1#top", "1"), "http://h/a.jpg?w=1&_=1#top"
        )


class BarTest(unittest.TestCase):
    def test_fill_proportional(self):
        self.assertEqual(netspeed.bar(0.5, 4), netspeed.FULL * 2 + netspeed.EMPTY * 2)

    def test_clamps_fraction(self):
        self.assertEqual(netspeed.bar(1.7, 3), netspeed.FULL * 3)
        self.assertEqual(netspeed.bar(-0.2, 3), netspeed.EMPTY * 3)


class CanPrintTest(unittest.TestCase):
    def test_depends_on_stdout_encoding(self):
        with mock.patch.object(sys, "stdout", FakeStdout(encoding="utf-8")):
            self.assertTrue(netspeed._can_print("█"))
        with mock.patch.object(sys, "stdout", FakeStdout(encoding="cp1251")):
            self.assertFalse(netspeed._can_print("█"))
        with mock.patch.object(sys, "stdout", FakeStdout(encoding="no-such-codec")):
            self.assertFalse(netspeed._can_print("a"))


class DescribeContentTest(unittest.TestCase):
    def test_from_header(self):
        d = netspeed.describe_content
        self.assertEqual(d("image/jpeg", "http://h/x"), "image (image/jpeg)")
        self.assertEqual(d("video/mp4", "http://h/x"), "video (video/mp4)")
        self.assertEqual(d("text/html", "http://h/"), "HTML page (text/html)")
        self.assertEqual(d("text/plain", "http://h/"), "text (text/plain)")
        self.assertEqual(d("application/zip", "http://h/x"), "archive (application/zip)")
        self.assertEqual(
            d("application/octet-stream", "http://h/x"), "binary file (application/octet-stream)"
        )
        self.assertEqual(d("application/x-unknown", "http://h/x"), "file (application/x-unknown)")

    def test_header_wins_over_extension(self):
        self.assertEqual(
            netspeed.describe_content("text/html", "http://h/a.jpg"), "HTML page (text/html)"
        )

    def test_guess_from_url_without_header(self):
        self.assertEqual(
            netspeed.describe_content(None, "http://h/pic.png?w=1"),
            "image (image/png, guessed from URL)",
        )
        self.assertEqual(
            netspeed.describe_content(None, "http://h/data"), "unknown (no Content-Type)"
        )


# --------------------------------------------------------------------------- progress / chart


class ProgressLineTest(unittest.TestCase):
    def test_disabled_prints_only_final_line(self):
        out = FakeStdout()
        with contextlib.redirect_stdout(out):
            p = netspeed.ProgressLine("[1/1]", enabled=False)
            p.update(10, 100, 1.0)
            p.finish("[1/1] done")
        self.assertEqual(out.getvalue(), "[1/1] done\n")

    def test_with_total_shows_percentage(self):
        out = FakeStdout()
        with contextlib.redirect_stdout(out):
            netspeed.ProgressLine("[1/1]", enabled=True).update(500_000, 1_000_000, 1.0)
        line = out.getvalue()
        self.assertTrue(line.startswith("\r[1/1] "))
        self.assertIn("50%", line)
        self.assertIn("0.50/1.00 MB", line)
        self.assertIn("4.00 Mbit/s", line)

    def test_without_total_shows_bytes_and_speed(self):
        out = FakeStdout()
        with contextlib.redirect_stdout(out):
            netspeed.ProgressLine("[1/1]", enabled=True).update(2_000_000, None, 2.0)
        self.assertNotIn("%", out.getvalue())
        self.assertIn("2.00 MB", out.getvalue())
        self.assertIn("8.00 Mbit/s", out.getvalue())

    def test_redraws_are_throttled(self):
        out = FakeStdout()
        with contextlib.redirect_stdout(out):
            p = netspeed.ProgressLine("[1/1]", enabled=True)
            p.update(1, 10, 1.0)
            p.update(2, 10, 1.0)  # within REDRAW_INTERVAL -> skipped
        self.assertEqual(out.getvalue().count("\r"), 1)

    def test_finish_wipes_longer_previous_line(self):
        out = FakeStdout()
        with contextlib.redirect_stdout(out):
            p = netspeed.ProgressLine("[1/1]", enabled=True)
            p.update(1, 10, 1.0)
            long_len = p.last_len
            p.finish("short")
        last = out.getvalue().split("\r")[-1]
        self.assertTrue(last.endswith("\n"))
        self.assertEqual(len(last.rstrip("\n")), long_len)  # padded to the old length


class PrintChartTest(unittest.TestCase):
    def chart(self, speeds, overall=1.0):
        out = FakeStdout()
        with contextlib.redirect_stdout(out):
            netspeed.print_chart(speeds, overall)
        return out.getvalue().splitlines()

    def test_fastest_request_gets_full_width(self):
        lines = self.chart([10.0, 5.0], overall=7.5)
        self.assertIn("overall 7.50", lines[1])
        self.assertEqual(lines[2].count(netspeed.FULL), netspeed.CHART_WIDTH)
        self.assertEqual(lines[3].count(netspeed.FULL), netspeed.CHART_WIDTH // 2)

    def test_failed_request_is_marked(self):
        lines = self.chart([3.0, None])
        self.assertTrue(lines[-1].endswith("error"))

    def test_all_zero_speeds_do_not_divide_by_zero(self):
        lines = self.chart([0.0])
        self.assertEqual(lines[-1].count(netspeed.FULL), 0)


# --------------------------------------------------------------------------- argument parsing


class ParseArgsTest(unittest.TestCase):
    def assertArgError(self, argv):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
            netspeed.parse_args(argv)
        self.assertEqual(cm.exception.code, 2)

    def test_defaults(self):
        args = netspeed.parse_args(["https://example.com/x"])
        self.assertEqual(args.count, 10)
        self.assertEqual(args.timeout, 30.0)
        self.assertEqual(args.max_time, 120.0)
        self.assertFalse(args.bust_cache)
        self.assertFalse(args.no_progress)
        self.assertFalse(args.no_chart)

    def test_test_flag_uses_test_url(self):
        self.assertEqual(netspeed.parse_args(["--test"]).url, netspeed.TEST_URL)

    def test_max_time_zero_is_allowed(self):
        self.assertEqual(netspeed.parse_args(["http://h", "--max-time", "0"]).max_time, 0)

    def test_invalid_args(self):
        self.assertArgError([])                                       # no URL
        self.assertArgError(["--test", "https://example.com"])        # both URL and --test
        self.assertArgError(["ftp://example.com"])                    # not http(s)
        self.assertArgError(["https://example.com", "-n", "0"])       # count < 1
        self.assertArgError(["https://example.com", "-n", "abc"])     # not a number
        self.assertArgError(["https://example.com", "-t", "0"])       # timeout <= 0
        self.assertArgError(["https://example.com", "--max-time", "-1"])


# --------------------------------------------------------------------------- fetch over HTTP


class FetchTest(LocalServerMixin, unittest.TestCase):
    def test_counts_body_bytes_and_reads_type(self):
        r = netspeed.fetch(f"{self.base}/ok", timeout=5)
        self.assertEqual(r.size, len(BODY))
        self.assertEqual(r.content_type, "image/png")
        self.assertGreater(r.ttfb, 0)
        self.assertGreaterEqual(r.elapsed, r.ttfb)

    def test_content_type_params_are_stripped(self):
        self.assertEqual(netspeed.fetch(f"{self.base}/html", timeout=5).content_type, "text/html")

    def test_missing_content_type_is_none(self):
        self.assertIsNone(netspeed.fetch(f"{self.base}/no-type", timeout=5).content_type)

    def test_body_without_content_length(self):
        r = netspeed.fetch(f"{self.base}/no-length", timeout=5)
        self.assertEqual(r.size, len(BODY))

    def test_sends_no_cache_headers_and_user_agent(self):
        netspeed.fetch(f"{self.base}/ok", timeout=5)
        _, headers = self.server.seen[-1]
        self.assertEqual(headers["User-Agent"], netspeed.USER_AGENT)
        self.assertEqual(headers["Cache-Control"], "no-cache")
        self.assertEqual(headers["Pragma"], "no-cache")

    def test_reports_progress(self):
        progress = mock.Mock()
        netspeed.fetch(f"{self.base}/ok", timeout=5, progress=progress)
        received, total, _ = progress.update.call_args.args
        self.assertEqual((received, total), (len(BODY), len(BODY)))

    def test_truncated_body_is_an_error(self):
        with self.assertRaises(netspeed.REQUEST_ERRORS) as cm:
            netspeed.fetch(f"{self.base}/broken", timeout=5)
        self.assertIn("connection closed after", str(cm.exception))

    def test_http_error_status(self):
        with self.assertRaises(netspeed.REQUEST_ERRORS):
            netspeed.fetch(f"{self.base}/missing", timeout=5)

    def test_max_time_deadline(self):
        with self.assertRaises(TimeoutError):
            netspeed.fetch(f"{self.base}/slow", timeout=5, max_time=0.2)

    def test_invalid_port_is_a_request_error(self):
        with self.assertRaises(netspeed.REQUEST_ERRORS):
            netspeed.fetch("http://127.0.0.1:abc/", timeout=5)


# --------------------------------------------------------------------------- whole run


class RunTest(LocalServerMixin, unittest.TestCase):
    def test_all_ok(self):
        code, out, _ = run_main([f"{self.base}/ok", "-n", "2"])
        self.assertEqual(code, netspeed.EXIT_OK)
        self.assertEqual(len(self.server.seen), 2)  # exactly N requests
        self.assertIn("Content type:        image (image/png)", out)
        self.assertIn("Successful requests: 2/2", out)
        self.assertIn(f"Total downloaded:    {netspeed.format_bytes(2 * len(BODY))}", out)
        self.assertIn("median", out)
        self.assertIn("Speed per request, Mbit/s (overall", out)  # chart

    def test_no_chart(self):
        _, out, _ = run_main([f"{self.base}/ok", "-n", "1", "--no-chart"])
        self.assertNotIn("overall", out)

    def test_all_failed(self):
        code, out, err = run_main([f"{self.base}/missing", "-n", "2"])
        self.assertEqual(code, netspeed.EXIT_ALL_FAILED)
        self.assertIn("error: HTTPError", out)
        self.assertIn("All requests failed.", err)

    def test_broken_body_does_not_crash(self):
        code, out, _ = run_main([f"{self.base}/broken", "-n", "2", "--no-chart"])
        self.assertEqual(code, netspeed.EXIT_ALL_FAILED)
        self.assertIn("connection closed after", out)

    def test_partial_failure(self):
        ok = netspeed.Result(1000, 0.1, 0.01, "image/png")
        with mock.patch.object(netspeed, "fetch", side_effect=[ok, OSError("boom"), ok]):
            code, out, _ = run_main(["https://example.com", "-n", "3"])
        self.assertEqual(code, netspeed.EXIT_PARTIAL)
        self.assertIn("Successful requests: 2/3", out)
        self.assertIn("error: OSError: boom", out)

    def test_bust_cache_makes_urls_unique(self):
        run_main([f"{self.base}/ok", "-n", "2", "--bust-cache", "--no-chart"])
        paths = [path for path, _ in self.server.seen]
        self.assertTrue(all("?_=" in p for p in paths))
        self.assertEqual(len(set(paths)), 2)

    def test_html_note(self):
        _, out, _ = run_main([f"{self.base}/html", "-n", "1", "--no-chart"])
        self.assertIn("HTML page (text/html)", out)
        self.assertIn("Note: this is an HTML page", out)

    def test_live_progress_only_in_terminal(self):
        _, out, _ = run_main([f"{self.base}/ok", "-n", "1"], stdout=FakeStdout(tty=True))
        self.assertIn("\r", out)
        _, out, _ = run_main(
            [f"{self.base}/ok", "-n", "1", "--no-progress"], stdout=FakeStdout(tty=True)
        )
        self.assertNotIn("\r", out)
        _, out, _ = run_main([f"{self.base}/ok", "-n", "1"], stdout=FakeStdout(tty=False))
        self.assertNotIn("\r", out)

    def test_ctrl_c(self):
        with mock.patch.object(netspeed, "fetch", side_effect=KeyboardInterrupt):
            code, _, err = run_main(["https://example.com", "-n", "1"])
        self.assertEqual(code, netspeed.EXIT_INTERRUPTED)
        self.assertIn("Interrupted.", err)


class CommandLineTest(unittest.TestCase):
    def test_script_entry_point_exit_code(self):
        script = Path(__file__).with_name("netspeed.py")
        proc = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, timeout=30
        )
        self.assertEqual(proc.returncode, 2)  # argparse error: no URL
        self.assertIn("URL is required", proc.stderr)


if __name__ == "__main__":
    unittest.main()
