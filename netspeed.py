"""netspeed - measure download speed with N sequential HTTP requests."""

import argparse
import http.client
import mimetypes
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import NamedTuple

CHUNK_SIZE = 64 * 1024  # read the body in chunks of up to 64 KiB, never the whole file in memory
USER_AGENT = "netspeed/1.0"  # some sites block urllib's default User-Agent
TEST_URL = "https://proof.ovh.net/files/10Mb.dat"  # public 10 MB test file hosted by OVH

BAR_WIDTH = 30          # live progress bar width, chars
CHART_WIDTH = 40        # width of the longest chart bar, chars
REDRAW_INTERVAL = 0.1   # redraw progress at most 10 times a second

# exit codes (2 is taken by argparse for invalid arguments)
EXIT_OK = 0
EXIT_ALL_FAILED = 1
EXIT_PARTIAL = 3
EXIT_INTERRUPTED = 130

# errors of a single request that must not abort the whole run:
# URLError (incl. HTTPError) - network / DNS / HTTP 4xx-5xx; HTTPException - truncated body,
# InvalidURL; OSError - socket errors, incl. TimeoutError; ValueError - malformed URL
REQUEST_ERRORS = (urllib.error.URLError, http.client.HTTPException, OSError, ValueError)


def _can_print(chars: str) -> bool:
    """Check that stdout can encode the chars (a Windows pipe may be cp1251)."""
    try:
        chars.encode(sys.stdout.encoding or "ascii")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


FULL, EMPTY, AXIS = ("█", "░", "│") if _can_print("█░│") else ("#", ".", "|")


class Result(NamedTuple):
    size: int       # response body bytes
    elapsed: float  # total request time, s
    ttfb: float     # time to first byte: DNS + TCP + TLS + waiting for headers, s
    content_type: str | None = None  # MIME type from Content-Type, without params


# exact MIME types -> human-readable category
CONTENT_LABELS = {
    "text/html": "HTML page",
    "application/xhtml+xml": "HTML page",
    "application/json": "JSON",
    "application/xml": "XML",
    "text/xml": "XML",
    "application/javascript": "JavaScript",
    "text/javascript": "JavaScript",
    "text/css": "CSS",
    "application/pdf": "PDF document",
    "application/zip": "archive",
    "application/gzip": "archive",
    "application/x-tar": "archive",
    "application/x-7z-compressed": "archive",
    "application/vnd.rar": "archive",
    "application/x-rar-compressed": "archive",
    "application/octet-stream": "binary file",
}
# no exact match -> fall back to the top-level type before "/"
CONTENT_PREFIXES = {
    "image": "image",
    "video": "video",
    "audio": "audio",
    "font": "font",
    "text": "text",
}


def describe_content(content_type: str | None, url: str) -> str:
    """Content category from Content-Type; without the header, guess from the URL extension."""
    source = ""
    if not content_type:
        content_type, _ = mimetypes.guess_type(urllib.parse.urlsplit(url).path)
        if not content_type:
            return "unknown (no Content-Type)"
        source = ", guessed from URL"
    label = CONTENT_LABELS.get(content_type) or CONTENT_PREFIXES.get(
        content_type.split("/")[0], "file"
    )
    return f"{label} ({content_type}{source})"


def mbit_per_s(n_bytes: int, seconds: float) -> float:
    """Bytes over seconds -> megabits per second (1 Mbit = 1,000,000 bits)."""
    return n_bytes * 8 / seconds / 1_000_000 if seconds > 0 else 0.0


def format_bytes(n: float) -> str:
    """Human-readable size in decimal units (1 MB = 1,000,000 bytes)."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000:
            return f"{n:.2f} {unit}"
        n /= 1000
    return f"{n:.2f} TB"


def add_cache_buster(url: str, token: str) -> str:
    """Append the _=<token> parameter to the URL, keeping the existing query string."""
    parts = urllib.parse.urlsplit(url)
    query = f"{parts.query}&_={token}" if parts.query else f"_={token}"
    return urllib.parse.urlunsplit(parts._replace(query=query))


def bar(fraction: float, width: int) -> str:
    filled = round(min(max(fraction, 0.0), 1.0) * width)
    return FULL * filled + EMPTY * (width - filled)


class ProgressLine:
    """Single-line live progress, redrawn in place with '\\r'.

    When stdout is not a terminal (pipe, file), intermediate states are skipped
    and only the final line is printed.
    """

    def __init__(self, prefix: str, enabled: bool):
        self.prefix = prefix
        self.enabled = enabled
        self.last_draw = 0.0
        self.last_len = 0

    def update(self, received: int, total: int | None, elapsed: float) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        if now - self.last_draw < REDRAW_INTERVAL:
            return
        self.last_draw = now
        speed = mbit_per_s(received, elapsed)
        if total:  # server sent Content-Length, so a percentage is possible
            frac = received / total
            text = (
                f"{self.prefix} {bar(frac, BAR_WIDTH)} {frac:4.0%}"
                f"  {received / 1e6:6.2f}/{total / 1e6:.2f} MB  {speed:7.2f} Mbit/s"
            )
        else:  # size unknown in advance: show bytes received and speed only
            text = f"{self.prefix} {format_bytes(received):>10}  {speed:7.2f} Mbit/s"
        self._write(text)

    def finish(self, text: str) -> None:
        if self.enabled:
            self._write(text, end="\n")
        else:
            print(text)

    def _write(self, text: str, end: str = "") -> None:
        # pad with spaces to wipe the tail of a previous, longer line
        pad = " " * max(self.last_len - len(text), 0)
        sys.stdout.write("\r" + text + pad + end)
        sys.stdout.flush()
        self.last_len = len(text)


def fetch(
    url: str,
    timeout: float,
    max_time: float | None = None,
    progress: ProgressLine | None = None,
) -> Result:
    """Download the whole URL.

    timeout  - timeout of every socket operation (that is how urllib works);
    max_time - overall deadline for the request, checked between reads.
    """
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )
    received = 0
    start = time.perf_counter()
    # every urlopen opens a new connection, so the time includes DNS, TCP and TLS
    with urllib.request.urlopen(request, timeout=timeout) as response:
        ttfb = time.perf_counter() - start  # urlopen returns once headers arrive
        # without the header get_content_type() would default to "text/plain", so check explicitly
        content_type = (
            response.headers.get_content_type() if response.headers.get("Content-Type") else None
        )
        length = response.headers.get("Content-Length")
        total = int(length) if length and length.isdigit() else None
        # read1 returns whatever one socket read delivered instead of waiting for a full
        # CHUNK_SIZE, so the deadline and progress are checked often even on a slow server
        while chunk := response.read1(CHUNK_SIZE):
            received += len(chunk)
            now = time.perf_counter() - start
            if max_time is not None and now > max_time:
                raise TimeoutError(f"request exceeded --max-time {max_time:g} s")
            if progress:
                progress.update(received, total, now)
        # unlike read(), read1() does not raise IncompleteRead when the connection drops,
        # it just returns b"", so verify that the whole body arrived
        if total is not None and received < total:
            raise http.client.HTTPException(
                f"connection closed after {received} of {total} bytes"
            )
    elapsed = time.perf_counter() - start
    return Result(received, elapsed, ttfb, content_type)


def print_chart(speeds: list[float | None], overall: float) -> None:
    """Horizontal bar chart of the speed of every request."""
    peak = max(s for s in speeds if s is not None)
    idx_w = len(str(len(speeds)))
    print(f"\nSpeed per request, Mbit/s (overall {overall:.2f}):")
    for i, s in enumerate(speeds, 1):
        if s is None:
            print(f"  {i:>{idx_w}} {AXIS} error")
            continue
        n = round(s / peak * CHART_WIDTH) if peak > 0 else 0
        print(f"  {i:>{idx_w}} {AXIS}{FULL * n}{' ' * (CHART_WIDTH - n)} {s:8.2f}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make N sequential HTTP requests to a URL and measure download speed.",
    )
    parser.add_argument("url", nargs="?", help="URL to download, e.g. a large image")
    parser.add_argument(
        "--test", action="store_true", help=f"use the built-in test URL ({TEST_URL})"
    )
    parser.add_argument(
        "-n", "--count", type=int, default=10, help="number of requests (default: 10)"
    )
    parser.add_argument(
        "-t", "--timeout", type=float, default=30.0,
        help="socket timeout per operation (connect / each read), seconds (default: 30)",
    )
    parser.add_argument(
        "--max-time", type=float, default=120.0,
        help="total time limit for one request, seconds; 0 disables it (default: 120)",
    )
    parser.add_argument(
        "--bust-cache", action="store_true",
        help="append ?_=<timestamp> to the URL so CDNs can't serve a cached copy "
             "(breaks signed URLs)",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="disable the live progress bar"
    )
    parser.add_argument("--no-chart", action="store_true", help="do not print the speed chart")
    args = parser.parse_args(argv)
    if args.test and args.url:
        parser.error("pass either a URL or --test, not both")
    if args.test:
        args.url = TEST_URL
    if not args.url:
        parser.error("URL is required (or use --test)")
    if args.count < 1:
        parser.error("--count must be >= 1")
    if args.timeout <= 0:
        parser.error("--timeout must be > 0")
    if args.max_time < 0:
        parser.error("--max-time must be >= 0")
    if not args.url.startswith(("http://", "https://")):
        parser.error("URL must start with http:// or https://")
    return args


def run(args: argparse.Namespace) -> int:
    print(f"URL:      {args.url}")
    print(f"Requests: {args.count}\n")

    live = sys.stdout.isatty() and not args.no_progress
    max_time = args.max_time or None  # 0 -> no overall deadline
    idx_w = len(str(args.count))
    speeds: list[float | None] = []  # per-request speed for the chart, None = failed
    results: list[Result] = []

    # strictly sequential: the next request starts after the previous body is fully read
    for i in range(1, args.count + 1):
        progress = ProgressLine(f"[{i:>{idx_w}}/{args.count}]", enabled=live)
        url = add_cache_buster(args.url, str(time.time_ns())) if args.bust_cache else args.url
        try:
            r = fetch(url, args.timeout, max_time, progress)
        except REQUEST_ERRORS as exc:
            # a failed request doesn't stop the others and is excluded from the stats
            progress.finish(f"{progress.prefix} error: {type(exc).__name__}: {exc}")
            speeds.append(None)
            continue
        results.append(r)
        speed = mbit_per_s(r.size, r.elapsed)
        speeds.append(speed)
        progress.finish(
            f"{progress.prefix} {format_bytes(r.size):>10}  {r.elapsed:7.3f} s"
            f"  (TTFB {r.ttfb:6.3f} s)  {speed:8.2f} Mbit/s"
        )

    print()
    if not results:
        sys.stdout.flush()  # otherwise, in a pipe, stderr overtakes buffered stdout
        print("All requests failed.", file=sys.stderr)
        return EXIT_ALL_FAILED

    times = [r.elapsed for r in results]
    ok_speeds = [s for s in speeds if s is not None]
    total_bytes = sum(r.size for r in results)
    total_time = sum(times)
    total_ttfb = sum(r.ttfb for r in results)
    # overall speed = total bytes downloaded / total download time
    overall = mbit_per_s(total_bytes, total_time)
    # body transfer only: excluding DNS / TCP / TLS / waiting for headers
    transfer = mbit_per_s(total_bytes, total_time - total_ttfb)

    # types come from the responses; if the server varied, list every distinct one
    types = list(dict.fromkeys(r.content_type for r in results))
    print(f"Content type:        {', '.join(describe_content(t, args.url) for t in types)}")
    print(f"Successful requests: {len(results)}/{args.count}")
    print(
        f"Request time:        avg {total_time / len(results):.3f} s"
        f"  (min {min(times):.3f} / median {statistics.median(times):.3f} / max {max(times):.3f})"
    )
    print(f"Average TTFB:        {total_ttfb / len(results):.3f} s")
    print(f"Total downloaded:    {format_bytes(total_bytes)} ({total_bytes} bytes)")
    print(f"Download speed:      {overall:.2f} Mbit/s ({overall / 8:.2f} MB/s)")
    print(
        f"Speed per request:   median {statistics.median(ok_speeds):.2f} Mbit/s"
        f"  (min {min(ok_speeds):.2f} / max {max(ok_speeds):.2f})"
    )
    print(f"Transfer speed:      {transfer:.2f} Mbit/s (excluding TTFB)")
    if any(t in ("text/html", "application/xhtml+xml") for t in types):
        print(
            "\nNote: this is an HTML page - only the document itself is downloaded "
            "(no images, CSS, JS).\n      It is usually small, so the result reflects "
            "latency more than bandwidth."
        )
    if not args.no_chart:
        print_chart(speeds, overall)
    return EXIT_OK if len(results) == args.count else EXIT_PARTIAL


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
