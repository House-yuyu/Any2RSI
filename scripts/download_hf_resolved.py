"""Resume a large public Hugging Face file through its resolved CDN URL.

Some institutional proxies break huggingface_hub's metadata HEAD request, and
aria2 cannot infer the size through the Hub's first redirect.  Resolving the
redirect explicitly preserves aria2's parallel range downloads.
"""
from __future__ import annotations

import argparse
import subprocess
import urllib.error
import urllib.request
from pathlib import Path


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def resolve(url):
    opener = urllib.request.build_opener(NoRedirect)
    request = urllib.request.Request(url, method="HEAD")
    try:
        opener.open(request)
    except urllib.error.HTTPError as error:
        if error.code not in (301, 302, 303, 307, 308):
            raise
        location = error.headers.get("Location")
        if location:
            return location
    raise RuntimeError(f"no redirect location returned for {url}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output")
    parser.add_argument("--connections", type=int, default=16)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cdn_url = resolve(args.url)
    subprocess.run(
        [
            "aria2c", "-c", "-x", str(args.connections),
            "-s", str(args.connections), "-k", "1M",
            "--file-allocation=none", "--max-tries=10", "--retry-wait=5",
            "-d", str(output.parent), "-o", output.name, cdn_url,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
