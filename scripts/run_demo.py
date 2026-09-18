"""
One-command server startup, formalizing (and fixing) the manual process
used throughout development:

    python scripts/run_demo.py

What it does, and why, based on real problems hit during development:

1. Kills anything already bound to the target port first. Without this,
   a stale process from a previous run silently keeps serving old
   results while a new process fails to bind - this exact bug produced
   identical, stale API output for an entire debugging session before
   being traced back to a port conflict.

2. Never reads subprocess.PIPE output in a blocking way. An earlier
   version of this workflow hung and then killed the server process by
   interrupting a blocking `.stdout.read()` call. This script uses
   DEVNULL for the server process and a bounded polling loop against the
   actual API instead of parsing logs to determine readiness.

3. Polls the real endpoint to confirm readiness, rather than a fixed
   `time.sleep(N)`. Different environments (different data on disk,
   different GPU/CPU speed) take different amounts of time to finish
   loading the model and building the initial heatmap on startup; a
   fixed sleep either wastes time or isn't long enough.

4. Prints the exact command needed to start a public tunnel afterward,
   rather than starting one automatically - a tunnel is often unwanted
   (e.g. for purely local testing) and this keeps that decision explicit.
"""

import argparse
import os
import subprocess
import sys
import time

import requests


def kill_existing(port):
    """Kill anything already bound to this port before starting a new
    server - see module docstring for why this matters."""
    subprocess.run(["pkill", "-9", "-f", "uvicorn"], stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
    time.sleep(2)


def start_server(repo_dir, port, packages_dir):
    env = os.environ.copy()
    if packages_dir and os.path.isdir(packages_dir):
        env["PYTHONPATH"] = os.path.abspath(packages_dir)

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.server:app",
         "--host", "0.0.0.0", "--port", str(port)],
        cwd=repo_dir,
        env=env,
        stdout=subprocess.DEVNULL,   # never PIPE here - see module docstring
        stderr=subprocess.DEVNULL,
    )
    return proc


def wait_until_ready(port, timeout_seconds=90, poll_interval=3):
    """Poll the real API instead of guessing a fixed sleep duration."""
    deadline = time.time() + timeout_seconds
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            r = requests.get(f"http://localhost:{port}/api/stats", timeout=3)
            if r.status_code == 200:
                return r.json()
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(poll_interval)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", default=".",
                         help="Path to the PathoLens repo root (containing app/, src/, outputs/)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--packages_dir", default="outputs/packages",
                         help="Optional cached package directory (see PROGRESS.md Day 3 "
                              "for why this exists - lets fastapi/uvicorn be reused "
                              "across sessions without reinstalling)")
    parser.add_argument("--timeout", type=int, default=90,
                         help="Max seconds to wait for the server to become ready")
    args = parser.parse_args()

    print("Stopping any existing server on this port...")
    kill_existing(args.port)

    print("Starting server...")
    proc = start_server(args.repo_dir, args.port, args.packages_dir)

    print(f"Waiting for server to become ready (up to {args.timeout}s)...")
    stats = wait_until_ready(args.port, timeout_seconds=args.timeout)

    if stats is None:
        print(f"\nServer did not become ready within {args.timeout}s.")
        print(f"Process status (None = still running, a number = it exited): {proc.poll()}")
        print("If it's still running but not responding, the model/mosaic files "
              "may be missing from outputs/ - check that first.")
        sys.exit(1)

    print("\nServer is ready.")
    print(f"Local URL: http://localhost:{args.port}")
    print(f"Live stats: {stats}")
    print("\nTo expose this publicly via a Cloudflare quick tunnel (no account needed), run:")
    print(f"  wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared")
    print(f"  chmod +x cloudflared")
    print(f"  ./cloudflared tunnel --url http://localhost:{args.port}")


if __name__ == "__main__":
    main()