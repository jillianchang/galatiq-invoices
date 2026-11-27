"""Acme Corp invoice processing: multi-agent pipeline CLI.

    python main.py --invoice_path=data/invoices/invoice_1001.txt
    python main.py --invoice_dir=data/invoices --reset

Structured JSON log lines go to stderr, followed by a readable results panel;
the final result JSON goes to stdout. --json prints only the JSON.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import sys
from pathlib import Path

from rich.console import Console

from src.config import Settings, load_policy
from src.db import DEFAULT_DB_PATH, connect, init_db, reset_db
from src.llm.client import LLMClient
from src.observability import RunLogger, new_run_id
from src.pipeline import build_graph, list_invoices, make_services, process_file
from src.report import render_result, render_summary


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process invoices end to end.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--invoice_path", type=Path, help="one invoice file")
    src.add_argument("--invoice_dir", type=Path, help="every invoice in a folder, in sorted order")
    p.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="SQLite database path")
    p.add_argument("--reset", action="store_true", help="clear processing history first")
    p.add_argument("--json", action="store_true", help="JSON output only (no panels)")
    p.add_argument(
        "--llm",
        choices=["record", "replay", "live"],
        help="LLM mode (default: record if LLM_API_KEY is set, otherwise replay)",
    )
    p.add_argument(
        "--approver",
        choices=["agent"],
        default="agent",
        help="who approves: the approval agent acts as the simulated VP",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.invoice_path:
        if not args.invoice_path.is_file():
            print(f"error: {args.invoice_path} is not a file", file=sys.stderr)
            return 2
        files = [args.invoice_path]
    else:
        if not args.invoice_dir.is_dir():
            print(f"error: {args.invoice_dir} is not a folder", file=sys.stderr)
            return 2
        files = list_invoices(args.invoice_dir)

    policy = load_policy()
    settings = Settings.from_env()
    if args.llm:
        settings = dataclasses.replace(settings, mode=args.llm)

    conn = connect(args.db)
    reset_db(conn) if args.reset else init_db(conn)
    logger = RunLogger(new_run_id(), conn)
    llm = LLMClient(settings, policy.llm, on_event=logger.llm_event)
    svc = make_services(conn, policy, llm, logger)
    graph = build_graph(svc)
    console = Console(stderr=True)

    results = []
    for path in files:
        # The spec's mock_payment prints to stdout; keep stdout clean for the JSON result.
        with contextlib.redirect_stdout(sys.stderr):
            result = process_file(svc, graph, path)
        results.append(result)
        if not args.json:
            render_result(result, console)

    if len(results) > 1 and not args.json:
        render_summary(results, console)
    output = results[0].to_json() if len(results) == 1 else [r.to_json() for r in results]
    print(json.dumps(output, indent=2, default=str))
    if not args.json:
        console.print(
            f"[dim]Trace: {logger.trace_path}   Run: {logger.run_id}   Mode: {settings.mode}[/]"
        )
    logger.close()
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
