#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple


class UCIEngine:
    def __init__(self, path: Path) -> None:
        self.proc = subprocess.Popen(
            [str(path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, cmd: str) -> None:
        if not self.proc.stdin:
            raise RuntimeError("stdin closed")
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _readline(self) -> str:
        if not self.proc.stdout:
            raise RuntimeError("stdout closed")
        line = self.proc.stdout.readline()
        if line == "":
            raise RuntimeError("engine terminated")
        return line.strip()

    def _wait_for(self, token: str) -> None:
        while True:
            line = self._readline()
            if line == token:
                return

    def set_option(self, name: str, value: Optional[str] = None) -> None:
        if value is None:
            self._send(f"setoption name {name}")
        else:
            self._send(f"setoption name {name} value {value}")
        self._send("isready")
        self._wait_for("readyok")

    def uci_new_game(self) -> None:
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok")

    def position(self, fen: str) -> None:
        self._send(f"position fen {fen}")

    def go_depth(self, depth: int) -> Tuple[Optional[Tuple[str, str]], Optional[int], str]:
        self._send(f"go depth {depth}")
        last_score: Optional[Tuple[str, str]] = None
        last_nodes: Optional[int] = None
        while True:
            line = self._readline()
            if line.startswith("info "):
                tokens = line.split()
                if "score" in tokens:
                    idx = tokens.index("score")
                    if idx + 2 < len(tokens):
                        last_score = (tokens[idx + 1], tokens[idx + 2])
                if "nodes" in tokens:
                    idx = tokens.index("nodes")
                    if idx + 1 < len(tokens):
                        try:
                            last_nodes = int(tokens[idx + 1])
                        except ValueError:
                            pass
            elif line.startswith("bestmove"):
                parts = line.split()
                bestmove = parts[1] if len(parts) > 1 else ""
                return last_score, last_nodes, bestmove

    def quit(self) -> None:
        self._send("quit")
        self.proc.wait(timeout=5)


def read_positions(path: Path, max_positions: int) -> list[str]:
    positions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            if len(parts) >= 6:
                fen = " ".join(parts[:6])
            else:
                fen = " ".join(parts[:4] + ["0", "1"])
            positions.append(fen)
            if len(positions) >= max_positions:
                break
    return positions


def score_to_cp(score: Optional[Tuple[str, str]]) -> Optional[int]:
    if not score:
        return None
    if score[0] != "cp":
        return None
    try:
        return int(score[1])
    except ValueError:
        return None


def score_to_str(score: Optional[Tuple[str, str]]) -> str:
    if not score:
        return ""
    return f"{score[0]} {score[1]}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run baseline vs shadow searches and compare evals."
    )
    parser.add_argument(
        "--stockfish",
        default="/Users/jakubciolek/25dec/Stockfish/src/stockfish",
        help="Path to Stockfish binary.",
    )
    parser.add_argument(
        "--epd",
        default="/Users/jakubciolek/25dec/books/UHO_Lichess_4852_v1.epd",
        help="EPD/FEN file path.",
    )
    parser.add_argument("--positions", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", type=int, default=128)
    parser.add_argument("--baseline-malus", type=int, default=0)
    parser.add_argument("--shadow-malus", type=int, default=-40)
    parser.add_argument("--verify-depth", type=int, default=0)
    parser.add_argument("--verify-malus", type=int, default=None)
    parser.add_argument(
        "--out-csv",
        default="/Users/jakubciolek/25dec/Stockfish/shadow_eval.csv",
    )
    parser.add_argument("--clear-hash", action="store_true")
    args = parser.parse_args()

    stockfish = Path(args.stockfish)
    epd_path = Path(args.epd)
    out_csv = Path(args.out_csv)

    positions = read_positions(epd_path, args.positions)
    if not positions:
        print("No positions loaded from file.", file=sys.stderr)
        return 2

    base = UCIEngine(stockfish)
    shadow = UCIEngine(stockfish)
    verify = UCIEngine(stockfish) if args.verify_depth > 0 else None

    for engine in (base, shadow, verify):
        if engine is None:
            continue
        engine.set_option("Threads", str(args.threads))
        engine.set_option("Hash", str(args.hash))
        engine.set_option("MultiPV", "1")

    base.set_option("SEEMarginMalus", str(args.baseline_malus))
    shadow.set_option("SEEMarginMalus", str(args.shadow_malus))
    if verify is not None:
        verify_malus = (
            args.verify_malus if args.verify_malus is not None else args.baseline_malus
        )
        verify.set_option("SEEMarginMalus", str(verify_malus))

    base.uci_new_game()
    shadow.uci_new_game()
    if verify is not None:
        verify.uci_new_game()

    diffs = []
    base_verify_diffs = []
    shadow_verify_diffs = []
    base_closer = 0
    shadow_closer = 0
    verify_ties = 0
    total = len(positions)
    start = time.time()

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "idx",
            "fen",
            "base_score",
            "base_cp",
            "base_nodes",
            "base_bestmove",
            "shadow_score",
            "shadow_cp",
            "shadow_nodes",
            "shadow_bestmove",
            "diff_cp",
        ]
        if verify is not None:
            header += [
                "verify_score",
                "verify_cp",
                "verify_nodes",
                "verify_bestmove",
                "base_vs_verify_cp",
                "shadow_vs_verify_cp",
                "closer",
            ]
        writer.writerow(header)

        for idx, fen in enumerate(positions, start=1):
            if args.clear_hash:
                base.set_option("Clear Hash")
                shadow.set_option("Clear Hash")
                if verify is not None:
                    verify.set_option("Clear Hash")

            base.position(fen)
            base_score, base_nodes, base_move = base.go_depth(args.depth)

            shadow.position(fen)
            shadow_score, shadow_nodes, shadow_move = shadow.go_depth(args.depth)

            base_cp = score_to_cp(base_score)
            shadow_cp = score_to_cp(shadow_score)
            diff_cp = shadow_cp - base_cp if base_cp is not None and shadow_cp is not None else ""

            if diff_cp != "":
                diffs.append(diff_cp)

            row = [
                idx,
                fen,
                score_to_str(base_score),
                base_cp if base_cp is not None else "",
                base_nodes if base_nodes is not None else "",
                base_move,
                score_to_str(shadow_score),
                shadow_cp if shadow_cp is not None else "",
                shadow_nodes if shadow_nodes is not None else "",
                shadow_move,
                diff_cp,
            ]

            if verify is not None:
                verify.position(fen)
                verify_score, verify_nodes, verify_move = verify.go_depth(args.verify_depth)
                verify_cp = score_to_cp(verify_score)
                base_vs_verify = (
                    base_cp - verify_cp if base_cp is not None and verify_cp is not None else ""
                )
                shadow_vs_verify = (
                    shadow_cp - verify_cp
                    if shadow_cp is not None and verify_cp is not None
                    else ""
                )

                closer = ""
                if (
                    base_cp is not None
                    and shadow_cp is not None
                    and verify_cp is not None
                ):
                    base_abs = abs(base_cp - verify_cp)
                    shadow_abs = abs(shadow_cp - verify_cp)
                    base_verify_diffs.append(base_abs)
                    shadow_verify_diffs.append(shadow_abs)
                    if base_abs < shadow_abs:
                        base_closer += 1
                        closer = "baseline"
                    elif shadow_abs < base_abs:
                        shadow_closer += 1
                        closer = "shadow"
                    else:
                        verify_ties += 1
                        closer = "tie"

                row += [
                    score_to_str(verify_score),
                    verify_cp if verify_cp is not None else "",
                    verify_nodes if verify_nodes is not None else "",
                    verify_move,
                    base_vs_verify,
                    shadow_vs_verify,
                    closer,
                ]

            writer.writerow(row)

            if idx % 50 == 0 or idx == total:
                elapsed = time.time() - start
                print(f"{idx}/{total} positions in {elapsed:.1f}s", file=sys.stderr)

    base.quit()
    shadow.quit()
    if verify is not None:
        verify.quit()

    print(f"wrote: {out_csv}")

    if diffs:
        mean_diff = statistics.mean(diffs)
        median_diff = statistics.median(diffs)
        mean_abs = statistics.mean(abs(d) for d in diffs)
        max_diff = max(diffs)
        min_diff = min(diffs)
        print(f"cp_pairs={len(diffs)}")
        print(f"mean_diff={mean_diff:.2f}cp median_diff={median_diff:.2f}cp")
        print(f"mean_abs_diff={mean_abs:.2f}cp min_diff={min_diff}cp max_diff={max_diff}cp")
        for th in (5, 10, 20, 50):
            count = sum(1 for d in diffs if abs(d) >= th)
            print(f"abs_diff>={th}cp: {count}")
    else:
        print("No comparable cp scores (only mate or missing).")

    if verify is not None and base_verify_diffs and shadow_verify_diffs:
        base_mean_abs = statistics.mean(base_verify_diffs)
        shadow_mean_abs = statistics.mean(shadow_verify_diffs)
        print(f"verify_depth={args.verify_depth}")
        print(f"verify_pairs={len(base_verify_diffs)}")
        print(f"base_mean_abs_vs_verify={base_mean_abs:.2f}cp")
        print(f"shadow_mean_abs_vs_verify={shadow_mean_abs:.2f}cp")
        print(
            "closer: "
            f"shadow={shadow_closer} baseline={base_closer} ties={verify_ties}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
