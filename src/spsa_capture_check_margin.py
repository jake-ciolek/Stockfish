#!/usr/bin/env python3
"""
SPSA tuner for capture/check SEE margin parameters using cutechess-cli.

Targets:
  - CaptureCheckSeeMarginBonus   in [-256, 256]
  - CaptureCheckSeeMarginDepth   in [-320, 320]
  - CaptureCheckSeeMarginHistDiv in [-96, 96]
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import pickle
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Param:
    name: str
    lo: int
    hi: int
    theta: float
    a: float
    c: float


SCORE_RE = re.compile(
    r"Score of plus vs minus:\s+(\d+)\s+-\s+(\d+)\s+-\s+(\d+)\s+\[([0-9.]+)\]\s+(\d+)"
)


def clip_round(value: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(value))))


def build_cutechess_command(
    args: argparse.Namespace,
    plus: Dict[str, int],
    minus: Dict[str, int],
    iter_seed: int,
) -> List[str]:
    cmd = [
        args.cutechess,
        "-engine",
        "name=plus",
        f"cmd={args.engine}",
        "proto=uci",
        f"option.Threads={args.threads}",
        f"option.Hash={args.hash}",
    ]

    for k, v in plus.items():
        cmd.append(f"option.{k}={v}")
    for opt in args.engine_option:
        cmd.append(f"option.{opt}")

    cmd += [
        "-engine",
        "name=minus",
        f"cmd={args.engine}",
        "proto=uci",
        f"option.Threads={args.threads}",
        f"option.Hash={args.hash}",
    ]

    for k, v in minus.items():
        cmd.append(f"option.{k}={v}")
    for opt in args.engine_option:
        cmd.append(f"option.{opt}")

    cmd += [
        "-openings",
        f"file={args.book}",
        "format=epd",
        f"order={args.openings_order}",
        f"plies={args.openings_plies}",
        "-repeat",
        "-games",
        str(args.games),
        "-rounds",
        str(args.rounds),
        "-concurrency",
        str(args.concurrency),
        "-draw",
        f"movenumber={args.draw_movenumber}",
        f"movecount={args.draw_movecount}",
        f"score={args.draw_score}",
        "-resign",
        f"movecount={args.resign_movecount}",
        f"score={args.resign_score}",
        f"twosided={str(args.resign_twosided).lower()}",
        "-each",
        f"tc={args.tc}",
        f"timemargin={args.timemargin}",
        "-srand",
        str(iter_seed),
    ]

    if args.pgnout:
        cmd += ["-pgnout", args.pgnout, "min"]

    return cmd


def run_and_capture(command: List[str], verbose: bool = True) -> Tuple[int, str]:
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        output_lines.append(line)
        if verbose:
            print(line, end="")

    return_code = proc.wait()
    return return_code, "".join(output_lines)


def parse_score(cutechess_output: str) -> Dict[str, float]:
    matches = SCORE_RE.findall(cutechess_output)
    if not matches:
        raise RuntimeError("Could not parse cutechess score line from output.")

    wins, losses, draws, p_txt, games = matches[-1]
    w = int(wins)
    l = int(losses)
    d = int(draws)
    n = int(games)
    p = float(p_txt)

    if n <= 0:
        raise RuntimeError("Invalid game count parsed from cutechess output.")

    p_check = (w + 0.5 * d) / n
    if abs(p - p_check) > 1e-3:
        p = p_check

    return {"wins": w, "losses": l, "draws": d, "games": n, "p": p}


def append_jsonl(path: Path, record: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


def append_csv(path: Path, record: Dict) -> None:
    header = [
        "iteration",
        "p",
        "wins",
        "losses",
        "draws",
        "games",
        "theta_bonus",
        "theta_depth",
        "theta_hist_div",
        "plus_bonus",
        "plus_depth",
        "plus_hist_div",
        "minus_bonus",
        "minus_depth",
        "minus_hist_div",
        "ak_bonus",
        "ak_depth",
        "ak_hist_div",
        "ck_bonus",
        "ck_depth",
        "ck_hist_div",
        "elapsed_sec",
        "seed",
    ]

    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def save_state(path: Path, iteration: int, params: List[Param], rng: random.Random) -> None:
    payload = {
        "iteration": iteration,
        "params": {p.name: p.theta for p in params},
        "rng_state_b64": base64.b64encode(pickle.dumps(rng.getstate())).decode("ascii"),
        "timestamp": int(time.time()),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_state(path: Path, params: List[Param], rng: random.Random) -> int:
    raw = json.loads(path.read_text(encoding="utf-8"))
    it = int(raw.get("iteration", 0))
    values = raw.get("params", {})
    for p in params:
        if p.name in values:
            p.theta = float(values[p.name])

    if "rng_state_b64" in raw:
        state = pickle.loads(base64.b64decode(raw["rng_state_b64"]))
        rng.setstate(state)
    elif "seed" in raw and raw["seed"] is not None:
        rng.seed(int(raw["seed"]))

    return it


def build_params(args: argparse.Namespace) -> List[Param]:
    return [
        Param(
            name="CaptureCheckSeeMarginBonus",
            lo=-256,
            hi=256,
            theta=float(args.init_bonus),
            a=float(args.a_bonus),
            c=float(args.c_bonus),
        ),
        Param(
            name="CaptureCheckSeeMarginDepth",
            lo=-320,
            hi=320,
            theta=float(args.init_depth),
            a=float(args.a_depth),
            c=float(args.c_depth),
        ),
        Param(
            name="CaptureCheckSeeMarginHistDiv",
            lo=-96,
            hi=96,
            theta=float(args.init_hist_div),
            a=float(args.a_hist_div),
            c=float(args.c_hist_div),
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local SPSA tuner for capture/check SEE margin parameters with cutechess-cli."
    )

    parser.add_argument(
        "--cutechess",
        default="/Users/jakubciolek/25dec/cutechess/build/cutechess-cli",
        help="Path to cutechess-cli.",
    )
    parser.add_argument(
        "--engine",
        default=str(Path(__file__).resolve().parent / "stockfish"),
        help="Path to Stockfish binary used for both sides.",
    )
    parser.add_argument(
        "--book",
        default="/Users/jakubciolek/25dec/books/UHO_Lichess_4852_v1.epd",
        help="Path to EPD opening book.",
    )
    parser.add_argument("--iterations", type=int, default=100, help="SPSA iterations.")
    parser.add_argument("--threads", type=int, default=1, help="Engine Threads option.")
    parser.add_argument("--hash", type=int, default=16, help="Engine Hash option (MB).")
    parser.add_argument("--games", type=int, default=2, help="Cutechess -games value.")
    parser.add_argument("--rounds", type=int, default=64, help="Cutechess -rounds value.")
    parser.add_argument(
        "--concurrency", type=int, default=8, help="Cutechess -concurrency value."
    )
    parser.add_argument("--tc", default="8+0.08", help="Time control (cutechess tc).")
    parser.add_argument(
        "--timemargin", type=int, default=50, help="Cutechess timemargin in ms."
    )
    parser.add_argument(
        "--openings-order",
        default="random",
        choices=["random", "sequential"],
        help="Opening order for cutechess.",
    )
    parser.add_argument(
        "--openings-plies", type=int, default=8, help="Opening plies from EPD."
    )
    parser.add_argument("--draw-movenumber", type=int, default=34)
    parser.add_argument("--draw-movecount", type=int, default=8)
    parser.add_argument("--draw-score", type=int, default=20)
    parser.add_argument("--resign-movecount", type=int, default=3)
    parser.add_argument("--resign-score", type=int, default=600)
    parser.add_argument(
        "--resign-twosided",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set twosided resign adjudication.",
    )
    parser.add_argument(
        "--engine-option",
        action="append",
        default=[],
        help="Extra UCI option as Name=Value (repeatable).",
    )

    parser.add_argument("--alpha", type=float, default=0.602, help="SPSA alpha.")
    parser.add_argument("--gamma", type=float, default=0.101, help="SPSA gamma.")
    parser.add_argument(
        "--A", type=float, default=20.0, help="SPSA stability constant A."
    )
    parser.add_argument(
        "--seed", type=int, default=20260210, help="RNG seed for SPSA signs and -srand."
    )

    parser.add_argument("--init-bonus", type=int, default=72)
    parser.add_argument("--init-depth", type=int, default=166)
    parser.add_argument("--init-hist-div", type=int, default=29)

    parser.add_argument(
        "--a-bonus", type=float, default=6.0, help="Base a for CaptureCheckSeeMarginBonus."
    )
    parser.add_argument(
        "--a-depth", type=float, default=10.0, help="Base a for CaptureCheckSeeMarginDepth."
    )
    parser.add_argument(
        "--a-hist-div", type=float, default=3.0, help="Base a for CaptureCheckSeeMarginHistDiv."
    )
    parser.add_argument(
        "--c-bonus", type=float, default=4.0, help="Base c for CaptureCheckSeeMarginBonus."
    )
    parser.add_argument(
        "--c-depth", type=float, default=8.0, help="Base c for CaptureCheckSeeMarginDepth."
    )
    parser.add_argument(
        "--c-hist-div", type=float, default=2.0, help="Base c for CaptureCheckSeeMarginHistDiv."
    )

    parser.add_argument(
        "--log-dir",
        default=str(Path.cwd() / "spsa_capture_check_margin_logs"),
        help="Directory for logs/state.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from state file in --log-dir.",
    )
    parser.add_argument(
        "--pgnout",
        default="",
        help="Optional cutechess PGN output file path. Empty disables PGN output.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress cutechess live output.",
    )

    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    for p in (args.cutechess, args.engine, args.book):
        if not Path(p).exists():
            raise FileNotFoundError(f"Required path does not exist: {p}")


def main() -> int:
    args = parse_args()
    validate_paths(args)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    state_path = log_dir / "state.json"
    jsonl_path = log_dir / "history.jsonl"
    csv_path = log_dir / "history.csv"

    params = build_params(args)
    rng = random.Random(args.seed)
    start_iter = 0

    if args.resume and state_path.exists():
        start_iter = load_state(state_path, params, rng)
        print(f"Resuming from iteration {start_iter + 1}.")

    print("Initial theta values:")
    for p in params:
        print(f"  {p.name}={clip_round(p.theta, p.lo, p.hi)}")

    last_completed = start_iter
    for iteration in range(start_iter + 1, start_iter + args.iterations + 1):
        t0 = time.time()

        deltas: Dict[str, int] = {}
        plus: Dict[str, int] = {}
        minus: Dict[str, int] = {}
        ak: Dict[str, float] = {}
        ck: Dict[str, float] = {}

        for p in params:
            deltas[p.name] = rng.choice([-1, 1])
            ak[p.name] = p.a / math.pow(iteration + args.A, args.alpha)
            ck[p.name] = p.c / math.pow(iteration, args.gamma)
            plus[p.name] = clip_round(p.theta + ck[p.name] * deltas[p.name], p.lo, p.hi)
            minus[p.name] = clip_round(p.theta - ck[p.name] * deltas[p.name], p.lo, p.hi)

        iter_seed = rng.randint(1, 2_147_483_647)
        cmd = build_cutechess_command(args, plus, minus, iter_seed)

        print(
            f"\n[iter {iteration}] plus={plus} minus={minus} ak={{{', '.join(f'{k}:{ak[k]:.4f}' for k in ak)}}}"
        )

        rc, out = run_and_capture(cmd, verbose=not args.quiet)
        if rc != 0:
            raise RuntimeError(f"cutechess-cli failed with exit code {rc}")

        score = parse_score(out)
        y = score["p"] - 0.5

        for p in params:
            # ghat_i = y / (ck_i * delta_i) and theta_i += ak_i * ghat_i
            p.theta += ak[p.name] * y / (ck[p.name] * deltas[p.name])
            p.theta = max(float(p.lo), min(float(p.hi), p.theta))

        elapsed = time.time() - t0
        print(
            f"[iter {iteration}] p={score['p']:.4f} WLD={score['wins']}-{score['losses']}-{score['draws']} elapsed={elapsed:.1f}s"
        )
        print(
            f"[iter {iteration}] theta={{"
            + ", ".join(
                f"{p.name}:{clip_round(p.theta, p.lo, p.hi)}" for p in params
            )
            + "}"
        )

        record = {
            "iteration": iteration,
            "p": score["p"],
            "wins": score["wins"],
            "losses": score["losses"],
            "draws": score["draws"],
            "games": score["games"],
            "theta_bonus": clip_round(params[0].theta, params[0].lo, params[0].hi),
            "theta_depth": clip_round(params[1].theta, params[1].lo, params[1].hi),
            "theta_hist_div": clip_round(params[2].theta, params[2].lo, params[2].hi),
            "plus_bonus": plus["CaptureCheckSeeMarginBonus"],
            "plus_depth": plus["CaptureCheckSeeMarginDepth"],
            "plus_hist_div": plus["CaptureCheckSeeMarginHistDiv"],
            "minus_bonus": minus["CaptureCheckSeeMarginBonus"],
            "minus_depth": minus["CaptureCheckSeeMarginDepth"],
            "minus_hist_div": minus["CaptureCheckSeeMarginHistDiv"],
            "ak_bonus": ak["CaptureCheckSeeMarginBonus"],
            "ak_depth": ak["CaptureCheckSeeMarginDepth"],
            "ak_hist_div": ak["CaptureCheckSeeMarginHistDiv"],
            "ck_bonus": ck["CaptureCheckSeeMarginBonus"],
            "ck_depth": ck["CaptureCheckSeeMarginDepth"],
            "ck_hist_div": ck["CaptureCheckSeeMarginHistDiv"],
            "elapsed_sec": elapsed,
            "seed": iter_seed,
        }
        append_jsonl(jsonl_path, record)
        append_csv(csv_path, record)
        save_state(state_path, iteration, params, rng)
        last_completed = iteration

    save_state(state_path, last_completed, params, rng)

    print("\nFinal theta values:")
    for p in params:
        print(f"  {p.name}={clip_round(p.theta, p.lo, p.hi)}")

    print(f"\nLogs: {jsonl_path}")
    print(f"Logs: {csv_path}")
    print(f"State: {state_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
