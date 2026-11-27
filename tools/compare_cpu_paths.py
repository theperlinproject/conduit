#!/usr/bin/env python3
"""Compile the same CPU benchmark against baseline and optimized Conduit source trees.

Uses a GCC/Clang-compatible compiler; no CMake, native graphics API, shader compiler,
display or GPU is required. The source file also builds as cpu_lifetime_bench via
Conduit's regular CMake/MSVC test configuration.
"""

import argparse
import hashlib
import json
import platform
import statistics
from pathlib import Path
import subprocess
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--optimized-source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--compiler", default="c++")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3, help="ABBA rounds after both binaries are compiled")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    baseline = args.baseline_source.resolve()
    optimized = args.optimized_source.resolve()
    benchmark = optimized / "tests/cpu_lifetime_bench.cpp"
    flags = ["-std=c++23", "-O3", "-DNDEBUG", "-pthread"]
    report = {
        "scope": "Isolated CPU work only; no GPU, rendering, FPS or end-to-end chunk-loading measurement",
        "platform": platform.platform(),
        "compiler": subprocess.check_output([args.compiler, "--version"], text=True).splitlines()[0],
        "compile_flags": flags,
        "benchmark_sha256": digest(benchmark),
        "run_order": "baseline, optimized, optimized, baseline per round; both binaries compiled before timing",
        "runs": [],
    }
    with tempfile.TemporaryDirectory(prefix="conduit-cpu-bench-") as scratch:
        binaries = {}
        for label, source in [("baseline", baseline), ("optimized", optimized)]:
            binary = Path(scratch) / label
            command = [args.compiler, *flags, "-I" + str(source / "src"), "-I" + str(source / "include")]
            if label == "baseline":
                command.append("-DCONDUIT_BENCH_BASELINE=1")
            command.extend([str(benchmark), "-o", str(binary)])
            subprocess.run(command, check=True)
            binaries[label] = (binary, source)
        for round_index in range(args.rounds):
            for label in ("baseline", "optimized", "optimized", "baseline"):
                binary, source = binaries[label]
                result = json.loads(subprocess.check_output([str(binary)], text=True))
                result["round"] = round_index + 1
                result["retirement_header_sha256"] = digest(source / "src/common/retire_queue.hpp")
                if label == "optimized":
                    result["binding_lookup_header_sha256"] = digest(source / "src/common/binding_lookup.hpp")
                report["runs"].append(result)
    # Workload results must agree before comparing timings.
    expected = report["runs"][0]["results"]
    for run in report["runs"][1:]:
        assert len(run["results"]) == len(expected)
        for before, after in zip(expected, run["results"]):
            for key in ("scenario", "callbacks_per_frame", "frames", "entries", "released", "checksum"):
                if key in before:
                    assert before[key] == after[key], (key, before, after)
    report["summary"] = []
    for index, scenario in enumerate(expected):
        entry = {key: scenario[key] for key in ("scenario", "callbacks_per_frame", "entries") if key in scenario}
        metrics = [key for key in scenario if key.startswith("median_ns_") or key == "allocations_per_frame"]
        for label in ("baseline", "optimized"):
            matching = [run["results"][index] for run in report["runs"] if run["implementation"] == label]
            entry[label] = {key: statistics.median(run[key] for run in matching) for key in metrics}
            entry[label]["runs"] = len(matching)
        report["summary"].append(entry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
