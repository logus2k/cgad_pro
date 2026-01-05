# Automated Benchmark Suite

Runs FEM solver benchmarks across all implementations with progress tracking and time estimation.

## Quick Start

```bash
cd /src/app/server/automated_benchmark
python run_benchmark.py
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `--solver`, `-s` | Run single solver: `cpu`, `cpu_threaded`, `cpu_multiprocess`, `numba`, `numba_cuda`, `gpu` |
| `--model`, `-m` | Run single model: `Y-Shaped`, `Venturi`, `S-Bend`, `T-Junction`, `Backward-Facing Step`, `Elbow 90` |
| `--max-nodes` | Limit mesh size (e.g., `--max-nodes 1000` for small meshes only) |
| `--runs`, `-r` | Override runs per test (default: 3) |
| `--dry-run`, `-n` | Preview test matrix without executing |
| `--clear` | Clear previous automated results before running |
| `--config`, `-c` | Custom config file path |
| `--gallery`, `-g` | Override gallery file path |
| `--output-dir`, `-o` | Override output directory |

## Examples

```bash
# Full suite (144 tests: 6 solvers x 6 models x 4 meshes)
python run_benchmark.py

# GPU solver only
python run_benchmark.py --solver gpu

# Single model
python run_benchmark.py --model Venturi

# Quick smoke test (small meshes)
python run_benchmark.py --max-nodes 1000

# Preview without running
python run_benchmark.py --dry-run

# Fresh start
python run_benchmark.py --clear
```

## Output

Results saved to `/src/app/server/benchmark/benchmark_{server_hash}_automated.json`

Compatible with existing benchmark UI - automated results appear as "{SERVER} (Automated)".

## Configuration

Edit `testing_procedure.json` to customize:
- `execution.runs_per_test` - Number of timed runs (default: 3)
- `execution.warmup_runs` - Warmup runs before timing (default: 1)
- `execution.abort_on_failure` - Stop on first failure (default: true)
- `solver_params.max_iterations` - Solver iteration limit (default: 15000)
- `filters` - Enable/disable specific solvers, models, or mesh sizes
