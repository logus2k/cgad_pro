#!/usr/bin/env python3
"""Analyze benchmark JSON files for duplicate/redundant records.

Usage:
    python analyze_benchmarks.py <benchmark_file.json>
    python analyze_benchmarks.py <benchmark_file.json> --fix
"""

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def load_gallery(benchmark_path: Path) -> dict | None:
    """Load gallery_files.json relative to benchmark file."""
    # Try relative path from benchmark location
    gallery_path = benchmark_path.parent / "../../client/config/gallery_files.json"
    gallery_path = gallery_path.resolve()
    
    if not gallery_path.exists():
        return None
    
    with open(gallery_path, 'r') as f:
        return json.load(f)


def build_expected_matrix(gallery: dict) -> set[tuple[str, str]]:
    """Build set of expected (mesh_file, solver_type) combinations."""
    expected = set()
    
    solver_ids = [s['id'] for s in gallery['solvers']]
    
    for model in gallery['models']:
        for mesh in model['meshes']:
            # Extract filename from path like "./mesh/y_tube_201.h5"
            mesh_file = Path(mesh['file']).name
            for solver_id in solver_ids:
                expected.add((mesh_file, solver_id))
    
    return expected


def analyze_benchmark_file(filepath: Path) -> dict:
    """Analyze a benchmark JSON file and return statistics."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    records = data.get('records', [])
    total = len(records)
    
    # Load gallery for expected matrix
    gallery = load_gallery(filepath)
    expected_matrix = build_expected_matrix(gallery) if gallery else None
    
    # Extract server info and check consistency
    hostnames = set()
    gpus = set()
    for rec in records:
        cfg = rec.get('server_config', {})
        hostnames.add(cfg.get('hostname', 'Unknown'))
        gpus.add(cfg.get('gpu_model', 'Unknown'))
    
    # Group by (model_file, solver_type)
    groups = defaultdict(list)
    for rec in records:
        key = (rec['model_file'], rec['solver_type'])
        groups[key].append(rec)
    
    # Count by model_file
    by_model = defaultdict(int)
    for rec in records:
        by_model[rec['model_file']] += 1
    
    # Count by solver_type
    by_solver = defaultdict(int)
    for rec in records:
        by_solver[rec['solver_type']] += 1
    
    # Find duplicates (>3 runs), incomplete (<3 runs), and records to remove
    duplicates = {}
    incomplete = {}
    to_remove = []
    
    for key, recs in groups.items():
        sorted_recs = sorted(recs, key=lambda r: r['timestamp'], reverse=True)
        
        if len(recs) > 3:
            duplicates[key] = {
                'count': len(recs),
                'excess': len(recs) - 3,
                'records': sorted_recs
            }
            # Mark oldest (excess) records for removal
            for rec in sorted_recs[3:]:
                to_remove.append({
                    'id': rec['id'],
                    'timestamp': rec['timestamp'],
                    'model_file': key[0],
                    'solver_type': key[1]
                })
        elif len(recs) < 3:
            incomplete[key] = {
                'count': len(recs),
                'needed': 3 - len(recs)
            }
    
    # Find completely missing configs (0 runs)
    missing = {}
    if expected_matrix:
        present_keys = set(groups.keys())
        for key in expected_matrix:
            if key not in present_keys:
                missing[key] = {'count': 0, 'needed': 3}
    
    # Combine incomplete and missing for command generation
    all_missing = {**incomplete, **missing}
    missing_meshes = sorted(set(key[0] for key in all_missing.keys()))
    
    return {
        'file': str(filepath),
        'filepath': filepath,
        'hostnames': sorted(hostnames),
        'gpus': sorted(gpus),
        'total_records': total,
        'expected': 432,
        'excess': total - 432,
        'unique_configs': len(groups),
        'expected_configs': len(expected_matrix) if expected_matrix else None,
        'by_model_file': dict(sorted(by_model.items())),
        'by_solver_type': dict(sorted(by_solver.items())),
        'duplicates': duplicates,
        'incomplete': incomplete,
        'missing': missing,
        'to_remove': to_remove,
        'missing_meshes': missing_meshes
    }


def print_report(result: dict, show_commands: bool = True) -> None:
    """Print a text report from analysis results."""
    print("=" * 70)
    print(f"BENCHMARK ANALYSIS: {result['file']}")
    print("=" * 70)
    
    # Server info
    if len(result['hostnames']) == 1:
        print(f"Hostname:        {result['hostnames'][0]}")
    else:
        print(f"Hostnames:       {', '.join(result['hostnames'])}  [WARNING: multiple]")
    
    if len(result['gpus']) == 1:
        print(f"GPU:             {result['gpus'][0]}")
    else:
        print(f"GPUs:            {', '.join(result['gpus'])}  [WARNING: multiple]")
    
    print(f"\nTotal records:   {result['total_records']}")
    print(f"Expected:        {result['expected']}")
    print(f"Difference:      {result['excess']:+d}")
    
    expected_str = f" (expected {result['expected_configs']})" if result['expected_configs'] else ""
    print(f"Unique configs:  {result['unique_configs']}{expected_str}")
    
    # By model file
    print("\n" + "-" * 70)
    print("RECORDS BY MODEL FILE")
    print("-" * 70)
    for model, count in result['by_model_file'].items():
        flag = ""
        if count > 18:
            flag = f"  [+{count - 18} excess]"
        elif count < 18:
            flag = f"  [{count - 18} missing]"
        print(f"  {model:30s} {count:3d}{flag}")
    
    # By solver type
    print("\n" + "-" * 70)
    print("RECORDS BY SOLVER TYPE")
    print("-" * 70)
    for solver, count in result['by_solver_type'].items():
        flag = ""
        if count > 72:
            flag = f"  [+{count - 72} excess]"
        elif count < 72:
            flag = f"  [{count - 72} missing]"
        print(f"  {solver:20s} {count:3d}{flag}")
    
    # Duplicates
    print("\n" + "-" * 70)
    print("DUPLICATES (>3 runs per config)")
    print("-" * 70)
    if result['duplicates']:
        for (model, solver), info in sorted(result['duplicates'].items()):
            print(f"\n  {model} | {solver}")
            print(f"  Count: {info['count']} ({info['excess']} excess)")
            print("  Records (newest first):")
            for rec in info['records']:
                print(f"    {rec['timestamp']}  {rec['id']}")
    else:
        print("  None found.")
    
    # Incomplete (1-2 runs)
    print("\n" + "-" * 70)
    print("INCOMPLETE (1-2 runs per config)")
    print("-" * 70)
    if result['incomplete']:
        for (model, solver), info in sorted(result['incomplete'].items()):
            print(f"  {model:30s} | {solver:20s}  has {info['count']}, needs {info['needed']} more")
    else:
        print("  None found.")
    
    # Missing (0 runs)
    print("\n" + "-" * 70)
    print("MISSING (0 runs - config not present)")
    print("-" * 70)
    if result['missing']:
        for (model, solver), info in sorted(result['missing'].items()):
            print(f"  {model:30s} | {solver:20s}")
    else:
        print("  None found.")
    
    # Records to remove
    print("\n" + "-" * 70)
    print("RECORDS TO REMOVE (keeping 3 newest per config)")
    print("-" * 70)
    if result['to_remove']:
        print(f"  Total: {len(result['to_remove'])} records\n")
        for rec in result['to_remove']:
            print(f"  {rec['id']}  {rec['model_file']} | {rec['solver_type']}")
    else:
        print("  None.")
    
    # Commands to run (optional, skipped in --fix mode after actions)
    if show_commands:
        print("\n" + "-" * 70)
        print("COMMANDS TO RUN MISSING TESTS")
        print("-" * 70)
        if result['missing_meshes']:
            for mesh in result['missing_meshes']:
                print(f"  python run_benchmark.py --mesh {mesh} --resume")
        else:
            print("  None needed.")
    
    print("\n" + "=" * 70)


def prompt_yes_no(prompt: str) -> bool:
    """Prompt user for yes/no confirmation."""
    while True:
        response = input(f"{prompt} [y/N]: ").strip().lower()
        if response in ('y', 'yes'):
            return True
        if response in ('n', 'no', ''):
            return False
        print("  Please enter 'y' or 'n'")


def prune_duplicates(filepath: Path, to_remove: list[dict]) -> int:
    """Remove duplicate records from benchmark JSON file.
    
    Creates a backup before modifying.
    
    Args:
        filepath: Path to benchmark JSON file
        to_remove: List of records to remove (must have 'id' key)
    
    Returns:
        Number of records removed
    """
    # Create backup in dedicated folder
    # backup folder: /src/app/server/automated_benchmark/backup/
    backup_dir = filepath.parent.parent / "automated_benchmark" / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{filepath.stem}_{timestamp}.bak"
    backup_path = backup_dir / backup_filename
    shutil.copy2(filepath, backup_path)
    print(f"  Backup created: {backup_path}")
    
    # Load data
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Build set of IDs to remove
    ids_to_remove = {rec['id'] for rec in to_remove}
    
    # Filter records
    original_count = len(data['records'])
    data['records'] = [r for r in data['records'] if r['id'] not in ids_to_remove]
    removed_count = original_count - len(data['records'])
    
    # Update timestamp
    data['updated_at'] = datetime.utcnow().isoformat() + 'Z'
    
    # Write back
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    return removed_count


def run_missing_tests(filepath: Path, missing_meshes: list[str]) -> bool:
    """Run benchmark tests for missing meshes.
    
    Calls run_benchmark.py as subprocess with output streaming.
    
    Args:
        filepath: Path to benchmark JSON file (used to locate run_benchmark.py)
        missing_meshes: List of mesh filenames to test
    
    Returns:
        True if all tests completed successfully, False otherwise
    """
    # Locate run_benchmark.py relative to benchmark file
    # benchmark file is in: /src/app/server/benchmark/
    # run_benchmark.py is in: /src/app/server/automated_benchmark/
    benchmark_dir = filepath.parent
    runner_path = benchmark_dir.parent / "automated_benchmark" / "run_benchmark.py"
    
    if not runner_path.exists():
        print(f"  Error: run_benchmark.py not found at {runner_path}")
        return False
    
    # Build command with all meshes
    cmd = [
        sys.executable,
        str(runner_path),
        "--resume"
    ]
    for mesh in missing_meshes:
        cmd.extend(["--mesh", mesh])
    
    print(f"\n  Executing: {' '.join(cmd)}\n")
    print("-" * 70)
    
    # Run with inherited stdout/stderr for real-time output
    try:
        result = subprocess.run(cmd, cwd=str(benchmark_dir.parent / "automated_benchmark"))
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n\n  Benchmark interrupted by user")
        return False
    except Exception as e:
        print(f"\n  Error running benchmark: {e}")
        return False


def run_fix_mode(filepath: Path) -> None:
    """Run interactive fix mode.
    
    1. Analyze benchmark file
    2. If duplicates: prompt to prune
    3. If missing: prompt to run tests
    4. Re-analyze and report final status
    """
    print("\n" + "=" * 70)
    print(" FIX MODE")
    print("=" * 70)
    
    # Initial analysis
    result = analyze_benchmark_file(filepath)
    print_report(result, show_commands=False)
    
    has_duplicates = len(result['to_remove']) > 0
    has_missing = len(result['missing_meshes']) > 0
    
    if not has_duplicates and not has_missing:
        print("\nNo issues found. Benchmark data is complete.")
        return
    
    # Step 1: Prune duplicates
    if has_duplicates:
        print("\n" + "-" * 70)
        print(f"STEP 1: PRUNE DUPLICATES ({len(result['to_remove'])} records)")
        print("-" * 70)
        print("\nThe following records will be removed (keeping 3 newest per config):\n")
        for rec in result['to_remove']:
            print(f"  {rec['id']}  {rec['model_file']} | {rec['solver_type']}")
        
        if prompt_yes_no("\nProceed with deletion?"):
            removed = prune_duplicates(filepath, result['to_remove'])
            print(f"  Removed {removed} records")
            # Re-analyze after pruning
            result = analyze_benchmark_file(filepath)
            has_missing = len(result['missing_meshes']) > 0
        else:
            print("  Skipped.")
    
    # Step 2: Run missing tests
    if has_missing:
        print("\n" + "-" * 70)
        print(f"STEP 2: RUN MISSING TESTS ({len(result['missing_meshes'])} meshes)")
        print("-" * 70)
        print("\nThe following commands will be executed:\n")
        
        cmd_preview = f"  python run_benchmark.py --resume"
        for mesh in result['missing_meshes']:
            cmd_preview += f" \\\n    --mesh {mesh}"
        print(cmd_preview)
        
        if prompt_yes_no("\nProceed with benchmark execution?"):
            success = run_missing_tests(filepath, result['missing_meshes'])
            if not success:
                print("\n  Benchmark execution failed or was interrupted.")
        else:
            print("  Skipped.")
    
    # Final analysis
    print("\n" + "=" * 70)
    print(" FINAL ANALYSIS")
    print("=" * 70)
    
    result = analyze_benchmark_file(filepath)
    print_report(result, show_commands=True)
    
    # Summary
    has_duplicates = len(result['to_remove']) > 0
    has_missing = len(result['missing_meshes']) > 0
    
    if not has_duplicates and not has_missing:
        print("\nSUCCESS: Benchmark data is now complete.")
    else:
        issues = []
        if has_duplicates:
            issues.append(f"{len(result['to_remove'])} duplicate records")
        if has_missing:
            issues.append(f"{len(result['missing_meshes'])} meshes with missing tests")
        print(f"\nWARNING: Issues remain: {', '.join(issues)}")
        print("  Run with --fix again to resolve.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark JSON files for duplicate/redundant records.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_benchmarks.py benchmark.json          # Analysis only
  python analyze_benchmarks.py benchmark.json --fix    # Interactive fix mode
        """
    )
    
    parser.add_argument(
        "benchmark_file",
        type=str,
        help="Path to benchmark JSON file"
    )
    
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Interactive mode: prune duplicates and run missing tests"
    )
    
    args = parser.parse_args()
    
    filepath = Path(args.benchmark_file)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    if args.fix:
        run_fix_mode(filepath)
    else:
        result = analyze_benchmark_file(filepath)
        print_report(result)


if __name__ == '__main__':
    main()
