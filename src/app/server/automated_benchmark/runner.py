"""
Automated Benchmark Runner for FEMulator Pro.

Orchestrates the execution of benchmark tests across all solver/model/mesh
combinations with progress reporting, time estimation, and result recording.
"""

import sys
import time
import signal
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import timedelta

from config_loader import ConfigLoader, TestCase, ExecutionConfig, SolverParams
from result_recorder import ResultRecorder


@dataclass
class RunResult:
    """Result of a single benchmark run."""
    success: bool
    converged: bool
    iterations: int
    duration: float
    timings: Dict[str, float]
    solution_stats: Dict[str, Any]
    memory: Dict[str, Any]
    mesh_info: Dict[str, Any]
    solver_config: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class TestResult:
    """Aggregated result for a test case (multiple runs)."""
    test_case: TestCase
    warmup_results: List[RunResult]
    run_results: List[RunResult]
    mean_duration: float
    std_duration: float
    all_converged: bool


class ProgressTracker:
    """Tracks progress and provides time estimations."""
    
    def __init__(self, total_tests: int, runs_per_test: int, warmup_runs: int):
        self.total_tests = total_tests
        self.runs_per_test = runs_per_test
        self.warmup_runs = warmup_runs
        self.total_runs = total_tests * (runs_per_test + warmup_runs)
        
        self.completed_tests = 0
        self.completed_runs = 0
        self.start_time = time.time()
        
        self.model_times: Dict[str, List[float]] = {}
        self.current_model: Optional[str] = None
        self.current_model_start: float = 0
        
        self.first_model_complete = False
        self.estimated_total_time: Optional[float] = None
    
    def start_test(self, test_case: TestCase) -> None:
        """Mark start of a test case."""
        if test_case.model_name != self.current_model:
            if self.current_model and self.current_model_start > 0:
                model_duration = time.time() - self.current_model_start
                if self.current_model not in self.model_times:
                    self.model_times[self.current_model] = []
                self.model_times[self.current_model].append(model_duration)
            
            self.current_model = test_case.model_name
            self.current_model_start = time.time()
    
    def complete_run(self) -> None:
        """Mark completion of a single run."""
        self.completed_runs += 1
    
    def complete_test(self, test_case: TestCase, duration: float) -> None:
        """Mark completion of a test case."""
        self.completed_tests += 1
    
    def complete_model(self, model_name: str) -> None:
        """Mark completion of all tests for a model."""
        if self.current_model == model_name and self.current_model_start > 0:
            model_duration = time.time() - self.current_model_start
            if model_name not in self.model_times:
                self.model_times[model_name] = []
            self.model_times[model_name].append(model_duration)
        
        if not self.first_model_complete and len(self.model_times) > 0:
            self.first_model_complete = True
            self._update_estimate()
    
    def _update_estimate(self) -> None:
        """Update total time estimate."""
        if not self.model_times:
            return
        
        elapsed = time.time() - self.start_time
        if self.completed_tests > 0:
            avg_per_test = elapsed / self.completed_tests
            self.estimated_total_time = avg_per_test * self.total_tests
    
    def get_elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    def get_eta(self) -> Optional[float]:
        """Get estimated time remaining in seconds."""
        if self.completed_tests == 0:
            return None
        
        elapsed = self.get_elapsed()
        avg_per_test = elapsed / self.completed_tests
        remaining_tests = self.total_tests - self.completed_tests
        return avg_per_test * remaining_tests
    
    def get_progress_percent(self) -> float:
        """Get completion percentage."""
        if self.total_tests == 0:
            return 100.0
        return 100.0 * self.completed_tests / self.total_tests
    
    def format_time(self, seconds: Optional[float]) -> str:
        """Format seconds as human-readable string."""
        if seconds is None:
            return "calculating..."
        
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return str(timedelta(seconds=int(seconds)))[2:]
        else:
            return str(timedelta(seconds=int(seconds)))


class BenchmarkRunner:
    """Main benchmark orchestrator."""
    
    def __init__(
        self,
        config: ConfigLoader,
        recorder: ResultRecorder,
        mesh_base_dir: Path
    ):
        """
        Initialize benchmark runner.
        
        Args:
            config: Configuration loader with test matrix
            recorder: Result recorder for saving results
            mesh_base_dir: Base directory for mesh files
        """
        self.config = config
        self.recorder = recorder
        self.mesh_base_dir = Path(mesh_base_dir)
        
        self._solver_wrapper_class = None
        self._abort_requested = False
        self._force_exit_requested = False
        
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully, with double-press for immediate exit."""
        if self._abort_requested:
            # Second Ctrl+C - force immediate exit
            print("\n\n[!] Force exit requested. Terminating immediately...")
            self._force_exit_requested = True
            import os
            os._exit(130)  # Forceful exit, bypasses all handlers
        else:
            # First Ctrl+C - graceful stop after current test
            print("\n\n[!] Interrupt received. Will stop after current test completes.")
            print("    Press Ctrl+C again to force immediate exit.")
            self._abort_requested = True
    
    def _get_solver_wrapper(self):
        """Lazy import of SolverWrapper."""
        if self._solver_wrapper_class is None:
            from solver_wrapper import SolverWrapper
            self._solver_wrapper_class = SolverWrapper
        return self._solver_wrapper_class
    
    def _resolve_mesh_path(self, mesh_file: str) -> Path:
        """Resolve mesh file path from gallery format to absolute path."""
        if mesh_file.startswith('./'):
            mesh_file = mesh_file[2:]
        
        return self.mesh_base_dir / mesh_file
    
    def _run_single(
        self,
        test_case: TestCase,
        is_warmup: bool = False
    ) -> RunResult:
        """Execute a single benchmark run."""
        SolverWrapper = self._get_solver_wrapper()
        
        mesh_path = self._resolve_mesh_path(test_case.mesh_file)
        
        if not mesh_path.exists():
            return RunResult(
                success=False,
                converged=False,
                iterations=0,
                duration=0,
                timings={},
                solution_stats={},
                memory={},
                mesh_info={},
                solver_config={},
                error=f"Mesh file not found: {mesh_path}"
            )
        
        params = {
            'mesh_file': str(mesh_path),
            'solver_type': test_case.solver_id,
            'max_iterations': self.config.solver_params.max_iterations,
            'progress_interval': self.config.solver_params.progress_interval,
            'verbose': self.config.solver_params.verbose
        }
        
        start_time = time.time()
        
        try:
            wrapper = SolverWrapper(
                solver_type=test_case.solver_id,
                params=params,
                progress_callback=None
            )
            
            results = wrapper.run()
            duration = time.time() - start_time
            
            return RunResult(
                success=True,
                converged=results.get('converged', False),
                iterations=results.get('iterations', 0),
                duration=duration,
                timings=results.get('timing_metrics', {}),
                solution_stats=results.get('solution_stats', {}),
                memory=results.get('memory', {}),
                mesh_info=results.get('mesh_info', {}),
                solver_config=results.get('solver_config', {})
            )
            
        except Exception as e:
            duration = time.time() - start_time
            import traceback
            traceback.print_exc()
            return RunResult(
                success=False,
                converged=False,
                iterations=0,
                duration=duration,
                timings={},
                solution_stats={},
                memory={},
                mesh_info={},
                solver_config={},
                error=str(e)
            )
    
    def _run_test_case(
        self,
        test_case: TestCase,
        progress: ProgressTracker,
        test_number: int
    ) -> Tuple[Optional[TestResult], Optional[str]]:
        """Execute all runs for a test case."""
        warmup_results = []
        run_results = []
        
        total_width = len(str(progress.total_tests))
        
        # Header
        print(f"\n[{test_number:>{total_width}}/{progress.total_tests}] "
              f"{test_case.solver_name} + {test_case.model_name} "
              f"({test_case.mesh_nodes:,} nodes)")
        
        # Warmup runs
        for w in range(self.config.execution.warmup_runs):
            if self._abort_requested:
                return None, "Aborted by user"
            
            result = self._run_single(test_case, is_warmup=True)
            warmup_results.append(result)
            progress.complete_run()
            
            if not result.success:
                status = f"FAILED - {result.error}"
                print(f"          Warmup {w+1}/{self.config.execution.warmup_runs}: {status}")
                if self.config.execution.abort_on_failure:
                    return None, f"Warmup failed: {result.error}"
            else:
                status = f"{result.duration:.2f}s"
                if not result.converged:
                    status += " (not converged)"
                print(f"          Warmup {w+1}/{self.config.execution.warmup_runs}: {status} OK")
        
        # Actual runs
        for r in range(self.config.execution.runs_per_test):
            if self._abort_requested:
                return None, "Aborted by user"
            
            result = self._run_single(test_case)
            run_results.append(result)
            progress.complete_run()
            
            if not result.success:
                status = f"FAILED - {result.error}"
                print(f"          Run {r+1}/{self.config.execution.runs_per_test}: {status}")
                if self.config.execution.abort_on_failure:
                    return None, f"Run {r+1} failed: {result.error}"
            else:
                conv_str = f"{result.iterations} iter" if result.converged else "NOT CONVERGED"
                print(f"          Run {r+1}/{self.config.execution.runs_per_test}: "
                      f"{result.duration:.2f}s ({conv_str}) OK")
                
                if not result.converged and self.config.execution.abort_on_failure:
                    return None, f"Solver did not converge after {result.iterations} iterations"
        
        # Calculate statistics
        successful_runs = [r for r in run_results if r.success]
        if successful_runs:
            durations = [r.duration for r in successful_runs]
            mean_duration = sum(durations) / len(durations)
            
            if len(durations) > 1:
                variance = sum((d - mean_duration) ** 2 for d in durations) / (len(durations) - 1)
                std_duration = variance ** 0.5
            else:
                std_duration = 0.0
            
            all_converged = all(r.converged for r in successful_runs)
            
            last_result = successful_runs[-1]
            peak_ram = last_result.memory.get('peak_ram_mb', 0)
            peak_vram = last_result.memory.get('peak_vram_mb', 0)
            
            mem_str = f"RAM: {peak_ram:.0f}MB"
            if peak_vram > 0:
                mem_str += f", VRAM: {peak_vram:.0f}MB"
            
            print(f"          -> Mean: {mean_duration:.2f}s +/- {std_duration:.2f}s | {mem_str}")
        else:
            mean_duration = 0.0
            std_duration = 0.0
            all_converged = False
        
        test_result = TestResult(
            test_case=test_case,
            warmup_results=warmup_results,
            run_results=run_results,
            mean_duration=mean_duration,
            std_duration=std_duration,
            all_converged=all_converged
        )
        
        return test_result, None
    
    def run(
        self,
        solver_filter: Optional[str] = None,
        model_filter: Optional[str] = None,
        max_nodes: Optional[int] = None,
        mesh_filter: Optional[List[str]] = None,
        dry_run: bool = False,
        resume: bool = False
    ) -> Tuple[List[TestResult], Optional[str]]:
        """Execute the benchmark suite."""
        test_cases = self.config.generate_test_matrix(
            solver_filter=solver_filter,
            model_filter=model_filter,
            max_nodes_override=max_nodes
        )
        
        # Filter by mesh file if specified
        if mesh_filter:
            mesh_names = set(mesh_filter)
            test_cases = [
                tc for tc in test_cases
                if Path(tc.mesh_file).name in mesh_names
            ]
        
        if not test_cases:
            return [], "No test cases match the specified filters"
        
        # Filter out already-completed tests if resuming
        if resume:
            original_count = len(test_cases)
            test_cases = [
                tc for tc in test_cases
                if not self.recorder.has_result(tc.solver_id, tc.model_name, tc.mesh_nodes)
            ]
            skipped = original_count - len(test_cases)
            if skipped > 0:
                print(f"\n[Resume] Skipping {skipped} already-completed tests")
            if not test_cases:
                print("[Resume] All tests already completed!")
                return [], None
        
        self._print_header(test_cases, dry_run)
        
        if dry_run:
            self._print_test_matrix(test_cases)
            return [], None
        
        progress = ProgressTracker(
            total_tests=len(test_cases),
            runs_per_test=self.config.execution.runs_per_test,
            warmup_runs=self.config.execution.warmup_runs
        )
        
        results: List[TestResult] = []
        current_model = None
        
        for i, test_case in enumerate(test_cases, 1):
            if self._abort_requested:
                break
            
            if test_case.model_name != current_model:
                if current_model:
                    progress.complete_model(current_model)
                    self._print_model_complete(current_model, progress)
                current_model = test_case.model_name
            
            progress.start_test(test_case)
            
            test_result, error = self._run_test_case(test_case, progress, i)
            
            if error or test_result is None:
                self._print_abort(test_case, error or "Unknown error", results)
                return results, error
            
            results.append(test_result)
            progress.complete_test(test_case, test_result.mean_duration)
            
            # Record results atomically - only if warmup AND all runs succeeded
            successful_warmups = [r for r in test_result.warmup_results if r.success]
            successful_runs = [r for r in test_result.run_results if r.success]
            
            warmup_ok = len(successful_warmups) == self.config.execution.warmup_runs
            runs_ok = len(successful_runs) == self.config.execution.runs_per_test
            
            if warmup_ok and runs_ok:
                for run_result in successful_runs:
                    self.recorder.add_record(
                        model_file=Path(test_case.mesh_file).name,
                        model_name=test_case.model_name,
                        model_nodes=test_case.mesh_nodes,
                        model_elements=test_case.mesh_elements,
                        solver_type=test_case.solver_id,
                        converged=run_result.converged,
                        iterations=run_result.iterations,
                        timings=run_result.timings,
                        solution_stats=run_result.solution_stats,
                        memory=run_result.memory,
                        mesh_info=run_result.mesh_info,
                        solver_config=run_result.solver_config
                    )
            else:
                if not warmup_ok:
                    print(f"          [!] Warmup incomplete ({len(successful_warmups)}/"
                          f"{self.config.execution.warmup_runs}) - not recording results")
                else:
                    print(f"          [!] Only {len(successful_runs)}/{self.config.execution.runs_per_test} "
                          f"runs succeeded - not recording results")
        
        if current_model:
            progress.complete_model(current_model)
        
        self._print_summary(results, progress)
        
        return results, None
    
    def _print_header(self, test_cases: List[TestCase], dry_run: bool) -> None:
        """Print benchmark header."""
        line = "=" * 79
        print(f"\n{line}")
        print(" FEMulator Pro - Automated Benchmark Suite")
        print(line)
        print(f" Server: {self.recorder.get_server_display_name()} ({self.recorder.server_hash})")
        
        cpu = self.recorder.server_config.get('cpu_model', 'Unknown')
        cores = self.recorder.server_config.get('cpu_cores', 0)
        print(f" CPU: {cpu} ({cores} cores)")
        
        gpu = self.recorder.server_config.get('gpu_model')
        if gpu:
            gpu_mem = self.recorder.server_config.get('gpu_memory_gb', 0)
            print(f" GPU: {gpu} ({gpu_mem} GB)")
        
        print()
        print(f" Gallery: {self.config.gallery_file.name}")
        
        solvers = len(set(tc.solver_id for tc in test_cases))
        models = len(set(tc.model_name for tc in test_cases))
        meshes = len(test_cases)
        
        print(f" Solvers: {solvers} | Models: {models} | Tests: {meshes}")
        print(f" Runs per test: {self.config.execution.runs_per_test} "
              f"(+ {self.config.execution.warmup_runs} warmup)")
        
        if dry_run:
            print(f"\n [DRY RUN - No tests will be executed]")
        
        print(line)
    
    def _print_test_matrix(self, test_cases: List[TestCase]) -> None:
        """Print test matrix for dry run."""
        print("\nTest Matrix:")
        print("-" * 79)
        
        current_model = None
        for i, tc in enumerate(test_cases, 1):
            if tc.model_name != current_model:
                if current_model:
                    print()
                print(f"\n  {tc.model_name}:")
                current_model = tc.model_name
            
            print(f"    [{i:>3}] {tc.solver_name:<25} | {tc.mesh_nodes:>10,} nodes")
        
        print("\n" + "-" * 79)
        print(f"Total: {len(test_cases)} test cases")
    
    def _print_model_complete(self, model_name: str, progress: ProgressTracker) -> None:
        """Print status after completing a model."""
        elapsed = progress.format_time(progress.get_elapsed())
        eta = progress.format_time(progress.get_eta())
        pct = progress.get_progress_percent()
        
        print(f"\n    [{model_name} complete] "
              f"Progress: {pct:.1f}% | Elapsed: {elapsed} | ETA: {eta}")
    
    def _print_abort(
        self,
        test_case: TestCase,
        error: str,
        partial_results: List[TestResult]
    ) -> None:
        """Print abort message."""
        line = "=" * 79
        print(f"\n{line}")
        print(" BENCHMARK ABORTED")
        print(line)
        print(f" Failed test: {test_case.display_name}")
        print(f" Error: {error}")
        print()
        print(f" Partial results saved: {self.recorder.data_file.name}")
        print(f" Completed tests: {len(partial_results)}")
        print(line)
    
    def _print_summary(
        self,
        results: List[TestResult],
        progress: ProgressTracker
    ) -> None:
        """Print final summary."""
        line = "=" * 79
        elapsed = progress.format_time(progress.get_elapsed())
        
        print(f"\n{line}")
        print(" BENCHMARK COMPLETE")
        print(line)
        print(f" Total time: {elapsed}")
        print(f" Tests completed: {len(results)}")
        print(f" Records saved: {self.recorder.get_record_count()}")
        print(f" Output file: {self.recorder.data_file.name}")
        print(line)
