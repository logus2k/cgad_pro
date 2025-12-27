"""
Pydantic models for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal


class SolverParams(BaseModel):
    """Parameters for starting a FEM solve"""
    mesh_file: str = Field(..., description="Path to mesh file (.h5, .npz, .xlsx)")
    solver_type: Literal["cpu", "gpu", "numba", "numba_cuda", "auto"] = Field("auto", description="Solver implementation")
    max_iterations: int = Field(15000, ge=100, le=100000, description="Maximum solver iterations")
    tolerance: float = Field(1e-8, gt=0, description="Convergence tolerance")
    progress_interval: int = Field(50, ge=1, description="Emit progress every N iterations")
    verbose: bool = Field(True, description="Enable console output")


class JobStatus(BaseModel):
    """Job status response"""
    job_id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    message: Optional[str] = None


class JobResult(BaseModel):
    """Complete job results"""
    job_id: str
    status: str
    converged: bool
    iterations: int
    timing_metrics: dict
    solution_stats: dict
    mesh_info: dict
