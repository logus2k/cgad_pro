"""
Pydantic models for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, Any


class ClientConfig(BaseModel):
    """Client hardware configuration for benchmark tracking"""
    browser: Optional[str] = None
    browser_version: Optional[str] = None
    os: Optional[str] = None
    os_version: Optional[str] = None
    device_type: Optional[str] = None
    screen_width: Optional[int] = None
    screen_height: Optional[int] = None
    device_pixel_ratio: Optional[float] = None
    gpu_vendor: Optional[str] = None
    gpu_renderer: Optional[str] = None
    webgl_version: Optional[str] = None
    hash: Optional[str] = None


class SolverParams(BaseModel):
    """Parameters for starting a FEM solve"""
    mesh_file: str = Field(..., description="Path to mesh file (.h5, .npz, .xlsx)")
    solver_type: Literal["cpu", "gpu", "numba", "numba_cuda", "cpu_threaded", "cpu_multiprocess", "auto"] = Field("auto", description="Solver implementation")
    max_iterations: int = Field(15000, ge=100, le=100000, description="Maximum solver iterations")
    tolerance: float = Field(1e-8, gt=0, description="Convergence tolerance")
    progress_interval: int = Field(50, ge=1, description="Emit progress every N iterations")
    verbose: bool = Field(True, description="Enable console output")
    client_config: Optional[ClientConfig] = Field(None, description="Client hardware configuration for benchmarking")


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
