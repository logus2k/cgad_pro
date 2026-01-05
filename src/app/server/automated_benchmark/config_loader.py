"""
Configuration Loader for Automated Benchmark Suite.

Loads and validates the testing procedure configuration and gallery files.
Generates the complete test matrix based on filters.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class SolverConfig:
    """Solver configuration from gallery."""
    id: str
    name: str
    description: str


@dataclass
class MeshConfig:
    """Mesh configuration from gallery."""
    label: str
    file: str
    nodes: int
    elements: int
    default: bool = False


@dataclass
class ModelConfig:
    """Model configuration from gallery."""
    name: str
    description: str
    problem_type: str
    thumbnail: str
    meshes: List[MeshConfig] = field(default_factory=list)


@dataclass
class TestCase:
    """Single test case combining solver, model, and mesh."""
    solver_id: str
    solver_name: str
    model_name: str
    mesh_label: str
    mesh_file: str
    mesh_nodes: int
    mesh_elements: int
    
    @property
    def test_id(self) -> str:
        """Unique identifier for this test case."""
        return f"{self.solver_id}_{self.model_name}_{self.mesh_nodes}"
    
    @property
    def display_name(self) -> str:
        """Human-readable test name."""
        return f"{self.solver_name} + {self.model_name} ({self.mesh_nodes:,} nodes)"


@dataclass
class ExecutionConfig:
    """Execution parameters."""
    runs_per_test: int = 3
    warmup_runs: int = 1
    timeout_seconds: int = 600
    abort_on_failure: bool = True
    verbose: bool = True


@dataclass
class SolverParams:
    """Solver parameters."""
    max_iterations: int = 15000
    progress_interval: int = 100
    verbose: bool = False


@dataclass
class Filters:
    """Test filtering configuration."""
    solvers_enabled: List[str] = field(default_factory=list)
    solvers_disabled: List[str] = field(default_factory=list)
    models_enabled: List[str] = field(default_factory=list)
    models_disabled: List[str] = field(default_factory=list)
    min_nodes: int = 0
    max_nodes: Optional[int] = None


class ConfigLoader:
    """
    Loads and processes benchmark configuration.
    
    Combines testing_procedure.json with gallery_files.json to generate
    the complete test matrix with appropriate filtering.
    """
    
    def __init__(self, config_file: Path, gallery_override: Optional[Path] = None):
        """
        Initialize config loader.
        
        Args:
            config_file: Path to testing_procedure.json
            gallery_override: Optional override for gallery file path
        """
        self.config_file = Path(config_file)
        self.config_dir = self.config_file.parent
        
        # Load testing procedure
        with open(self.config_file, 'r') as f:
            self.raw_config = json.load(f)
        
        # Determine gallery file path
        if gallery_override:
            self.gallery_file = Path(gallery_override)
        else:
            gallery_path = self.raw_config.get('gallery_file', '../../client/config/gallery_files.json')
            self.gallery_file = (self.config_dir / gallery_path).resolve()
        
        if not self.gallery_file.exists():
            raise FileNotFoundError(f"Gallery file not found: {self.gallery_file}")
        
        # Load gallery
        with open(self.gallery_file, 'r') as f:
            self.gallery = json.load(f)
        
        # Parse configurations
        self.execution = self._parse_execution()
        self.solver_params = self._parse_solver_params()
        self.filters = self._parse_filters()
        self.solvers = self._parse_solvers()
        self.models = self._parse_models()
    
    def _parse_execution(self) -> ExecutionConfig:
        """Parse execution configuration."""
        exec_config = self.raw_config.get('execution', {})
        return ExecutionConfig(
            runs_per_test=exec_config.get('runs_per_test', 3),
            warmup_runs=exec_config.get('warmup_runs', 1),
            timeout_seconds=exec_config.get('timeout_seconds', 600),
            abort_on_failure=exec_config.get('abort_on_failure', True),
            verbose=exec_config.get('verbose', True)
        )
    
    def _parse_solver_params(self) -> SolverParams:
        """Parse solver parameters."""
        params = self.raw_config.get('solver_params', {})
        return SolverParams(
            max_iterations=params.get('max_iterations', 15000),
            progress_interval=params.get('progress_interval', 100),
            verbose=params.get('verbose', False)
        )
    
    def _parse_filters(self) -> Filters:
        """Parse filter configuration."""
        filters = self.raw_config.get('filters', {})
        solvers = filters.get('solvers', {})
        models = filters.get('models', {})
        mesh_sizes = filters.get('mesh_sizes', {})
        
        return Filters(
            solvers_enabled=solvers.get('enabled', []),
            solvers_disabled=solvers.get('disabled', []),
            models_enabled=models.get('enabled', []),
            models_disabled=models.get('disabled', []),
            min_nodes=mesh_sizes.get('min_nodes', 0),
            max_nodes=mesh_sizes.get('max_nodes')
        )
    
    def _parse_solvers(self) -> List[SolverConfig]:
        """Parse solver definitions from gallery."""
        solvers = []
        for s in self.gallery.get('solvers', []):
            solvers.append(SolverConfig(
                id=s['id'],
                name=s['name'],
                description=s.get('description', '')
            ))
        return solvers
    
    def _parse_models(self) -> List[ModelConfig]:
        """Parse model definitions from gallery."""
        models = []
        for m in self.gallery.get('models', []):
            meshes = []
            for mesh in m.get('meshes', []):
                meshes.append(MeshConfig(
                    label=mesh['label'],
                    file=mesh['file'],
                    nodes=mesh['nodes'],
                    elements=mesh['elements'],
                    default=mesh.get('default', False)
                ))
            
            models.append(ModelConfig(
                name=m['name'],
                description=m.get('description', ''),
                problem_type=m.get('problem_type', ''),
                thumbnail=m.get('thumbnail', ''),
                meshes=meshes
            ))
        return models
    
    def _is_solver_enabled(self, solver_id: str) -> bool:
        """Check if solver passes filters."""
        if self.filters.solvers_enabled:
            if solver_id not in self.filters.solvers_enabled:
                return False
        
        if solver_id in self.filters.solvers_disabled:
            return False
        
        return True
    
    def _is_model_enabled(self, model_name: str) -> bool:
        """Check if model passes filters."""
        if self.filters.models_enabled:
            if model_name not in self.filters.models_enabled:
                return False
        
        if model_name in self.filters.models_disabled:
            return False
        
        return True
    
    def _is_mesh_enabled(self, mesh: MeshConfig) -> bool:
        """Check if mesh passes size filters."""
        if mesh.nodes < self.filters.min_nodes:
            return False
        
        if self.filters.max_nodes is not None and mesh.nodes > self.filters.max_nodes:
            return False
        
        return True
    
    def generate_test_matrix(
        self,
        solver_filter: Optional[str] = None,
        model_filter: Optional[str] = None,
        max_nodes_override: Optional[int] = None
    ) -> List[TestCase]:
        """
        Generate the complete test matrix based on configuration and filters.
        
        Args:
            solver_filter: CLI override to run only specific solver
            model_filter: CLI override to run only specific model
            max_nodes_override: CLI override for maximum mesh size
        
        Returns:
            List of TestCase objects representing all tests to run
        """
        test_cases = []
        
        effective_max_nodes = max_nodes_override or self.filters.max_nodes
        
        for solver in self.solvers:
            if solver_filter and solver.id != solver_filter:
                continue
            if not self._is_solver_enabled(solver.id):
                continue
            
            for model in self.models:
                if model_filter and model.name != model_filter:
                    continue
                if not self._is_model_enabled(model.name):
                    continue
                
                for mesh in model.meshes:
                    if mesh.nodes < self.filters.min_nodes:
                        continue
                    if effective_max_nodes and mesh.nodes > effective_max_nodes:
                        continue
                    
                    test_cases.append(TestCase(
                        solver_id=solver.id,
                        solver_name=solver.name,
                        model_name=model.name,
                        mesh_label=mesh.label,
                        mesh_file=mesh.file,
                        mesh_nodes=mesh.nodes,
                        mesh_elements=mesh.elements
                    ))
        
        return test_cases
    
    def get_model_name_for_file(self, mesh_filename: str) -> str:
        """Get display name for a mesh file."""
        filename = Path(mesh_filename).name
        
        for model in self.models:
            for mesh in model.meshes:
                if Path(mesh.file).name == filename:
                    return model.name
        
        return filename
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary for display."""
        test_matrix = self.generate_test_matrix()
        
        return {
            'gallery_file': str(self.gallery_file),
            'solvers_total': len(self.solvers),
            'solvers_enabled': len([s for s in self.solvers if self._is_solver_enabled(s.id)]),
            'models_total': len(self.models),
            'models_enabled': len([m for m in self.models if self._is_model_enabled(m.name)]),
            'meshes_total': sum(len(m.meshes) for m in self.models),
            'test_cases': len(test_matrix),
            'runs_per_test': self.execution.runs_per_test,
            'warmup_runs': self.execution.warmup_runs,
            'total_runs': len(test_matrix) * (self.execution.runs_per_test + self.execution.warmup_runs)
        }
