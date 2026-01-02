"""
Report Service - Manages project report Markdown files.

This service:
- Loads report section configuration from JSON
- Serves Markdown content for individual sections or all combined
- Saves edited Markdown content back to files
- Auto-creates section files if they don't exist

Location: /src/app/server/report_service.py
Config file: /src/app/server/report/report_sections.json
Data files: /src/app/server/report/*.md
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class ReportService:
    """Service for managing project report Markdown files."""
    
    def __init__(self, report_dir: Path):
        """
        Initialize report service.
        
        Args:
            report_dir: Directory containing report_sections.json and .md files
        """
        self.report_dir = Path(report_dir)
        self.config_file = self.report_dir / "report_sections.json"
        self.config: Dict[str, Any] = {}
        self.sections: List[Dict[str, str]] = []
        self.config_loaded = False
        
        # Ensure directory exists
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config
        self._load_config()
        
        # Ensure all section files exist
        if self.config_loaded:
            self._ensure_section_files()
    
    def _load_config(self):
        """Load report sections configuration from JSON."""
        if not self.config_file.exists():
            print(f"[Report] ERROR: Config file not found: {self.config_file}")
            self.config_loaded = False
            return
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
                self.sections = self.config.get('sections', [])
                self.config_loaded = True
                print(f"[Report] Loaded {len(self.sections)} sections from config")
        except Exception as e:
            print(f"[Report] ERROR: Failed to load config: {e}")
            self.config_loaded = False
    
    def _ensure_section_files(self):
        """Ensure all section .md files exist, create with template if not."""
        for section in self.sections:
            file_path = self.report_dir / section['file']
            if not file_path.exists():
                self._create_section_file(section)
    
    def _create_section_file(self, section: Dict[str, str]):
        """Create a section file with default template."""
        file_path = self.report_dir / section['file']
        title = section['title']
        
        template = f"""# {title}

To be done.
"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(template)
            print(f"[Report] Created section file: {section['file']}")
        except Exception as e:
            print(f"[Report] Error creating section file {section['file']}: {e}")
    
    def get_sections(self) -> Dict[str, Any]:
        """
        Get list of available sections.
        
        Returns:
            Dict with title, sections list, and config status
        """
        if not self.config_loaded:
            return {
                "title": "Project Report",
                "sections": [],
                "error": "Default configuration not found"
            }
        
        return {
            "title": self.config.get('title', 'Project Report'),
            "sections": [
                {
                    "id": s['id'],
                    "title": s['title']
                }
                for s in self.sections
            ]
        }
    
    def get_section_content(self, section_id: str) -> Optional[Dict[str, Any]]:
        """
        Get content of a single section.
        
        Args:
            section_id: Section identifier
            
        Returns:
            Dict with id, title, content, and editable flag, or None if not found
        """
        if not self.config_loaded:
            return None
        
        section = self._find_section(section_id)
        if not section:
            return None
        
        file_path = self.report_dir / section['file']
        content = ""
        
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                content = f"*File not found: {section['file']}*"
        except Exception as e:
            print(f"[Report] Error reading {section['file']}: {e}")
            content = f"*Error loading content: {e}*"
        
        return {
            "id": section['id'],
            "title": section['title'],
            "file": section['file'],
            "content": content,
            "editable": True
        }
    
    def get_all_content(self) -> Dict[str, Any]:
        """
        Get combined content of all sections.
        
        Returns:
            Dict with title, combined content, and section metadata
        """
        if not self.config_loaded:
            return {
                "title": "Project Report",
                "content": "*Default configuration not found*",
                "sections": [],
                "editable": False
            }
        
        combined_content = []
        section_metadata = []
        
        for section in self.sections:
            file_path = self.report_dir / section['file']
            content = ""
            
            try:
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                else:
                    content = f"*File not found: {section['file']}*"
            except Exception as e:
                print(f"[Report] Error reading {section['file']}: {e}")
                content = f"*Error loading {section['title']}: {e}*"
            
            combined_content.append(content)
            section_metadata.append({
                "id": section['id'],
                "title": section['title']
            })
        
        # Join sections with horizontal rule separator
        full_content = "\n\n---\n\n".join(combined_content)
        
        return {
            "title": self.config.get('title', 'Project Report'),
            "content": full_content,
            "sections": section_metadata,
            "editable": False
        }
    
    def save_section_content(self, section_id: str, content: str) -> Dict[str, Any]:
        """
        Save content to a section file.
        
        Args:
            section_id: Section identifier
            content: Markdown content to save
            
        Returns:
            Dict with status and metadata
        """
        if not self.config_loaded:
            return {
                "success": False,
                "error": "Configuration not loaded"
            }
        
        section = self._find_section(section_id)
        if not section:
            return {
                "success": False,
                "error": f"Section '{section_id}' not found"
            }
        
        file_path = self.report_dir / section['file']
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                "success": True,
                "id": section['id'],
                "title": section['title'],
                "file": section['file'],
                "saved_at": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"[Report] Error saving {section['file']}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _find_section(self, section_id: str) -> Optional[Dict[str, str]]:
        """Find section by ID."""
        for section in self.sections:
            if section['id'] == section_id:
                return section
        return None
    
    def refresh(self):
        """Reload configuration from file."""
        self._load_config()
        if self.config_loaded:
            self._ensure_section_files()


def create_report_router(service: ReportService):
    """Create FastAPI router for report API endpoints."""
    from fastapi import APIRouter, HTTPException, Query
    from pydantic import BaseModel
    from typing import Optional
    
    router = APIRouter(prefix="/api/report", tags=["report"])
    
    class SaveContentRequest(BaseModel):
        content: str
    
    @router.get("/sections")
    async def get_sections():
        """Get list of available report sections."""
        return service.get_sections()
    
    @router.get("/content")
    async def get_content(section: Optional[str] = Query(None)):
        """
        Get report content.
        
        Args:
            section: Section ID, or None/empty for all sections combined
        """
        if section and section != "all":
            result = service.get_section_content(section)
            if result is None:
                raise HTTPException(status_code=404, detail=f"Section '{section}' not found")
            return result
        else:
            return service.get_all_content()
    
    @router.put("/content")
    async def save_content(section: str, request: SaveContentRequest):
        """
        Save content to a report section.
        
        Args:
            section: Section ID (required, cannot save to 'all')
        """
        if not section or section == "all":
            raise HTTPException(
                status_code=400, 
                detail="Cannot save to 'all' sections. Select a specific section."
            )
        
        result = service.save_section_content(section, request.content)
        
        if not result.get('success'):
            raise HTTPException(status_code=500, detail=result.get('error', 'Save failed'))
        
        return result
    
    @router.post("/refresh")
    async def refresh():
        """Reload configuration and ensure all files exist."""
        service.refresh()
        return {
            "status": "refreshed", 
            "sections": len(service.sections),
            "config_loaded": service.config_loaded
        }
    
    return router


def init_report_service(report_dir: Path | str) -> ReportService:
    """
    Initialize report service.
    
    Args:
        report_dir: Directory for report files
    
    Returns:
        ReportService instance
    """
    return ReportService(Path(report_dir))
