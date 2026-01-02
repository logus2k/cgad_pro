"""
Report Service - Manages project report Markdown files.

This service:
- Loads report configuration with documents and sections from JSON
- Serves Markdown content for individual sections or all combined per document
- Saves edited Markdown content back to files
- Auto-creates section files if they don't exist
- Supports multiple documents, each in its own folder

Location: /src/app/server/report_service.py
Config file: /src/app/server/report/report_sections.json
Data files: /src/app/server/report/{folder}/*.md
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
            report_dir: Directory containing report_sections.json and document folders
        """
        self.report_dir = Path(report_dir)
        self.config_file = self.report_dir / "report_sections.json"
        self.config: Dict[str, Any] = {}
        self.documents: List[Dict[str, Any]] = []
        self.config_loaded = False
        
        # Ensure directory exists
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config
        self._load_config()
        
        # Ensure all document folders and section files exist
        if self.config_loaded:
            self._ensure_document_files()
    
    def _load_config(self):
        """Load report configuration from JSON."""
        if not self.config_file.exists():
            print(f"[Report] ERROR: Config file not found: {self.config_file}")
            self.config_loaded = False
            return
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
                self.documents = self.config.get('documents', [])
                self.config_loaded = True
                doc_count = len(self.documents)
                section_count = sum(len(d.get('sections', [])) for d in self.documents)
                print(f"[Report] Loaded {doc_count} documents, {section_count} sections from config")
        except Exception as e:
            print(f"[Report] ERROR: Failed to load config: {e}")
            self.config_loaded = False
    
    def _ensure_document_files(self):
        """Ensure all document folders and section files exist."""
        for doc in self.documents:
            folder = doc.get('folder', '')
            if not folder:
                continue
            
            doc_dir = self.report_dir / folder
            doc_dir.mkdir(parents=True, exist_ok=True)
            
            for section in doc.get('sections', []):
                file_path = doc_dir / section['file']
                if not file_path.exists():
                    self._create_section_file(file_path, section['title'])
    
    def _create_section_file(self, file_path: Path, title: str):
        """Create a section file with default template."""
        template = f"""# {title}

To be done.
"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(template)
            print(f"[Report] Created section file: {file_path.name}")
        except Exception as e:
            print(f"[Report] Error creating section file {file_path}: {e}")
    
    def get_documents(self) -> Dict[str, Any]:
        """
        Get list of available documents with their sections.
        
        Returns:
            Dict with title, documents list, and config status
        """
        if not self.config_loaded:
            return {
                "title": "Project Reports",
                "documents": [],
                "error": "Default configuration not found"
            }
        
        return {
            "title": self.config.get('title', 'Project Reports'),
            "documents": [
                {
                    "id": doc['id'],
                    "title": doc['title'],
                    "sections": [
                        {
                            "id": s['id'],
                            "title": s['title']
                        }
                        for s in doc.get('sections', [])
                    ]
                }
                for doc in self.documents
            ]
        }
    
    def get_section_content(self, document_id: str, section_id: str) -> Optional[Dict[str, Any]]:
        """
        Get content of a single section.
        
        Args:
            document_id: Document identifier
            section_id: Section identifier
            
        Returns:
            Dict with content and metadata, or None if not found
        """
        if not self.config_loaded:
            return None
        
        doc = self._find_document(document_id)
        if not doc:
            return None
        
        section = self._find_section(doc, section_id)
        if not section:
            return None
        
        folder = doc.get('folder', '')
        file_path = self.report_dir / folder / section['file']
        content = ""
        
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                content = f"*File not found: {section['file']}*"
        except Exception as e:
            print(f"[Report] Error reading {file_path}: {e}")
            content = f"*Error loading content: {e}*"
        
        return {
            "document_id": document_id,
            "document_title": doc['title'],
            "section_id": section['id'],
            "section_title": section['title'],
            "file": section['file'],
            "content": content,
            "editable": True
        }
    
    def get_document_content(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get combined content of all sections in a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Dict with combined content and metadata
        """
        if not self.config_loaded:
            return None
        
        doc = self._find_document(document_id)
        if not doc:
            return None
        
        folder = doc.get('folder', '')
        combined_content = []
        section_metadata = []
        
        for section in doc.get('sections', []):
            file_path = self.report_dir / folder / section['file']
            content = ""
            
            try:
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                else:
                    content = f"*File not found: {section['file']}*"
            except Exception as e:
                print(f"[Report] Error reading {file_path}: {e}")
                content = f"*Error loading {section['title']}: {e}*"
            
            combined_content.append(content)
            section_metadata.append({
                "id": section['id'],
                "title": section['title']
            })
        
        # Join sections with horizontal rule separator
        full_content = "\n\n---\n\n".join(combined_content)
        
        return {
            "document_id": document_id,
            "document_title": doc['title'],
            "content": full_content,
            "sections": section_metadata,
            "editable": False  # Combined view is read-only
        }
    
    def save_section_content(self, document_id: str, section_id: str, content: str) -> Dict[str, Any]:
        """
        Save content to a section file.
        
        Args:
            document_id: Document identifier
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
        
        doc = self._find_document(document_id)
        if not doc:
            return {
                "success": False,
                "error": f"Document '{document_id}' not found"
            }
        
        section = self._find_section(doc, section_id)
        if not section:
            return {
                "success": False,
                "error": f"Section '{section_id}' not found in document '{document_id}'"
            }
        
        folder = doc.get('folder', '')
        file_path = self.report_dir / folder / section['file']
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                "success": True,
                "document_id": document_id,
                "section_id": section['id'],
                "section_title": section['title'],
                "file": section['file'],
                "saved_at": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"[Report] Error saving {file_path}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _find_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Find document by ID."""
        for doc in self.documents:
            if doc['id'] == document_id:
                return doc
        return None
    
    def _find_section(self, doc: Dict[str, Any], section_id: str) -> Optional[Dict[str, str]]:
        """Find section by ID within a document."""
        for section in doc.get('sections', []):
            if section['id'] == section_id:
                return section
        return None
    
    def refresh(self):
        """Reload configuration from file."""
        self._load_config()
        if self.config_loaded:
            self._ensure_document_files()


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
        """Get list of available documents and sections."""
        return service.get_documents()
    
    @router.get("/content")
    async def get_content(
        document: str = Query(..., description="Document ID"),
        section: Optional[str] = Query(None, description="Section ID (omit for all sections)")
    ):
        """
        Get report content.
        
        Args:
            document: Document ID (required)
            section: Section ID, or None for all sections combined
        """
        if section and section != "all":
            result = service.get_section_content(document, section)
            if result is None:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Section '{section}' not found in document '{document}'"
                )
            return result
        else:
            result = service.get_document_content(document)
            if result is None:
                raise HTTPException(status_code=404, detail=f"Document '{document}' not found")
            return result
    
    @router.put("/content")
    async def save_content(
        document: str = Query(..., description="Document ID"),
        section: str = Query(..., description="Section ID"),
        request: SaveContentRequest = ...
    ):
        """
        Save content to a report section.
        
        Args:
            document: Document ID (required)
            section: Section ID (required, cannot save to 'all')
        """
        if not section or section == "all":
            raise HTTPException(
                status_code=400, 
                detail="Cannot save to 'all' sections. Select a specific section."
            )
        
        result = service.save_section_content(document, section, request.content)
        
        if not result.get('success'):
            raise HTTPException(status_code=500, detail=result.get('error', 'Save failed'))
        
        return result
    
    @router.post("/refresh")
    async def refresh():
        """Reload configuration and ensure all files exist."""
        service.refresh()
        return {
            "status": "refreshed", 
            "documents": len(service.documents),
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
