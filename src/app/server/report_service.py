"""
Report Service - Manages project report Markdown files.

Simplified: each document is a single .md file.

Location: /src/app/server/report_service.py
Config file: /src/app/server/report/report_sections.json
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class ReportService:
    """Service for managing project report Markdown files."""
    
    def __init__(self, report_dir: Path):
        self.report_dir = Path(report_dir)
        self.config_file = self.report_dir / "report_sections.json"
        self.config: Dict[str, Any] = {}
        self.documents: List[Dict[str, Any]] = []
        self.config_loaded = False
        
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._load_config()
        
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
                print(f"[Report] Loaded {len(self.documents)} documents from config")
        except Exception as e:
            print(f"[Report] ERROR: Failed to load config: {e}")
            self.config_loaded = False
    
    def _ensure_document_files(self):
        """Ensure all document files and folders exist."""
        for doc in self.documents:
            file_path = self.report_dir / doc['file']
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if not file_path.exists():
                self._create_document_file(file_path, doc['title'])
    
    def _create_document_file(self, file_path: Path, title: str):
        """Create a document file with default template."""
        template = f"""# {title}

To be done.
"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(template)
            print(f"[Report] Created document file: {file_path}")
        except Exception as e:
            print(f"[Report] Error creating document file {file_path}: {e}")
    
    def get_documents(self) -> Dict[str, Any]:
        """Get list of available documents."""
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
                    "title": doc['title']
                }
                for doc in self.documents
            ]
        }
    
    def get_document_content(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get content of a document."""
        if not self.config_loaded:
            return None
        
        doc = self._find_document(document_id)
        if not doc:
            return None
        
        file_path = self.report_dir / doc['file']
        content = ""
        
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                content = f"*File not found: {doc['file']}*"
        except Exception as e:
            print(f"[Report] Error reading {file_path}: {e}")
            content = f"*Error loading content: {e}*"
        
        return {
            "document_id": document_id,
            "title": doc['title'],
            "file": doc['file'],
            "content": content
        }
    
    def save_document_content(self, document_id: str, content: str) -> Dict[str, Any]:
        """Save content to a document file."""
        if not self.config_loaded:
            return {"success": False, "error": "Configuration not loaded"}
        
        doc = self._find_document(document_id)
        if not doc:
            return {"success": False, "error": f"Document '{document_id}' not found"}
        
        file_path = self.report_dir / doc['file']
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                "success": True,
                "document_id": document_id,
                "title": doc['title'],
                "saved_at": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"[Report] Error saving {file_path}: {e}")
            return {"success": False, "error": str(e)}
    
    def _find_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Find document by ID."""
        for doc in self.documents:
            if doc['id'] == document_id:
                return doc
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
    
    router = APIRouter(prefix="/api/report", tags=["report"])
    
    class SaveContentRequest(BaseModel):
        content: str
    
    @router.get("/sections")
    async def get_sections():
        """Get list of available documents."""
        return service.get_documents()
    
    @router.get("/content")
    async def get_content(document: str = Query(..., description="Document ID")):
        """Get document content."""
        result = service.get_document_content(document)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Document '{document}' not found")
        return result
    
    @router.put("/content")
    async def save_content(
        document: str = Query(..., description="Document ID"),
        request: SaveContentRequest = ...
    ):
        """Save content to a document."""
        result = service.save_document_content(document, request.content)
        if not result.get('success'):
            raise HTTPException(status_code=500, detail=result.get('error', 'Save failed'))
        return result
    
    @router.post("/refresh")
    async def refresh():
        """Reload configuration."""
        service.refresh()
        return {
            "status": "refreshed",
            "documents": len(service.documents),
            "config_loaded": service.config_loaded
        }
    
    return router


def init_report_service(report_dir: Path | str) -> ReportService:
    """Initialize report service."""
    return ReportService(Path(report_dir))
