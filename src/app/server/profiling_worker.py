"""
Profiling Worker - Connects ProfilingService to Socket.IO for real-time updates.

This worker:
- Sets up event callbacks on ProfilingService
- Emits Socket.IO events when profiling status changes
- Allows clients to track profiling progress in real-time

Location: /src/app/server/profiling_worker.py
"""

import asyncio
from typing import Optional, Any


class ProfilingWorker:
    """
    Worker that bridges ProfilingService events to Socket.IO.
    
    Emits the following Socket.IO events:
    - profiling_started: Session created, profiling beginning
    - profiling_progress: Status update (running, extracting, etc.)
    - profiling_complete: Session finished successfully
    - profiling_error: Session failed with error
    """
    
    def __init__(
        self,
        profiling_service,
        socketio,
        event_loop: Optional[asyncio.AbstractEventLoop] = None
    ):
        """
        Initialize profiling worker.
        
        Args:
            profiling_service: ProfilingService instance
            socketio: Socket.IO AsyncServer instance
            event_loop: Event loop for async operations
        """
        self.service = profiling_service
        self.sio = socketio
        self.loop = event_loop
        
        # Register callback with profiling service
        self.service.set_event_callback(self._on_profiling_event)
        
        print("[ProfilingWorker] Initialized with event callback")
    
    def _on_profiling_event(self, event_type: str, session_id: str, data: dict):
        """
        Handle profiling events from service.
        
        Called from background thread, so we need to schedule
        the async emit on the event loop.
        """
        if self.loop is None:
            print(f"[ProfilingWorker] No event loop, skipping emit: {event_type}")
            return
        
        # Schedule async emit on the event loop
        asyncio.run_coroutine_threadsafe(
            self._emit_event(event_type, session_id, data),
            self.loop
        )
    
    async def _emit_event(self, event_type: str, session_id: str, data: dict):
        """Emit Socket.IO event to all clients in the profiling room."""
        event_name = f"profiling_{event_type}"
        
        payload = {
            "session_id": session_id,
            **data
        }
        
        # Emit to profiling room (clients join when they open profiling panel)
        # Also emit to specific session room for targeted updates
        try:
            await self.sio.emit(event_name, payload, room=f"profiling_{session_id}")
            await self.sio.emit(event_name, payload, room="profiling")
            print(f"[ProfilingWorker] Emitted {event_name}: {session_id}")
        except Exception as e:
            print(f"[ProfilingWorker] Failed to emit {event_name}: {e}")
    
    def start(self):
        """Start the worker (currently no-op, events are callback-driven)."""
        print("[ProfilingWorker] Started")
    
    def stop(self):
        """Stop the worker."""
        self.service.set_event_callback(None)
        print("[ProfilingWorker] Stopped")


def create_profiling_worker(
    profiling_service,
    socketio,
    event_loop: Optional[asyncio.AbstractEventLoop] = None,
    auto_start: bool = True
) -> ProfilingWorker:
    """
    Factory function to create and optionally start a ProfilingWorker.
    
    Args:
        profiling_service: ProfilingService instance
        socketio: Socket.IO AsyncServer instance
        event_loop: Event loop for async operations
        auto_start: Whether to start the worker immediately
        
    Returns:
        ProfilingWorker instance
    """
    worker = ProfilingWorker(
        profiling_service=profiling_service,
        socketio=socketio,
        event_loop=event_loop
    )
    
    if auto_start:
        worker.start()
    
    return worker
