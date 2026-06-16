"""FastAPI web application for Prototype Visualizer."""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from utils.proto_viz_session import ProtoVizSession, list_available_datasets


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TEMPLATES = Jinja2Templates(directory=os.path.join(_THIS_DIR, "templates"))
_STATIC_DIR = os.path.join(_THIS_DIR, "static")


def _asset_version() -> int:
    """Use latest static file mtime as a cache-busting query value."""
    tracked = ["app.js", "styles.css"]
    mtimes = []
    for name in tracked:
        path = os.path.join(_STATIC_DIR, name)
        if os.path.exists(path):
            mtimes.append(int(os.path.getmtime(path)))
    return max(mtimes) if mtimes else 0


class LoadRequest(BaseModel):
    mode: str = "live"
    dataset: str = "eurosat"
    config: str = "configs_multi_proto"
    backbone: str = "ViT-B/16"
    data_root: str = "./data"
    n_samples: int = 200
    records: Optional[str] = None


class SelectRequest(BaseModel):
    selected_class_name: Optional[str] = None


class SetIndexRequest(BaseModel):
    idx: int
    selected_class_name: Optional[str] = None


def create_app(preload_records: Optional[str] = None) -> FastAPI:
    """Create a configured FastAPI app for the prototype visualizer."""
    app = FastAPI(title="Prototype Visualizer Web")
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    session = ProtoVizSession()

    if preload_records:
        session.load_replay(preload_records)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return _TEMPLATES.TemplateResponse(
            "index.html",
            {
                "request": request,
                "asset_version": _asset_version(),
            },
        )

    @app.get("/api/status")
    def status(selected_class_name: Optional[str] = None):
        if not session.loaded:
            return {
                "loaded": False,
                "datasets": list_available_datasets("configs_multi_proto"),
            }
        return {
            "loaded": True,
            "state": session.current_payload(selected_class_name=selected_class_name),
            "datasets": list_available_datasets("configs_multi_proto"),
        }

    @app.post("/api/load")
    def load_data(req: LoadRequest):
        try:
            if req.mode == "replay":
                if not req.records:
                    raise ValueError("records path is required for replay mode")
                session.load_replay(req.records, dataset_name=req.dataset)
            else:
                session.load_live(
                    dataset=req.dataset,
                    config=req.config,
                    backbone=req.backbone,
                    data_root=req.data_root,
                    n_samples=req.n_samples,
                )
            return {
                "ok": True,
                "state": session.current_payload(),
                "source": session.source,
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/next")
    def next_sample(req: SelectRequest):
        try:
            session.next()
            return {"ok": True, "state": session.current_payload(req.selected_class_name)}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/restart")
    def restart(req: SelectRequest):
        try:
            session.restart()
            return {"ok": True, "state": session.current_payload(req.selected_class_name)}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/set-index")
    def set_index(req: SetIndexRequest):
        try:
            session.fast_forward_to(req.idx)
            return {
                "ok": True,
                "state": session.current_payload(selected_class_name=req.selected_class_name),
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/export-current")
    def export_current():
        try:
            out_path = session.export_current()
            return {"ok": True, "path": out_path}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
