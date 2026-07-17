"""FastAPI web application for Prototype Visualizer."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import yaml
from pydantic import BaseModel

from utils.proto_viz_session import ProtoVizSession, list_available_datasets


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_TEMPLATES = Jinja2Templates(directory=os.path.join(_THIS_DIR, "templates"))
_STATIC_DIR = os.path.join(_THIS_DIR, "static")


def _list_config_dirs() -> list[str]:
    """Return sorted list of config directory names found in the repo root."""
    entries = []
    for name in os.listdir(_REPO_ROOT):
        if os.path.isdir(os.path.join(_REPO_ROOT, name)) and name.startswith("configs"):
            entries.append(name)
    return sorted(entries)


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
    config: str = "configs_patch_modulated_pta"
    config_dir: str = "configs_patch_modulated_pta"
    backbone: str = "ViT-B/16"
    data_root: str = "./data"
    n_samples: int = 200
    records: Optional[str] = None


class SelectRequest(BaseModel):
    selected_class_name: Optional[str] = None


class SetIndexRequest(BaseModel):
    idx: int
    selected_class_name: Optional[str] = None


class ConfigLoadRequest(BaseModel):
    config_dir: str
    dataset: str


class ConfigSaveRequest(BaseModel):
    config_dir: str
    dataset: str
    overrides: Dict[str, Any]


class ConfigCleanupRequest(BaseModel):
    temp_path: str


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
    def status(config_dir: str = "configs_patch_modulated_pta",
               selected_class_name: Optional[str] = None):
        if not session.loaded:
            return {
                "loaded": False,
                "config_dirs": _list_config_dirs(),
                "datasets": list_available_datasets(config_dir),
            }
        return {
            "loaded": True,
            "config_dirs": _list_config_dirs(),
            "state": session.current_payload(selected_class_name=selected_class_name),
            "datasets": list_available_datasets(config_dir),
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

    _TMP_CONFIGS_DIR = os.path.join(_REPO_ROOT, "outputs", "viz", "tmp_configs")

    @app.post("/api/config/load")
    def config_load(req: ConfigLoadRequest):
        config_path = os.path.join(_REPO_ROOT, req.config_dir, f"{req.dataset}.yaml")
        if not os.path.isfile(config_path):
            raise HTTPException(status_code=404, detail=f"Config not found: {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return {"ok": True, "config": config, "path": config_path}

    @app.post("/api/config/save-temp")
    def config_save_temp(req: ConfigSaveRequest):
        config_path = os.path.join(_REPO_ROOT, req.config_dir, f"{req.dataset}.yaml")
        if not os.path.isfile(config_path):
            raise HTTPException(status_code=404, detail=f"Config not found: {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        for key, value in req.overrides.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)
            else:
                config[key] = value
        os.makedirs(_TMP_CONFIGS_DIR, exist_ok=True)
        ts = int(time.time() * 1000)
        temp_name = f"{req.config_dir}_{req.dataset}_{ts}.yaml"
        temp_path = os.path.join(_TMP_CONFIGS_DIR, temp_name)
        with open(temp_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        return {"ok": True, "temp_path": temp_path, "config": config}

    @app.post("/api/config/cleanup")
    def config_cleanup(req: ConfigCleanupRequest):
        real_tmp = os.path.realpath(req.temp_path)
        real_dir = os.path.realpath(_TMP_CONFIGS_DIR)
        if not real_tmp.startswith(real_dir + os.sep) and real_tmp != real_dir:
            raise HTTPException(status_code=400, detail="Path is not inside tmp_configs directory")
        if os.path.isfile(real_tmp):
            os.remove(real_tmp)
        return {"ok": True}

    return app


app = create_app()
