#!/usr/bin/env python3
"""Launch the web-based Prototype Visualizer.

Examples:
    python proto_viz_app.py
    python proto_viz_app.py --host 0.0.0.0 --port 8080
    python proto_viz_app.py --records outputs/viz/caltech101_n500.pkl
"""

from __future__ import annotations

import argparse
import os

import uvicorn

from gui.proto_viz_web.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MultiProtoPTA Prototype Visualizer web app")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload")
    parser.add_argument("--records", type=str, default=None,
                        help="Optional records .pkl to preload in replay mode")
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    app = create_app(preload_records=args.records)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
