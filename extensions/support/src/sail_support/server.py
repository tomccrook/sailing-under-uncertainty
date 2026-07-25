"""Optional loopback-only importer. Public builds never submit research/cloud jobs."""

import re
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from sail_support.tracks import import_track, save_track

ORIGINS = {"http://127.0.0.1:5187", "http://localhost:5187", "http://127.0.0.1:5188"}


def create_app(output: Path) -> FastAPI:
    app = FastAPI(title="Sail Lab private local importer", docs_url=None, redoc_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=sorted(ORIGINS), allow_methods=["GET", "POST"])

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "local", "cloud_submission": "disabled"}

    @app.post("/tracks/import")
    async def upload(request: Request, format: Literal["csv", "gpx"]) -> dict[str, object]:
        # CORS alone doesn't stop cross-origin writes from reaching a local server.
        if request.headers.get("origin") not in ORIGINS | {None}:
            raise HTTPException(403, "Only the local demo can import a private track")
        chunks = bytearray()
        async for chunk in request.stream():
            chunks.extend(chunk)
            if len(chunks) > 5_000_000:
                raise HTTPException(413, "Track exceeds 5 MB")
        identifier = uuid4().hex
        try:
            track = import_track(bytes(chunks), format, session_id=f"local-{identifier}")
            save_track(track, bytes(chunks), output / identifier)
        except (ValueError, UnicodeError) as error:
            raise HTTPException(422, str(error)) from error
        return {"import_id": identifier, "track": track.model_dump(mode="json")}

    @app.get("/tracks/{identifier}")
    def get_track(identifier: str) -> dict[str, object]:
        import json

        if not re.fullmatch(r"[0-9a-f]{32}", identifier):
            raise HTTPException(400, "Invalid import identifier")
        path = output / identifier / "track.json"
        if not path.exists():
            raise HTTPException(404, "Private import not found")
        result: dict[str, object] = json.loads(path.read_text())
        return result

    return app
