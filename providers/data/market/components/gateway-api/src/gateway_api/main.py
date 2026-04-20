from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from .settings import settings

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger("gateway-api")

app = FastAPI(title="Market Data Gateway API", version="0.1.0")


@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> Response:
    logger.info("%s %s", request.method, request.url.path)
    response = await call_next(request)
    logger.info("%s %s -> %s", request.method, request.url.path, response.status_code)
    return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.api_route("/providers/kite/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy_kite(path: str, request: Request) -> Response:
    target_url = f"{settings.provider_api_base_url.rstrip('/')}/{path.lstrip('/')}"

    try:
        body = await request.body()
        headers = dict(request.headers)
        headers.pop("host", None)

        async with httpx.AsyncClient(timeout=60.0) as client:
            proxied = await client.request(
                method=request.method,
                url=target_url,
                params=request.query_params,
                content=body,
                headers=headers,
            )

        response_headers = {
            key: value
            for key, value in proxied.headers.items()
            if key.lower() not in {"content-encoding", "transfer-encoding", "connection"}
        }
        return Response(
            content=proxied.content,
            status_code=proxied.status_code,
            headers=response_headers,
            media_type=proxied.headers.get("content-type"),
        )
    except httpx.HTTPError as exc:
        return JSONResponse(
            status_code=502,
            content={"status": "error", "message": f"Provider unavailable: {exc}"},
        )
