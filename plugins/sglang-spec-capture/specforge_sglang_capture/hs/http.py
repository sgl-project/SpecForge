"""HTTP forwarding only; all HS state changes execute on TP schedulers."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse


def install_control_route(result, *args, **kwargs):
    from sglang.srt.entrypoints import http_server

    app = http_server.app
    if any(
        getattr(route, "path", None) == "/spec_capture/control" for route in app.routes
    ):
        return result

    async def spec_capture_control(request: Request):
        body = await request.json()
        if not isinstance(body, dict):
            return JSONResponse(
                {"error": "capture control requires a JSON object"}, status_code=400
            )
        manager = http_server.get_global_state().tokenizer_manager
        try:
            # plugin_control owns the request/reply communicator. It invokes
            # runtime.control on every scheduler rank in the same phase order.
            reply = await manager.plugin_control("specforge_hs", body)
        except (ValueError, RuntimeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
        return JSONResponse(reply)

    app.add_api_route("/spec_capture/control", spec_capture_control, methods=["POST"])
    return result
