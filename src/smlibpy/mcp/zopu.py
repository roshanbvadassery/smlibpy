"""Zopu integration of the MCP server: a persistent session and live publishing to a Zopu canvas.

Active only when ``ZOPU_SERVICES_URL`` is set (the Zopu agent machine sets it); without it the
server is exactly upstream smlibpy.

- The session's scripts and checkpoints are kept in ``$SMLIBPY_STATE_DIR/session.json`` after
  every change, and replayed when the server starts, so a model survives the machine sleeping.
- ``publish`` writes GLB, STEP, STL and a PNG of what is shown, uploads them to Zopu storage
  (``$ZOPU_SERVICES_URL/store-file``) and places or updates a ``cad_model`` shape on the
  project canvas through the Zopu canvas MCP (``place_cad_model`` at ``$ZOPU_MCP_URL``).
- Once published, the part is *bound*: every later successful script publishes it again, so
  the canvas follows the model.

Environment:
    ZOPU_SERVICES_URL       Zopu services base URL (file uploads); turns this module on
    ZOPU_MCP_URL            the Zopu canvas MCP endpoint (streamable HTTP)
    ZOPU_MCP_REFRESH_TOKEN  bearer token for it (the owner's Firebase refresh token)
    ZOPU_WS                 the workspace; ``.session.json`` there holds the user id
    ZOPU_UID                the user id, instead of ``.session.json``
    SMLIBPY_STATE_DIR       where the session is kept (default: ~/.smlibpy/state)
    SMLIBPY_FILES_DIR       where published files are written (default: <state dir>/../files)
    SMLIBPY_AUTOPUBLISH     ``0`` turns off publishing after every script
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

from .session import Session

STATE_FILE = "session.json"
UPLOAD_TIMEOUT = 120.0
PLACE_TIMEOUT = 60.0
MCP_PROTOCOL = "2025-03-26"
MIME = {".glb": "model/gltf-binary", ".step": "model/step", ".stl": "model/stl", ".png": "image/png"}


def _send(req: urllib.request.Request, timeout: float, with_headers: bool = False) -> Any:
    """Status and body of an HTTP request (and lower-cased headers); HTTP errors are answers too."""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            status, text, headers = res.status, res.read().decode("utf-8", "replace"), res.headers
    except urllib.error.HTTPError as e:
        status, text, headers = e.code, e.read().decode("utf-8", "replace"), e.headers
    if with_headers:
        return status, text, {k.lower(): v for k, v in (headers or {}).items()}
    return status, text


def enabled() -> bool:
    return bool(os.environ.get("ZOPU_SERVICES_URL", "").strip())


def _slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()
    return s[:60] or "part"


class Zopu:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.services = os.environ["ZOPU_SERVICES_URL"].rstrip("/")
        self.mcp_url = os.environ.get("ZOPU_MCP_URL", "").strip()
        self.mcp_token = os.environ.get("ZOPU_MCP_REFRESH_TOKEN", "").strip()
        self.state_dir = os.path.abspath(os.path.expanduser(
            os.environ.get("SMLIBPY_STATE_DIR") or os.path.join("~", ".smlibpy", "state")))
        self.files_dir = os.path.abspath(os.path.expanduser(
            os.environ.get("SMLIBPY_FILES_DIR") or os.path.join(os.path.dirname(self.state_dir), "files")))
        self.autopublish = os.environ.get("SMLIBPY_AUTOPUBLISH", "1").strip() not in ("0", "false", "off")
        self.bound: dict[str, Any] | None = None
        self._io = threading.Lock()
        self._restoring = False

    # -- persistence ------------------------------------------------------------------
    @property
    def _state_path(self) -> str:
        return os.path.join(self.state_dir, STATE_FILE)

    def save_state(self) -> None:
        """Called after every change of the session (Session.on_change)."""
        if self._restoring:
            return
        data = {
            "history": [{"code": c, "label": l} for c, l in self.session.history],
            "checkpoints": [{"label": k, "n": n} for k, n in self.session.checkpoints],
            "bound": self.bound,
            "saved": time.time(),
        }
        with self._io:
            try:
                os.makedirs(self.state_dir, exist_ok=True)
                tmp = self._state_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                os.replace(tmp, self._state_path)
            except OSError:
                pass  # never break modeling over a state file

    def restore_state(self) -> None:
        """Replay the kept session in the background; tools wait on the session lock meanwhile."""
        try:
            with open(self._state_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        self.bound = data.get("bound")
        history = [(h["code"], h.get("label")) for h in data.get("history", []) if h.get("code")]
        checkpoints = [(c["label"], int(c["n"])) for c in data.get("checkpoints", [])]
        if not history:
            return
        ready = threading.Event()

        def replay() -> None:
            with self.session._lock:
                ready.set()
                self._restoring = True
                try:
                    self.session.history = history
                    self.session._rebuild()
                    n = len(self.session.history)
                    self.session.checkpoints = [(k, m) for k, m in checkpoints if m <= n]
                finally:
                    self._restoring = False
            self.save_state()

        threading.Thread(target=replay, name="smlibpy-restore", daemon=True).start()
        ready.wait(5)  # the lock is held before the first tool call can take it

    # -- publishing -------------------------------------------------------------------
    def _uid(self) -> str:
        uid = os.environ.get("ZOPU_UID", "").strip()
        if uid:
            return uid
        return str(self._session_file().get("uid") or "")

    def _session_file(self) -> dict[str, Any]:
        ws = os.environ.get("ZOPU_WS") or os.getcwd()
        try:
            with open(os.path.join(ws, ".session.json"), encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _token(self) -> str:
        """The owner's Firebase refresh token: from the environment, else the one the agent keeps in
        the workspace's .session.json (it rotates independently of the machine env)."""
        return self.mcp_token or str(self._session_file().get("refreshToken") or "").strip()

    def _upload(self, path: str, uid: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        boundary = "smlibpy" + uuid.uuid4().hex
        with open(path, "rb") as f:
            data = f.read()
        filename = os.path.basename(path)
        body = b"".join([
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"userid\"\r\n\r\n{uid}\r\n".encode(),
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
            f"Content-Type: {MIME.get(ext, 'application/octet-stream')}\r\n\r\n".encode(),
            data,
            f"\r\n--{boundary}--\r\n".encode(),
        ])
        req = urllib.request.Request(self.services + "/store-file", data=body, method="POST",
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        status, text = _send(req, UPLOAD_TIMEOUT)
        try:
            answer = json.loads(text)
        except ValueError:
            answer = {}
        if status >= 400 or not answer.get("success") or not answer.get("fileUrl"):
            raise RuntimeError(f"upload of {filename} failed ({status}): {answer.get('error') or text[:200]}")
        return str(answer["fileUrl"])

    def _rpc(self, method: str, params: dict[str, Any] | None, sid: str | None,
             notify: bool = False) -> tuple[dict[str, Any], str | None]:
        """One JSON-RPC message to the canvas MCP (streamable HTTP; JSON or SSE answers)."""
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        rid = uuid.uuid4().hex
        if not notify:
            msg["id"] = rid
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                   "MCP-Protocol-Version": MCP_PROTOCOL}
        token = self._token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if sid:
            headers["Mcp-Session-Id"] = sid
        req = urllib.request.Request(self.mcp_url, data=json.dumps(msg).encode(), method="POST", headers=headers)
        status, text, resp_headers = _send(req, PLACE_TIMEOUT, with_headers=True)
        sid = resp_headers.get("mcp-session-id") or sid
        if notify:
            return {}, sid
        if status >= 400:
            raise RuntimeError(f"canvas MCP {method} failed ({status}): {text[:300]}")
        answer: dict[str, Any] | None = None
        if resp_headers.get("content-type", "").startswith("text/event-stream"):
            for frame in text.split("\n\n"):
                payload = "\n".join(l[5:].lstrip() for l in frame.splitlines() if l.startswith("data:"))
                if not payload:
                    continue
                try:
                    m = json.loads(payload)
                except ValueError:
                    continue
                if m.get("id") == rid:
                    answer = m
        else:
            answer = json.loads(text or "{}")
        if not answer:
            raise RuntimeError(f"canvas MCP {method}: no answer")
        if answer.get("error"):
            raise RuntimeError(f"canvas MCP {method}: {answer['error'].get('message')}")
        return answer.get("result") or {}, sid

    def _place(self, args: dict[str, Any]) -> str:
        """Call place_cad_model on the Zopu canvas MCP; returns its text answer."""
        if not self.mcp_url:
            raise RuntimeError("ZOPU_MCP_URL is not set: cannot reach the Zopu canvas")
        _, sid = self._rpc("initialize", {"protocolVersion": MCP_PROTOCOL, "capabilities": {},
                                          "clientInfo": {"name": "smlibpy-zopu", "version": "1"}}, None)
        self._rpc("notifications/initialized", None, sid, notify=True)
        result, _ = self._rpc("tools/call", {"name": "place_cad_model", "arguments": args}, sid)
        text = "\n".join(c.get("text", "") for c in result.get("content", []) if isinstance(c, dict))
        if result.get("isError") or text.startswith("Error"):
            raise RuntimeError(text or "place_cad_model failed")
        return text

    def _stats(self, names: list[str] | None) -> dict[str, Any]:
        if names:
            rows = []
            for n in names:
                res = self.session.call("inspect", name=n)
                if res.get("ok"):
                    rows.append(res["info"])
        else:
            rows = self.session.call("scene").get("scene", [])
        lo = [float("inf")] * 3
        hi = [float("-inf")] * 3
        volume = area = 0.0
        valid = True
        for r in rows:
            bb = r.get("bbox")
            if bb:
                lo = [min(a, b) for a, b in zip(lo, bb["min"])]
                hi = [max(a, b) for a, b in zip(hi, bb["max"])]
            volume += float(r.get("volume") or 0)
            area += float(r.get("area") or 0)
            if r.get("valid") is False:
                valid = False
        stats: dict[str, Any] = {"bodies": [r.get("name") for r in rows], "valid": valid,
                                 "volume": round(volume, 3), "area": round(area, 3)}
        if lo[0] != float("inf"):
            stats["bbox"] = [round(h - l, 3) for l, h in zip(lo, hi)]
            stats["min"] = [round(v, 3) for v in lo]
            stats["max"] = [round(v, 3) for v in hi]
        return stats

    async def publish(self, project_id: str | None = None, name: str | None = None,
                      names: list[str] | None = None, shape_id: str | None = None) -> str:
        import anyio

        b = dict(self.bound or {})
        if project_id and project_id != b.get("project_id"):
            b = {}  # another project: a new shape
        project_id = project_id or b.get("project_id")
        if not project_id:
            raise ValueError("project_id is required for the first publish")
        name = name or b.get("name") or "Part"
        names = names if names is not None else b.get("names")
        shape_id = shape_id or b.get("shape_id")
        version = int(b.get("version") or 0) + 1

        uid = self._uid()
        if not uid:
            raise RuntimeError("no user id: .session.json in the workspace has no uid (set ZOPU_UID)")

        base = os.path.join(self.files_dir, _slug(name))
        os.makedirs(self.files_dir, exist_ok=True)

        def export() -> dict[str, str]:
            out: dict[str, str] = {}
            for fmt in ("glb", "step", "stl"):
                res = self.session.call("save", 600.0, format=fmt, path=f"{base}.{fmt}", names=names)
                if not res.get("ok"):
                    raise RuntimeError(f"{fmt} export failed: {res.get('error')}")
                out[fmt] = res["path"]
            snap = self.session.call("snapshot", 300.0, view="iso", size=(960, 720), names=names,
                                     path=f"{base}.png")
            if snap.get("ok"):
                out["png"] = f"{base}.png"
            return out

        files = await anyio.to_thread.run_sync(export)
        stats = await anyio.to_thread.run_sync(self._stats, names)
        urls: dict[str, str] = {}
        for fmt, path in files.items():
            urls[fmt] = await anyio.to_thread.run_sync(self._upload, path, uid)

        args: dict[str, Any] = {"projectId": project_id, "name": name, "glbUrl": urls["glb"],
                                "stepUrl": urls.get("step", ""), "stlUrl": urls.get("stl", ""),
                                "pngUrl": urls.get("png", ""), "stats": json.dumps(stats)}
        if shape_id:
            args["shapeId"] = shape_id
        try:
            text = await anyio.to_thread.run_sync(self._place, args)
        except RuntimeError as e:
            if shape_id and "not found" in str(e):  # the user deleted the shape: place a new one
                args.pop("shapeId")
                text = await anyio.to_thread.run_sync(self._place, args)
            else:
                raise
        m = re.search(r"id:\s+(shape:\S+)", text)
        if m:
            shape_id = m.group(1)
        m = re.search(r"version:\s+(\d+)", text)
        if m:
            version = int(m.group(1))

        self.bound = {"project_id": project_id, "shape_id": shape_id, "name": name, "names": names,
                      "version": version, "urls": urls, "stats": stats}
        self.save_state()
        size = " × ".join(f"{v:g}" for v in stats.get("bbox", [])) or "?"
        return (f"Published {name!r} v{version} to the canvas (shape {shape_id}).\n"
                f"  size:   {size} mm\n  volume: {stats.get('volume', 0) / 1000:.3g} cm³\n"
                f"  valid:  {stats.get('valid')}\n"
                f"  STEP:   {urls.get('step', '')}\n  STL:    {urls.get('stl', '')}\n  GLB:    {urls['glb']}\n"
                "Later successful scripts publish this part again automatically (unbind() to stop).")

    async def after_run(self) -> str | None:
        """After a successful script: publish the bound part again."""
        if not (self.bound and self.autopublish):
            return None
        try:
            text = await self.publish()
            return text.splitlines()[0]
        except Exception as e:  # the script worked; say what went wrong with publishing
            return f"Canvas update failed: {e}"

    def unbind(self) -> str:
        was = self.bound
        self.bound = None
        self.save_state()
        if not was:
            return "No part was bound."
        return f"Stopped publishing {was.get('name')!r} (shape {was.get('shape_id')}); it stays on the canvas."


GUIDE = """
ZOPU CANVAS
You work for a Zopu user; the part you build is the deliverable on their project canvas.
- publish(project_id, name) exports what is shown (GLB, STEP, STL, a picture), uploads it and places a \
cad_model shape on the canvas: a live 3D view with downloads. Publish once the first valid body exists.
- After that every successful run_script publishes the part again (same shape, next version), so the \
user watches it evolve. show(...) the deliverable so helpers and tools are left out.
- To change a part already on the canvas, pass its shape_id to publish. To start a different part, \
unbind() first, then publish it under its own name.
- The session is kept on disk: it survives the machine sleeping.
"""
