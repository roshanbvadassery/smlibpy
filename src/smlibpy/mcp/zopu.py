"""Zopu integration of the MCP server: a persistent session and live publishing to a Zopu canvas.

Active only when ``ZOPU_SERVICES_URL`` is set (the Zopu agent machine sets it); without it the
server is exactly upstream smlibpy.

- Work is organised in *parts*: ``new_part`` starts a design with an empty session, ``open_part``
  brings an earlier one back. Each part keeps its scripts and checkpoints in
  ``$SMLIBPY_STATE_DIR/parts/<id>/session.json`` after every change; the open part is replayed
  when the server starts, so a model survives the machine sleeping, and a new design never
  inherits (or overwrites) the previous one.
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
    SMLIBPY_FILES_DIR       where published files are written (default: <state dir>/../files/<part>)
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
CURRENT_FILE = "current.json"
STALE_AFTER_S = 30 * 60
OPEN_SOURCE_LINES = 120
UPLOAD_TIMEOUT = 120.0
PLACE_TIMEOUT = 60.0
MCP_PROTOCOL = "2025-03-26"
# urllib's default "Python-urllib/x.y" is blocked outright by Cloudflare's browser-signature
# check (403, error 1010) on zones in front of Zopu services: always identify ourselves.
USER_AGENT = os.environ.get("ZOPU_HTTP_USER_AGENT") or "smlibpy-zopu/0.2 (+https://zopu.ai)"
MIME = {".glb": "model/gltf-binary", ".step": "model/step", ".stl": "model/stl", ".png": "image/png",
        ".dxf": "image/vnd.dxf"}


def _send(req: urllib.request.Request, timeout: float, with_headers: bool = False) -> Any:
    """Status and body of an HTTP request (and lower-cased headers); HTTP errors are answers too."""
    if not req.has_header("User-agent"):
        req.add_header("User-Agent", USER_AGENT)
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
        self.files_root = os.path.abspath(os.path.expanduser(
            os.environ.get("SMLIBPY_FILES_DIR") or os.path.join(os.path.dirname(self.state_dir), "files")))
        self.autopublish = os.environ.get("SMLIBPY_AUTOPUBLISH", "1").strip() not in ("0", "false", "off")
        self.part_id: str | None = None  # the open part (a directory under parts/)
        self.bound: dict[str, Any] | None = None  # its metadata: project, name, shape, version, ...
        self.notice: str | None = None  # told once, with the next tool answer
        self._io = threading.Lock()
        self._restoring = False

    # -- parts on disk ----------------------------------------------------------------
    @property
    def parts_dir(self) -> str:
        return os.path.join(self.state_dir, "parts")

    def _part_file(self, part_id: str) -> str:
        return os.path.join(self.parts_dir, part_id, STATE_FILE)

    @property
    def files_dir(self) -> str:
        return os.path.join(self.files_root, self.part_id or "_scratch")

    def _write_json(self, path: str, data: Any) -> None:
        with self._io:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                os.replace(tmp, path)
            except OSError:
                pass  # never break modeling over a state file

    def _read_json(self, path: str) -> dict[str, Any]:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _set_current(self, part_id: str | None) -> None:
        self.part_id = part_id
        self._write_json(os.path.join(self.state_dir, CURRENT_FILE), {"part": part_id})

    def save_state(self) -> None:
        """Called after every change of the session (Session.on_change): keeps the open part."""
        if self._restoring or not self.part_id:
            return
        if self.bound is not None:
            self.bound["updated"] = time.time()
        self._write_json(self._part_file(self.part_id), {
            "history": [{"code": c, "label": l} for c, l in self.session.history],
            "checkpoints": [{"label": k, "n": n} for k, n in self.session.checkpoints],
            "bound": self.bound,
            "saved": time.time(),
        })
        self._write_source(self.part_id, self.session.history, self.bound)

    # -- the part's visible source (cad/parts/<id>/model.py) ------------------------------
    @property
    def cad_dir(self) -> str:
        return os.path.dirname(self.state_dir)  # $WS/cad

    def _source_path(self, part_id: str) -> str:
        return os.path.join(self.cad_dir, "parts", part_id, "model.py")

    def _spec_path(self, bound: dict[str, Any] | None) -> str | None:
        b = bound or {}
        if b.get("spec") and os.path.exists(os.path.join(os.path.dirname(self.cad_dir), b["spec"])):
            return b["spec"]
        specs = os.path.join(self.cad_dir, "specs")
        slug = _slug(str(b.get("name") or ""))
        try:
            for f in sorted(os.listdir(specs)):
                if slug and f.endswith(".json") and _slug(f[:-5]).startswith(slug[:24]):
                    return f"cad/specs/{f}"
        except OSError:
            pass
        return None

    @staticmethod
    def _source_text(history: list, bound: dict[str, Any] | None, part_id: str) -> str:
        b = bound or {}
        head = [f"# Part: {b.get('name') or part_id}  (id {part_id}, shape {b.get('shape_id') or '-'}, "
                f"v{b.get('version') or 0}, project {b.get('project_id') or '-'})",
                "# Generated from the modeling session: every successful step, in order. Edit by running",
                "# new steps (or a build(spec) rebuild) — this file is a readable mirror, not executed directly.", ""]
        body = []
        for i, (code, label) in enumerate(history, 1):
            body.append(f"# --- step {i}: {label or 'unlabelled'} ---")
            body.append(code.rstrip())
            body.append("")
        return "\n".join(head + body)

    def _write_source(self, part_id: str, history: list, bound: dict[str, Any] | None) -> None:
        path = self._source_path(part_id)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._source_text(history, bound, part_id))
            b = bound or {}
            spec = self._spec_path(bound)
            with open(os.path.join(os.path.dirname(path), "README.md"), "w", encoding="utf-8") as f:
                f.write(f"# {b.get('name') or part_id}\n\n- part id: `{part_id}`\n- canvas shape: `{b.get('shape_id') or '-'}` "
                        f"(v{b.get('version') or 0})\n- project: `{b.get('project_id') or '-'}`\n"
                        f"- spec: `{spec or 'none yet'}`\n- source: `cad/parts/{part_id}/model.py` ({len(history)} steps)\n")
        except OSError:
            pass

    def source(self, key: str | None = None) -> str:
        if not key:
            if not self.part_id:
                raise KeyError("no part is open")
            return self._source_text(self.session.history, self.bound, self.part_id)
        for pid, data in self._all_parts():
            b = data.get("bound") or {}
            if key in (pid, b.get("shape_id")) or key.lower() == str(b.get("name") or "").lower():
                hist = [(h["code"], h.get("label")) for h in data.get("history", []) if h.get("code")]
                return self._source_text(hist, b, pid)
        raise KeyError(f"no part {key!r}")

    def _replay(self, data: dict[str, Any], background: bool) -> None:
        history = [(h["code"], h.get("label")) for h in data.get("history", []) if h.get("code")]
        checkpoints = [(c["label"], int(c["n"])) for c in data.get("checkpoints", [])]
        ready = threading.Event()

        def replay() -> None:
            with self.session._lock:
                ready.set()
                self._restoring = True
                try:
                    if history:
                        self.session.history = history
                        self.session._rebuild()
                    else:
                        self.session.clear()
                    n = len(self.session.history)
                    self.session.checkpoints = [(k, m) for k, m in checkpoints if m <= n]
                finally:
                    self._restoring = False
            self.save_state()

        if background:
            threading.Thread(target=replay, name="smlibpy-restore", daemon=True).start()
            ready.wait(5)  # the lock is held before the first tool call can take it
        else:
            replay()

    def _migrate_flat_state(self) -> None:
        """Before parts existed the session lived in state/session.json: keep it as a closed part."""
        old = os.path.join(self.state_dir, STATE_FILE)
        data = self._read_json(old)
        if not data:
            return
        if data.get("history") or data.get("bound"):
            name = (data.get("bound") or {}).get("name") or "Earlier session"
            part_id = f"{_slug(name)}-{uuid.uuid4().hex[:6]}"
            self._write_json(self._part_file(part_id), data)
        try:
            os.replace(old, old + ".migrated")
        except OSError:
            pass

    def restore_state(self) -> None:
        """At start: reopen the part that was open (replayed in the background)."""
        self._migrate_flat_state()
        part_id = self._read_json(os.path.join(self.state_dir, CURRENT_FILE)).get("part")
        if not part_id:
            return
        data = self._read_json(self._part_file(part_id))
        if not data:
            return
        self.part_id = part_id
        self.bound = data.get("bound")
        b = self.bound or {}
        age = time.time() - float(b.get("updated") or data.get("saved") or 0)
        if age > STALE_AFTER_S:
            shape = b.get("shape_id") or "not published"
            self.notice = (f"Note: the session still holds part {b.get('name')!r} (shape {shape}, "
                           f"v{b.get('version') or 0}, last edited {age / 60:.0f} min ago). For a different "
                           "design call new_part first; to keep editing it, go on.")
        self._replay(data, background=True)

    def take_notice(self) -> str | None:
        note, self.notice = self.notice, None
        return note

    # -- the part lifecycle -----------------------------------------------------------
    def new_part(self, project_id: str, name: str) -> str:
        if not project_id or not name:
            raise ValueError("new_part needs project_id and name")
        previous = self.bound
        part_id = f"{_slug(name)}-{uuid.uuid4().hex[:6]}"
        now = time.time()
        self.bound = {"project_id": project_id, "name": name, "shape_id": None, "version": 0, "names": None,
                      "plan_z": None, "created": now, "updated": now}
        self._set_current(part_id)
        self.session.clear()  # empty session; on_change writes the new part's state
        self.save_state()
        self.notice = None
        text = f"New part {name!r} (project {project_id}): the session is empty."
        if previous and previous.get("name"):
            text += (f" The previous part {previous['name']!r} (shape {previous.get('shape_id') or '-'}) is kept: "
                     "open_part brings it back.")
        return text + " Model it, then publish() places it on the canvas."

    def _all_parts(self) -> list[tuple[str, dict[str, Any]]]:
        out = []
        try:
            ids = os.listdir(self.parts_dir)
        except OSError:
            return []
        for pid in ids:
            data = self._read_json(self._part_file(pid))
            if data:
                out.append((pid, data))
        out.sort(key=lambda x: float((x[1].get("bound") or {}).get("updated") or x[1].get("saved") or 0), reverse=True)
        return out

    def list_parts(self) -> str:
        parts = self._all_parts()
        if not parts:
            return "No parts yet: new_part(project_id, name) starts one."
        rows = []
        for pid, data in parts[:40]:
            b = data.get("bound") or {}
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(b.get("updated") or data.get("saved") or 0)))
            size = " × ".join(f"{v:g}" for v in (b.get("stats") or {}).get("bbox", [])) or "-"
            rows.append(f"{'*' if pid == self.part_id else '-'} {b.get('name') or pid!r}: shape {b.get('shape_id') or '-'}"
                        f", v{b.get('version') or 0}, project {b.get('project_id') or '-'}, {size} mm, "
                        f"{len(data.get('history', []))} scripts, edited {when}")
        return "Parts (* = open), newest first:\n" + "\n".join(rows)

    def open_part(self, key: str) -> str:
        key = (key or "").strip()
        found = None
        for pid, data in self._all_parts():
            b = data.get("bound") or {}
            if key in (pid, b.get("shape_id")) or key.lower() == str(b.get("name") or "").lower():
                found = (pid, data)
                break
        if found is None:
            raise KeyError(f"no part {key!r}. list_parts() shows them; for a part made elsewhere, new_part and "
                           "rebuild it, then publish(shape_id=...) to update its shape.")
        pid, data = found
        self.save_state()
        self.bound = data.get("bound")
        self._set_current(pid)
        self.notice = None
        self._replay(data, background=False)
        b = self.bound or {}
        steps = [label or "unlabelled" for _, label in self.session.history]
        spec = self._spec_path(b)
        src = self._source_text(self.session.history, b, pid).splitlines()
        head = "\n".join(src[:OPEN_SOURCE_LINES])
        more = f"\n… ({len(src) - OPEN_SOURCE_LINES} more lines: smlib_source() for all)" if len(src) > OPEN_SOURCE_LINES else ""
        try:
            vars_ = self.session.call("vars").get("vars", [])
            var_line = ", ".join(f"{v['name']} ({v['type']})" for v in vars_[:30]) or "(none)"
        except Exception:
            var_line = "(unavailable)"
        return (f"Opened part {b.get('name')!r} (shape {b.get('shape_id') or '-'}, v{b.get('version') or 0}): "
                f"{len(steps)} scripts replayed. Later successful scripts update its shape.\n"
                f"SOURCE: cad/parts/{pid}/model.py   SPEC: {spec or 'none — derive cad/specs/<part>.json from the source before editing'}\n"
                f"STEPS: {' | '.join(steps)}\nVARIABLES: {var_line}\n\n{head}{more}")

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
                      names: list[str] | None = None, shape_id: str | None = None,
                      plan_z: float | None = None) -> str:
        import anyio

        if not self.part_id or not self.bound:
            raise RuntimeError("no part is open: call new_part(project_id, name) first (or open_part to edit one)")
        b = dict(self.bound)
        held = f"this session holds part {b.get('name')!r}; call new_part for a new design, or open_part to edit another"
        if project_id and b.get("project_id") and project_id != b["project_id"]:
            raise RuntimeError(f"project {project_id} is not this part's project ({b['project_id']}): {held}")
        if name and b.get("shape_id") and name != b.get("name"):
            raise RuntimeError(f"name {name!r} differs from the published part: {held}")
        if shape_id and b.get("shape_id") and shape_id != b["shape_id"]:
            raise RuntimeError(f"shape {shape_id} is not this part's shape ({b['shape_id']}): {held}")
        project_id = project_id or b.get("project_id")
        if not project_id:
            raise ValueError("project_id is required")
        name = name or b.get("name") or "Part"
        names = names if names is not None else b.get("names")
        shape_id = shape_id or b.get("shape_id")
        plan_z = plan_z if plan_z is not None else b.get("plan_z")
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
            if plan_z is not None:
                plan = self.session.call("plan", 300.0, z=float(plan_z), path_dxf=f"{base}-plan.dxf",
                                         path_png=f"{base}-plan.png", names=names)
                if not plan.get("ok"):
                    raise RuntimeError(f"floor plan at z = {plan_z} failed: {plan.get('error')}")
                out["plan_dxf"] = plan["dxf"]
                out["plan_png"] = plan["png"]
            return out

        files = await anyio.to_thread.run_sync(export)
        stats = await anyio.to_thread.run_sync(self._stats, names)
        urls: dict[str, str] = {}
        for fmt, path in files.items():
            urls[fmt] = await anyio.to_thread.run_sync(self._upload, path, uid)

        args: dict[str, Any] = {"projectId": project_id, "name": name, "glbUrl": urls["glb"],
                                "stepUrl": urls.get("step", ""), "stlUrl": urls.get("stl", ""),
                                "pngUrl": urls.get("png", ""), "stats": json.dumps(stats)}
        if plan_z is not None:
            args["dxfUrl"] = urls.get("plan_dxf", "")
            args["planUrl"] = urls.get("plan_png", "")
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

        self.bound = {**b, "project_id": project_id, "shape_id": shape_id, "name": name, "names": names,
                      "plan_z": plan_z, "version": version, "urls": urls, "stats": stats}
        self.save_state()
        size = " × ".join(f"{v:g}" for v in stats.get("bbox", [])) or "?"
        return (f"Published {name!r} v{version} to the canvas (shape {shape_id}).\n"
                f"  size:   {size} mm\n  volume: {stats.get('volume', 0) / 1000:.3g} cm³\n"
                f"  valid:  {stats.get('valid')}\n"
                f"  STEP:   {urls.get('step', '')}\n  STL:    {urls.get('stl', '')}\n  GLB:    {urls['glb']}\n"
                + (f"  PLAN:   {urls.get('plan_png', '')} (cut at z = {plan_z:g} mm)\n"
                   f"  DXF:    {urls.get('plan_dxf', '')}\n" if plan_z is not None else "")
                + "Later successful scripts publish this part again automatically.")

    async def after_run(self) -> str | None:
        """After a successful script: publish the bound part again."""
        if not (self.bound and self.bound.get("shape_id") and self.autopublish):
            if not self.part_id:
                return "No part open — call new_part(project_id, name) before publishing."
            return None
        try:
            text = await self.publish()
            return text.splitlines()[0]
        except Exception as e:  # the script worked; say what went wrong with publishing
            return f"Canvas update failed: {e}"

    def unbind(self) -> str:
        """Close the open part: it stays on disk and on the canvas; scripts no longer update it."""
        was = self.bound
        self.save_state()
        self.bound = None
        self._set_current(None)
        if not was:
            return "No part was open."
        return (f"Closed part {was.get('name')!r} (shape {was.get('shape_id') or '-'}); it stays on the canvas, "
                "open_part brings it back.")


VIEWER_NOTE = ("VIEWER: the user's Zopu canvas is the viewer. publish puts the model there and later scripts "
               "update it. There is no local viewer: never give the user a localhost or 127.0.0.1 link.")


GUIDE = """
ZOPU CANVAS — PARTS
You work for a Zopu user; what you build is the deliverable on their project canvas.
- EVERY new design starts with new_part(project_id, name): an empty session for that part. Never build a \
new design on top of what the session holds.
- To change a part made earlier (another conversation): list_parts(), then open_part(shape_id or name) \
replays its scripts; then modify it.
- publish() exports what is shown (GLB, STEP, STL, a picture), uploads it and places the part's cad_model \
shape on the canvas. Publish once the first valid body exists. After that every successful run_script \
publishes again (same shape, next version), so the user watches it evolve. show(...) the deliverable.
- Buildings and layouts: publish(plan_z=1200) also cuts a floor plan at that height (DXF + hatched plan \
picture) on the same shape; later updates keep cutting it.
- publish refuses another project, name or shape than the open part's: that is a new part (new_part).
- Parts are kept on disk: they survive the machine sleeping.
"""
