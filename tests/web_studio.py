"""ltx studio's server for its tests, with helmstudio played by an in-memory stand-in.

ltx studio keeps everything through helmstudio's runtime SDK, so its tests need
a helmstudio. ``FakeHelmstudio`` is its studio API in memory, shaped as the
SDK's client (``client.<group>.<method>``), so neither helmstudio nor
helm-runtime-sdk has to be installed: stdlib only, no MLX, no HTTP server, no
weights.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import os
import queue
import stat
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_server() -> Any:
    name = "ltx_studio_server"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "web" / "server.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


server = _load_server()


class RefusedError(Exception):
    """The runtime SDK's HelmError, as far as ltx studio reads one."""

    def __init__(self, status: int, code: str) -> None:
        super().__init__(f"{code} ({status})")
        self.status, self.code = status, code


def apply_merge_patch(target: Any, patch: Any) -> Any:
    """RFC 7396, as helmstudio applies a merge patch."""
    if not isinstance(patch, dict):
        return patch
    result = dict(target) if isinstance(target, dict) else {}
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = apply_merge_patch(result.get(key), value)
    return result


class FakeHelmstudio:
    """helmstudio's studio API in memory, as far as ltx studio calls it."""

    def __init__(self, root: Path) -> None:
        self.stage, self.data = root / "stage", root / "data"
        self.stage.mkdir(parents=True)
        self.data.mkdir()
        self.ids = itertools.count(1)
        self.clock = itertools.count(1)
        self.session_rows: dict[str, dict[str, Any]] = {}
        self.updates: list[dict[str, Any]] = []  # every session update's body, in order
        self.asset_rows: dict[str, dict[str, Any]] = {}
        self.blobs: dict[str, bytes] = {}
        self.items: dict[str, dict[str, Any]] = {}
        self.timelines: dict[str, dict[str, Any]] = {}
        self.records: dict[str, list[dict[str, Any]]] = {}
        self.kv: dict[tuple[str, str], dict[str, Any]] = {}
        self.job_rows: dict[str, dict[str, Any]] = {}
        self.job_logs: dict[str, list[str]] = {}
        self.events: queue.Queue = queue.Queue()

    def client(self) -> SimpleNamespace:
        """The runtime SDK's client, backed by this fake."""
        paths = {"stage": str(self.stage), "data": str(self.data)}
        return SimpleNamespace(
            me=SimpleNamespace(get=lambda: {"paths": paths}),
            sessions=SimpleNamespace(list=self._sessions_list, create=self._session_create,
                                     update=self._session_update, delete=self._session_delete,
                                     duplicate=self._session_duplicate, activate=self._session_activate),
            kv=SimpleNamespace(get=self._kv_get, put=self._kv_put, patch=self._kv_patch),
            records=SimpleNamespace(insert=self._record_insert),
            assets=SimpleNamespace(adopt=self._adopt, read=lambda id: SimpleNamespace(read=lambda: self.blobs[id])),
            gallery=SimpleNamespace(add=self._gallery_add, query=self._gallery_query,
                                    update=lambda id, body: self.items[id].update(body),
                                    delete=lambda id: self.items[id].update(deleted=True)),
            timeline=SimpleNamespace(list=self._timeline_list),
            jobs=SimpleNamespace(create=self._job_create, update=self._job_update, append_log=self._append_log,
                                 list=self._jobs_list, logs=self._job_log),
            events=SimpleNamespace(subscribe=self._subscribe),
        )  # fmt: skip

    def _id(self) -> str:
        return f"01TEST{next(self.ids):020d}"

    @staticmethod
    def _page(rows: list[dict[str, Any]], limit: int, cursor: str | None) -> dict[str, Any]:
        start = int(cursor or 0)
        return {"items": copy.deepcopy(rows[start : start + limit]),
                "next_cursor": str(start + limit) if start + limit < len(rows) else None}  # fmt: skip

    # sessions
    def _live(self, id: str) -> dict[str, Any]:
        row = self.session_rows.get(id)
        if row is None or row.get("deleted"):
            raise RefusedError(404, "not_found")
        return row

    def _sessions_list(self, limit: int, cursor: str | None = None) -> dict[str, Any]:
        live = [row for row in self.session_rows.values() if not row.get("deleted")]
        live.sort(key=lambda row: -(row["opened_at"] or 0))  # most recently opened first, never opened last
        return self._page(live, limit, cursor)

    def _session_create(self, body: dict[str, Any]) -> dict[str, Any]:
        if any(row["name"] == body["name"] for row in self.session_rows.values() if not row.get("deleted")):
            raise RefusedError(409, "name_taken")
        row = {"id": self._id(), "name": body["name"], "state": body.get("state") or {}, "etag": self._id(),
               "opened_at": None}  # fmt: skip
        self.session_rows[row["id"]] = row
        return copy.deepcopy(row)

    def _session_update(self, id: str, body: dict[str, Any], *, if_match: str | None = None) -> dict[str, Any]:
        row = self._live(id)
        if if_match is not None and if_match != row["etag"]:
            raise RefusedError(409, "etag_mismatch")
        self.updates.append(copy.deepcopy(body))
        row["state"], row["etag"] = apply_merge_patch(row["state"], body["state"]), self._id()
        return copy.deepcopy(row)

    def _session_delete(self, id: str) -> None:
        self._live(id)["deleted"] = True

    def _session_duplicate(self, id: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._session_create({"name": body["name"], "state": copy.deepcopy(self._live(id)["state"])})

    def _session_activate(self, id: str) -> None:
        self._live(id)["opened_at"] = next(self.clock)

    # kv
    def _kv_get(self, ns: str, key: str) -> dict[str, Any]:
        if (ns, key) not in self.kv:
            raise RefusedError(404, "not_found")
        return {"ns": ns, "key": key, "doc": copy.deepcopy(self.kv[(ns, key)]), "etag": "e", "updated_at": "t"}

    def _kv_put(self, ns: str, key: str, body: dict[str, Any], *, if_match: str | None = None) -> dict[str, Any]:
        self.kv[(ns, key)] = copy.deepcopy(body)
        return self._kv_get(ns, key)

    def _kv_patch(self, ns: str, key: str, body: dict[str, Any], *, if_match: str | None = None) -> dict[str, Any]:
        document = self._kv_get(ns, key)["doc"]
        self.kv[(ns, key)] = apply_merge_patch(document, body)
        return self._kv_get(ns, key)

    # records
    def _record_insert(self, collection: str, body: dict[str, Any]) -> dict[str, Any]:
        record = {"id": self._id(), "collection": collection, "doc": copy.deepcopy(body), "etag": "e"}
        self.records.setdefault(collection, []).append(record)
        return copy.deepcopy(record)

    # assets
    def _adopt(self, body: dict[str, Any]) -> dict[str, Any]:
        path = Path(body["path"])
        from_stage = self.stage in path.parents
        assert from_stage or self.data in path.parents, "adopts only from the stage or the data directory"
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        row = next((row for row in self.asset_rows.values() if row["sha256"] == digest), None)
        if row is None:
            row = {"id": self._id(), "sha256": digest, "kind": body["kind"], "bytes": len(data), "pinned": False}
            self.asset_rows[row["id"]], self.blobs[row["id"]] = row, data
        row["pinned"] = row["pinned"] or bool(body.get("pinned"))
        if from_stage:
            path.unlink()
        else:  # stays where it is, sharing the blob's read-only inode
            os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        return dict(row)

    # gallery
    def _gallery_add(self, body: dict[str, Any]) -> dict[str, Any]:
        self._live(body["session_id"])
        item = {"id": self._id(), "kind": body["kind"], "asset_id": body["asset_id"],
                "asset": self.asset_rows[body["asset_id"]], "session_id": body["session_id"], "timeline_id": None,
                "title": body.get("title"), "params": body["params"], "starred": False,
                "inputs": body.get("inputs", []), "created_at": f"t{next(self.clock)}"}  # fmt: skip
        self.items[item["id"]] = item
        return copy.deepcopy(item)

    def _gallery_query(self, *, limit: int, cursor: str | None = None, session_id: str | None = None,
                       kind: str | None = None) -> dict[str, Any]:  # fmt: skip
        found = [item for item in reversed(self.items.values()) if not item.get("deleted")
                 and session_id in (None, item["session_id"]) and kind in (None, item["kind"])]  # fmt: skip
        return self._page(found, limit, cursor)

    # timeline
    def add_sequence(self, name: str, clips: list[dict[str, Any]], **fields: Any) -> dict[str, Any]:
        """A sequence helmstudio holds for this studio, as ``GET /timeline`` answers with it."""
        row = {"id": self._id(), "name": name, "revision": 1, "duration_s": None, "etag": "e",
               "target": {"width": 704, "height": 448, "fps": 24, "sample_rate": 48000},
               "created_at": f"t{next(self.clock)}", "updated_at": f"t{next(self.clock)}",
               "tracks": [{"kind": "video", "name": "V1", "clips": clips}]}  # fmt: skip
        row.update(fields)
        self.timelines[row["id"]] = row
        return copy.deepcopy(row)

    def _timeline_list(self, *, limit: int, cursor: str | None = None) -> dict[str, Any]:
        return self._page(list(reversed(self.timelines.values())), limit, cursor)

    # jobs
    def _job_create(self, body: dict[str, Any]) -> dict[str, Any]:
        job = {"id": self._id(), "kind": "task", "state": body["state"], "subject_kind": body["subject_kind"],
               "subject_id": body["subject_id"], "progress_num": 0, "progress_den": 0}  # fmt: skip
        self.job_rows[job["id"]], self.job_logs[job["id"]] = job, []
        return dict(job)

    def _job_update(self, id: str, body: dict[str, Any]) -> None:
        assert self.job_rows[id]["state"] in ("queued", "running"), "a finished job cannot change"
        self.job_rows[id].update(body)

    def _append_log(self, id: str, body: dict[str, Any]) -> None:
        assert self.job_rows[id]["state"] in ("queued", "running") and 1 <= len(body["lines"]) <= 1000
        self.job_logs[id] += body["lines"]

    def _jobs_list(self, limit: int, cursor: str | None = None) -> dict[str, Any]:
        return self._page(list(reversed(self.job_rows.values())), limit, cursor)

    def _job_log(self, id: str) -> Any:
        for line in list(self.job_logs[id]):
            yield SimpleNamespace(name="line", json=lambda line=line: {"text": line})
        yield SimpleNamespace(name="end", json=lambda: {"state": self.job_rows[id]["state"]})

    # events
    def _subscribe(self, *, last_event_id: str | None = None) -> Any:
        while True:
            yield self.events.get()


def fake_helmstudio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHelmstudio:
    """A fake helmstudio under ``tmp_path``, whose refusals ltx studio reads as the SDK's."""
    monkeypatch.setattr(server, "HelmError", RefusedError)
    return FakeHelmstudio(tmp_path)


def runner_for(state: Any) -> Any:
    """A Runner without its worker thread, so a submitted job stays queued."""
    runner = server.Runner.__new__(server.Runner)
    runner.state = state
    runner.jobs, runner.pending, runner.current, runner.proc = {}, [], None, None
    runner.cond, runner.job_lock, runner.sub_lock, runner.subscribers = (
        threading.Condition(), threading.Lock(), threading.Lock(), [])  # fmt: skip
    return runner
