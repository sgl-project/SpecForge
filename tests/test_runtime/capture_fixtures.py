"""Fake capture servers shared by capture protocol integration tests."""

import uuid

from specforge.inference.capture_protocol import attempt_keys
from tests.test_runtime.test_server_capture import (
    AUX_LAYERS,
    HIDDEN,
    _request_specs,
    _StubCaptureServer,
)


class ProtocolServer:
    def __init__(self, backend):
        self.backend = backend
        self.stub = _StubCaptureServer(backend)
        self.stub._result = lambda result: result
        self.active = {}
        self.retired = {}
        self.drained = True
        self.events = []
        self.capability = dict(
            protocol_version=1,
            server_epoch="epoch",
            capture_namespace="namespace",
            capture_signature="signature",
            cache_enabled=True,
            schema=[
                dict(
                    name="aux",
                    width=HIDDEN * len(AUX_LAYERS),
                    dtype="float32",
                    layer_ids=list(AUX_LAYERS),
                ),
                dict(name="last_hidden", width=HIDDEN, dtype="float32", layer_ids=[]),
            ],
        )

    def __call__(self, url, json_body, timeout):
        if url.endswith("/spec_capture/control"):
            body = json_body
            action = body["action"]
            self.events.append(action)
            if action == "capability":
                return self.capability
            sid = body["sample_id"]
            if action == "begin":
                if sid in self.active:
                    old = self.active[sid]
                    if (
                        old["attempt_id"] == body["attempt_id"]
                        and old["request_digest"] == body["request_digest"]
                    ):
                        return dict(old)
                    raise RuntimeError("prior attempt unclean")
                entry = {
                    k: body[k]
                    for k in (
                        "store_id",
                        "sample_id",
                        "gen",
                        "attempt_id",
                        "server_epoch",
                    )
                }
                entry.update(
                    reservation_token=uuid.uuid4().hex,
                    state="RESERVED",
                    writer_drained=False,
                    request_digest=body["request_digest"],
                    cleanup_owner=None,
                    cleanup_token=None,
                    readers={},
                    released_readers=set(),
                )
                self.active[sid] = entry
                return dict(entry)
            entry = self.active.get(sid)
            if entry is None or entry["reservation_token"] != body["reservation_token"]:
                entry = self.retired.get(body["reservation_token"])
            if entry is None or entry["reservation_token"] != body["reservation_token"]:
                raise RuntimeError("unknown attempt")
            if action == "resolve":
                if entry["state"] == "RETIRED":
                    return dict(entry)
                entry["writer_drained"] = self.drained
                if body.get("cancel"):
                    entry["state"] = "WRITER_DRAINED" if self.drained else "DRAINING"
                return dict(entry)
            if action == "read_acquire":
                if entry["state"] != "PUBLISHED" or entry["cleanup_token"]:
                    raise RuntimeError("attempt is not readable")
                owner = body["reader_owner"]
                token = entry["readers"].setdefault(owner, uuid.uuid4().hex)
                return dict(entry, reader_owner=owner, reader_token=token)
            if action == "read_release":
                owner, token = body["reader_owner"], body["reader_token"]
                if entry["readers"].get(owner) == token:
                    entry["readers"].pop(owner)
                    entry["released_readers"].add((owner, token))
                elif (owner, token) not in entry["released_readers"]:
                    raise RuntimeError("unknown read permit")
                return dict(
                    entry, reader_owner=owner, reader_token=token, reader_released=True
                )
            if action == "cleanup_acquire":
                if entry["readers"]:
                    raise RuntimeError("active readers")
                if entry["state"] == "RETIRED":
                    return dict(entry)
                if not entry["writer_drained"]:
                    raise RuntimeError("writer still live")
                if entry["cleanup_owner"] not in (None, body["cleanup_owner"]):
                    raise RuntimeError("cleanup owned by another store")
                entry["cleanup_owner"] = body["cleanup_owner"]
                entry["cleanup_token"] = entry["cleanup_token"] or uuid.uuid4().hex
                return dict(entry)
            if action == "cleanup_done":
                if body.get("cleanup_token") != entry["cleanup_token"]:
                    raise RuntimeError("cleanup token mismatch")
                if not entry["writer_drained"]:
                    raise RuntimeError("writer still live")
                if any(
                    key.startswith(f"{entry['store_id']}/{sid}/g1/")
                    for key in self.backend._d
                ):
                    raise RuntimeError("cleanup ack before key removal")
                entry["state"] = "RETIRED"
                self.retired[entry["reservation_token"]] = entry
                self.active.pop(sid, None)
                return dict(entry)
            raise RuntimeError(action)
        rows = self.stub(url, json_body, timeout)
        for row, spec, tokens in zip(
            rows, _request_specs(json_body), json_body["input_ids"]
        ):
            entry = self.active[spec["sample_id"]]
            entry.update(state="PUBLISHED", writer_drained=True)
            row["meta_info"]["spec_capture"].update(entry)
            row["meta_info"]["spec_capture"].update(
                {
                    k: self.capability[k]
                    for k in (
                        "protocol_version",
                        "capture_namespace",
                        "capture_signature",
                    )
                }
            )
            row["meta_info"]["spec_capture"].update(
                H=0,
                L=0,
                N=len(tokens),
                span=[0, len(tokens)],
                execution_epoch=0,
                reused_tokens=0,
            )
        return rows


class TransferServer(ProtocolServer):
    def __init__(self, backend):
        super().__init__(backend)
        self.capability.update(begin_batch=True, published_ownership="attempt-v1")
        for feature in self.capability["schema"]:
            feature["dtype"] = "bfloat16"
        self.calls = []

    def __call__(self, url, json_body, timeout):
        action = json_body.get("action", "generate")
        self.calls.append(action)
        if action == "begin_batch":
            replies = []
            for body in json_body["tasks"]:
                reply = super().__call__(url, dict(body, action="begin"), timeout)
                reply["object_key_layout"] = "attempt-v1"
                replies.append(reply)
            return dict(attempts=replies)
        if action == "begin":
            reply = super().__call__(url, json_body, timeout)
            reply["object_key_layout"] = "attempt-v1"
            return reply
        rows = super().__call__(url, json_body, timeout)
        if action != "generate":
            return rows
        for row in rows:
            result = row["meta_info"]["spec_capture"]
            keys = attempt_keys(result, result["features"])
            for name, key in keys.items():
                self.backend._d[key] = self.backend._d.pop(
                    f"{result['store_id']}/{result['sample_id']}/g1/{name}"
                )
                result["features"][name]["key"] = key
            result.update(
                object_key_layout="attempt-v1",
                publication_ownership="attempt-v1",
                physical_keys=keys,
            )
            entry = self.active.pop(result["sample_id"])
            certificate = {
                k: result[k]
                for k in (
                    "store_id",
                    "sample_id",
                    "gen",
                    "attempt_id",
                    "server_epoch",
                    "reservation_token",
                    "object_key_layout",
                    "publication_ownership",
                    "physical_keys",
                    "writer_drained",
                    "state",
                )
            }
            entry.update(
                state="RETIRED",
                publication_certificate=certificate,
                publication_ownership="attempt-v1",
                physical_keys=keys,
            )
            self.retired[entry["reservation_token"]] = entry
        return rows
