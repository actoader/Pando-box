#!/usr/bin/env python3
"""the loader - what the box image runs until Pando's code arrives
(thread H sitting 2, 2026-09-16).

WHY: the image is the MACHINE (CUDA, vLLM, poppler, tesseract and its
packs, the layout weights) and nothing of Pando's own code, because
core/ changes every sitting and an image with the code inside must be
rebuilt and pushed before every night: the pet machine in a new coat.
So the image boots this loader on port 8000, the Mac sends core/ (the
tarball core/dies/pod.py packs, one megabyte) as the first thing after
Start, and the loader becomes box_server: no version skew is possible,
the box always runs the Mac's own dies (PRINCIPLES.md rule 12).

WHAT (stdlib only, no Pando import: the code is not here yet):
    GET  /health   {"ok": true, "code": null, "image": the image's
                    version, "reader": the model vLLM serves on 8001 or
                    null while it loads, "key": whether LAYOUT_KEY is
                    set, "up_seconds"}; the same Authorization: Bearer
                    <LAYOUT_KEY> rule as box_server (401 without it)
    POST /code     body: the tarball (core/ and catalog/clean/
                    wordlist*.txt, as pod.pack makes it) -> 200
                    {"ok", "sha256"}, unpacked under /pando, then this
                    process EXECS core/dies/box_server.py on the same
                    port (the listening socket is not inherited: the
                    port frees, box_server binds it a second later;
                    the Mac's poll retries). /pando/code.json records
                    the sha256 and when; box_server's /health shows it
                    as "code", so the Mac can see what the box runs.
    anything else  404

THE GUARD: LAYOUT_KEY unset means the box has no key and the Mac
cannot have read one off the pod: /health says so and /code is
refused (403), so nothing runs unkeyed on a public port. And a pod
started but never given code (the app crashed, the Mac closed) stops
itself after BOX_LOADER_WAIT minutes (30): the same stop as
box_server's watchdog, the 15 lines it takes rather than an import
that does not exist yet.

Run by the image's entrypoint: python3 /box/loader.py --port 8000
"""
import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PANDO = os.environ.get("BOX_PANDO") or "/pando"     # where core/ unpacks
READER = os.environ.get("BOX_READER_URL") or \
    "http://127.0.0.1:8001/v1/chat/completions"
WAIT = int(os.environ.get("BOX_LOADER_WAIT") or 30)   # minutes, 0 never
STARTED = time.time()
_EXEC = []


def say(msg):
    print("%s  loader: %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg),
          flush=True)


def image_version():
    try:
        with open("/box/VERSION") as f:
            return f.read().strip()
    except OSError:
        return None


def reader_model():
    """The model id vLLM serves, or None while it loads (asked fresh
    each time: the loader lives minutes, not nights)."""
    try:
        base = READER.rsplit("/chat/completions", 1)[0]
        with urllib.request.urlopen(base + "/models", timeout=5) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
        ids = [m.get("id") for m in data.get("data") or [] if m.get("id")]
        return ids[0] if ids else None
    except Exception:
        return None


def unpack(data, where=None):
    """The tarball under `where` (PANDO): core/ and catalog/ replaced
    whole (never merged: a file deleted on the Mac must not survive
    here), every member checked to land inside `where`. -> sha256."""
    where = where or PANDO
    sha = hashlib.sha256(data).hexdigest()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
        names = t.getnames()
        for n in names:
            p = os.path.normpath(n)
            if p.startswith("..") or os.path.isabs(p):
                raise ValueError("a member escapes the hub: %s" % n)
        tops = {os.path.normpath(n).split(os.sep)[0] for n in names}
        os.makedirs(where, exist_ok=True)
        for top in tops:
            shutil.rmtree(os.path.join(where, top), ignore_errors=True)
        try:
            t.extractall(where, filter="data")
        except TypeError:                       # a python before 3.12
            t.extractall(where)
    with open(os.path.join(where, "code.json"), "w") as f:
        json.dump({"sha256": sha, "received": time.strftime(
            "%Y-%m-%dT%H:%M:%S"), "members": len(names)}, f)
    return sha


def stop_pod():
    """(ok, note): this pod stopped over RunPod's API with the
    pod-scoped key RunPod sets in every pod, runpodctl as the second
    arm; the same two arms as box_server's watchdog."""
    pod = os.environ.get("RUNPOD_POD_ID")
    if not pod:
        return False, "not a RunPod pod (RUNPOD_POD_ID unset)"
    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        try:
            req = urllib.request.Request(
                "https://api.runpod.io/v2/pods/%s/action" % pod,
                data=json.dumps({"action": "stop"}).encode(),
                method="POST", headers={
                    "Authorization": "Bearer " + key,
                    "Content-Type": "application/json",
                    "User-Agent": "pando-box/1"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return True, "api v2 stop -> %d" % r.status
        except Exception as e:
            note = "api v2 stop failed: %s" % str(e)[:120]
    else:
        note = "RUNPOD_API_KEY unset"
    if shutil.which("runpodctl"):
        try:
            r = subprocess.run(["runpodctl", "stop", "pod", pod],
                               capture_output=True, timeout=120)
            return r.returncode == 0, "%s; runpodctl -> %d %s" % (
                note, r.returncode,
                (r.stdout or r.stderr).decode("utf-8", "replace")[:120])
        except Exception as e:
            return False, "%s; runpodctl failed: %s" % (note, e)
    return False, note + "; runpodctl not on PATH"


def guard():
    """No code within WAIT minutes: stop the pod (a Start that nobody
    followed through costs by the hour)."""
    while WAIT and not _EXEC:
        time.sleep(30)
        if not _EXEC and time.time() - STARTED > WAIT * 60:
            say("no code after %d min: stopping the pod" % WAIT)
            ok, note = stop_pod()
            say("stop: %s (%s)" % ("done" if ok else "FAILED", note))
            time.sleep(600)


class H(BaseHTTPRequestHandler):
    key = os.environ.get("LAYOUT_KEY") or None
    port = 8000

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _authed(self):
        if not self.key:
            return True                 # /health says so; /code refuses
        return self.headers.get("Authorization") == "Bearer " + self.key

    def do_GET(self):
        if not self._authed():
            return self._send(401, {"error": "key"})
        if self.path.split("?")[0] != "/health":
            return self._send(404, {"error": "not found"})
        self._send(200, {"ok": True, "code": None, "loader": True,
                         "image": image_version(),
                         "reader": reader_model(), "key": bool(self.key),
                         "up_seconds": int(time.time() - STARTED),
                         "why": None if self.key else
                         "no key: set LAYOUT_KEY on the pod"})

    def do_POST(self):
        if not self._authed():
            return self._send(401, {"error": "key"})
        if self.path.split("?")[0] != "/code":
            return self._send(404, {"error": "not found: the loader "
                                    "takes /code only"})
        if not self.key:
            return self._send(403, {"error": "no key: set LAYOUT_KEY on "
                                    "the pod and Start again"})
        if _EXEC:
            return self._send(409, {"error": "code already received"})
        n = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(n)
        if data[:2] != b"\x1f\x8b":
            return self._send(400, {"error": "the body is not a gzip "
                                    "tarball"})
        try:
            sha = unpack(data)
        except Exception as e:
            say("code refused: %s" % str(e)[:200])
            return self._send(400, {"error": "bad tarball: %s" % str(e)[:200]})
        server = os.path.join(PANDO, "core", "dies", "box_server.py")
        if not os.path.exists(server):
            return self._send(400, {"error": "the tarball holds no "
                                    "core/dies/box_server.py"})
        say("code %s received (%d bytes): becoming box_server" % (
            sha[:12], len(data)))
        self._send(200, {"ok": True, "sha256": sha})
        _EXEC.append(sha)
        threading.Timer(0.5, become, [server]).start()


def become(server):
    """This process replaced by box_server on the same port; the
    listening socket is non-inheritable, so the port frees."""
    argv = [sys.executable, server, "--port", str(H.port),
            "--host", "0.0.0.0"]
    env = dict(os.environ)
    env.setdefault("BOX_ROOT", "/workspace/box"
                   if os.path.isdir("/workspace")
                   and os.access("/workspace", os.W_OK)
                   else os.path.join(PANDO, "box"))
    env.setdefault("PANDO_LAYOUT_MODEL", os.path.join(PANDO, "llm",
                                                       "layout-heron"))
    sys.stdout.flush()
    os.execve(sys.executable, argv, env)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    H.port = a.port
    say("image %s | pod %s | key %s | reader %s | waiting for code on "
        ":%d (%s)" % (image_version() or "unversioned",
                      os.environ.get("RUNPOD_POD_ID") or "none",
                      "set" if H.key else "UNSET: set LAYOUT_KEY on the pod",
                      reader_model() or "loading",
                      a.port, ("stop after %d min without it" % WAIT)
                      if WAIT else "no guard"))
    threading.Thread(target=guard, daemon=True).start()
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
