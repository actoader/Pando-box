#!/usr/bin/env python3
"""the loader - what the box image runs until Pando's code arrives
(thread H sitting 2, 2026-09-16; the autonomous box, thread V,
2026-09-19).

WHY: the image is the MACHINE (CUDA, vLLM, poppler, tesseract and its
packs, the layout weights) and nothing of Pando's own code, because
core/ changes every sitting and an image with the code inside must be
rebuilt and pushed before every night: the pet machine in a new coat.
So the image boots this loader on port 8000 and the loader becomes
box_server the moment the code is here: no version skew is possible,
the box always runs the Mac's own dies (PRINCIPLES.md rule 12).

WHERE THE CODE COMES FROM (2026-09-19, A: "we weren't even listening
to the pod"; the Mac used to drive the boot over the ocean in five
handshakes, and a slow image pull between two clocks killed a healthy
pod): the VOLUME. The Mac puts the tarball at BOX_ROOT/code/core-<sha12>
.tgz and then BOX_ROOT/code/CURRENT (JSON: sha256, file) before the
pod even exists; this loader looks at CURRENT every LOOK seconds,
takes the tarball whose sha256 checks, unpacks it and becomes
box_server. The Mac is never required to be present. POST /code stays
as the fast path for a machine of A's on the LAN.

THE HEARTBEAT FROM THE FIRST SECOND: BOX_ROOT/heartbeat/<pod>.json,
the same file box_server rewrites once it runs, written here every
LOOK seconds with {"box", "t", "loader": true, "phase", "image",
"reader", "code": null, "up_seconds"}; the phase is booting, reader
loading, waiting for code, unpacking. The Mac reads it through the
volume's door (core/dies/volume.py), so the app says what the box is
doing before any port is mapped and whether or not this web server is
ever reached.

WHAT (stdlib only, no Pando import: the code is not here yet):
    GET  /health   {"ok": true, "code": null, "loader": true, "phase",
                    "image", "reader": the model vLLM serves on 8001 or
                    null while it loads, "key": whether LAYOUT_KEY is
                    set, "up_seconds"}; the same Authorization: Bearer
                    <LAYOUT_KEY> rule as box_server (401 without it)
    POST /code     body: the tarball (core/ and catalog/clean/
                    wordlist*.txt, as pod.pack makes it) -> 200
                    {"ok", "sha256"}, unpacked under /pando, then this
                    process EXECS core/dies/box_server.py on the same
                    port (the listening socket is not inherited: the
                    port frees, box_server binds it a second later).
                    /pando/code.json records the sha256 and when;
                    box_server's /health and heartbeat show it as
                    "code", so the Mac can see what the box runs.
    anything else  404

THE GUARD: LAYOUT_KEY unset means the box has no key and the Mac
cannot have read one off the pod: /health says so and /code is
refused (403), so nothing runs unkeyed on a public port. The stop is
the honest one (the 30-minute "no code" clock of v2 is gone): nothing
to do, meaning no CURRENT on the volume AND an empty queue folder,
for BOX_IDLE_STOP minutes (20; 0 never) stops the pod, and so does
BOX_MAX_HOURS since boot (36; 0 never). A queue with books in it and
no code is the Mac's mistake, not the box's: a queued book proves the
code is coming, and the loader waits for it. Both guards read the
volume before they read the clock.

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
IDLE_STOP = int(os.environ.get("BOX_IDLE_STOP") or 20)   # minutes, 0 never
MAX_HOURS = float(os.environ.get("BOX_MAX_HOURS") or 36)  # 0 never
LOOK = float(os.environ.get("BOX_LOOK_S") or 15)          # the beat here
STARTED = time.time()
_EXEC = []                    # the sha taken, once; the one lock's flag
_LOCK = threading.Lock()
PHASE = ["booting"]


def say(msg):
    print("%s  loader: %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg),
          flush=True)


def root():
    """The queue's home: BOX_ROOT, else /workspace/box on a pod (the
    volume), else the code's own folder (a laptop's test)."""
    r = os.environ.get("BOX_ROOT")
    if not r and os.path.isdir("/workspace") and os.access("/workspace",
                                                           os.W_OK):
        r = "/workspace/box"
    return r or os.path.join(PANDO, "box")


def box_name():
    import socket
    return os.environ.get("BOX_NAME") or os.environ.get("RUNPOD_POD_ID") \
        or socket.gethostname()


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


# ---------------------------------------------------------- the volume

def current(where=None):
    """What CURRENT names on the volume: {"sha256", "file", ...} or
    None (no code put yet, or a half-written file: the next look)."""
    try:
        with open(os.path.join(where or root(), "code", "CURRENT")) as f:
            d = json.load(f)
        return d if isinstance(d, dict) and d.get("sha256") \
            and d.get("file") else None
    except (OSError, ValueError):
        return None


def take_from_volume(where=None):
    """The tarball CURRENT names, read off the volume and checked:
    (sha256, bytes), or (None, why) when it is not there yet or does
    not match (a tarball still arriving, or a CURRENT ahead of its
    file: the next look settles it)."""
    cur = current(where)
    if not cur:
        return None, "no CURRENT on the volume"
    p = os.path.join(where or root(), "code", os.path.basename(cur["file"]))
    try:
        with open(p, "rb") as f:
            data = f.read()
    except OSError:
        return None, "CURRENT names %s, not on the volume yet" % cur["file"]
    sha = hashlib.sha256(data).hexdigest()
    if sha != cur["sha256"]:
        return None, "%s does not match CURRENT's sha (%s vs %s)" % (
            cur["file"], sha[:12], cur["sha256"][:12])
    return sha, data


def queue_empty(where=None):
    """No job folder under queue/ (a folder with anything in it counts:
    a book arriving is a book)."""
    q = os.path.join(where or root(), "queue")
    try:
        return not any(os.listdir(os.path.join(q, d))
                       for d in os.listdir(q)
                       if os.path.isdir(os.path.join(q, d)))
    except OSError:
        return True


def guard_verdict(now, started, has_code, empty, idle_minutes=None,
                  max_hours=None):
    """The reason to stop the pod now, or None. Pure, for the test:
    the ceiling since boot, else nothing to do (no code on the volume
    and an empty queue) for idle_minutes."""
    idle_minutes = IDLE_STOP if idle_minutes is None else idle_minutes
    max_hours = MAX_HOURS if max_hours is None else max_hours
    if max_hours and now - started > max_hours * 3600:
        return "%.1f h since boot (the ceiling is %g h)" % (
            (now - started) / 3600, max_hours)
    if idle_minutes and not has_code and empty and \
            now - started > idle_minutes * 60:
        return "no code on the volume and nothing queued for %d min " \
               "(the idle stop is %d min)" % ((now - started) // 60,
                                              idle_minutes)
    return None


# -------------------------------------------------------- the heartbeat

def heartbeat_path(where=None):
    return os.path.join(where or root(), "heartbeat", box_name() + ".json")


def write_heartbeat(where=None):
    """This loader's state onto the volume, whole (a .part then a
    rename), at the file box_server will take over."""
    hb = {"box": box_name(), "t": time.time(), "beat_s": LOOK,
          "loader": True, "phase": PHASE[0], "image": image_version(),
          "reader": reader_model(), "code": None, "key": bool(H.key),
          "started": STARTED, "up_seconds": int(time.time() - STARTED),
          "watchdog": {"armed": bool(stopper()[0]),
                       "idle_stop_minutes": IDLE_STOP,
                       "max_hours": MAX_HOURS}}
    p = heartbeat_path(where)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + ".part", "w", encoding="utf-8") as f:
        json.dump(hb, f)
    os.replace(p + ".part", p)
    return hb


def beat(once=False):
    """Every LOOK seconds: the phase, the heartbeat, a look at CURRENT
    (the code taken when it is there), the guard."""
    while True:
        try:
            if not _EXEC:
                PHASE[0] = "reader loading" if reader_model() is None \
                    else "waiting for code"
            try:
                write_heartbeat()
            except OSError as e:
                say("heartbeat not written: %s" % str(e)[:120])
            if not _EXEC:
                sha, data = take_from_volume()
                if sha:
                    become_with(data, sha, "the volume")
                else:
                    why = guard_verdict(time.time(), STARTED, False,
                                        queue_empty())
                    if why:
                        say("stopping the pod: %s" % why)
                        ok, note = stop_pod()
                        say("stop: %s (%s)" % ("done" if ok else "FAILED",
                                               note))
                        time.sleep(600)
        except Exception as e:  # noqa: BLE001 - the beat never dies
            say("beat failed: %s" % str(e)[:160])
        if once:
            return
        time.sleep(LOOK)


def stopper():
    pod = os.environ.get("RUNPOD_POD_ID")
    if not pod:
        return None, "not a RunPod pod (RUNPOD_POD_ID unset)"
    if os.environ.get("RUNPOD_API_KEY"):
        return ["api", pod], "api v2 stop with the pod's own key"
    if shutil.which("runpodctl"):
        return ["runpodctl", "stop", "pod", pod], "runpodctl"
    return None, "neither RUNPOD_API_KEY nor runpodctl here"


def stop_pod():
    """(ok, note): this pod stopped over RunPod's API with the
    pod-scoped key RunPod sets in every pod, runpodctl as the second
    arm; the same two arms as box_server's watchdog."""
    how, note = stopper()
    if not how:
        return False, note
    pod = how[-1]
    if how[0] == "api":
        try:
            req = urllib.request.Request(
                "https://api.runpod.io/v2/pods/%s/action" % pod,
                data=json.dumps({"action": "stop"}).encode(),
                method="POST", headers={
                    "Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"],
                    "Content-Type": "application/json",
                    "User-Agent": "pando-box/1"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return True, "api v2 stop -> %d" % r.status
        except Exception as e:
            note = "api v2 stop failed: %s" % str(e)[:120]
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


# ------------------------------------------------------------ the exec

def become_with(data, sha, how):
    """The tarball unpacked and this process handed to box_server, once
    (the volume's look and a POST share the lock). -> True when this
    call is the one that did it."""
    with _LOCK:
        if _EXEC:
            return False
        PHASE[0] = "unpacking"
        try:
            write_heartbeat()
        except OSError:
            pass
        try:
            got = unpack(data)
        except Exception as e:
            say("code from %s refused: %s" % (how, str(e)[:200]))
            PHASE[0] = "waiting for code"
            raise
        server = os.path.join(PANDO, "core", "dies", "box_server.py")
        if not os.path.exists(server):
            PHASE[0] = "waiting for code"
            raise ValueError("the tarball holds no core/dies/box_server.py")
        say("code %s taken from %s (%d bytes): becoming box_server" % (
            got[:12], how, len(data)))
        _EXEC.append(got)
    threading.Timer(0.5, become, [server]).start()
    return True


def become(server):
    """This process replaced by box_server on the same port; the
    listening socket is non-inheritable, so the port frees."""
    argv = [sys.executable, server, "--port", str(H.port),
            "--host", "0.0.0.0"]
    env = dict(os.environ)
    env.setdefault("BOX_ROOT", root())
    env.setdefault("PANDO_LAYOUT_MODEL", os.path.join(PANDO, "models",
                                                       "layout-heron"))
    sys.stdout.flush()
    os.execve(sys.executable, argv, env)


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
            # a job asked of the loader is not lost, the box is booting:
            # 503, never 404 (the Mac closes a ticket as lost on a 404)
            return self._send(503, {"error": "booting: the code is not on "
                                    "the box yet"})
        self._send(200, {"ok": True, "code": None, "loader": True,
                         "phase": PHASE[0], "image": image_version(),
                         "reader": reader_model(), "key": bool(self.key),
                         "up_seconds": int(time.time() - STARTED),
                         "why": None if self.key else
                         "no key: set LAYOUT_KEY on the pod"})

    def do_POST(self):
        if not self._authed():
            return self._send(401, {"error": "key"})
        if self.path.split("?")[0] != "/code":
            return self._send(503, {"error": "booting: the loader takes "
                                    "/code only; the code is not on the "
                                    "box yet"})
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
        sha = hashlib.sha256(data).hexdigest()
        try:
            if not become_with(data, sha, "POST /code"):
                return self._send(409, {"error": "code already received"})
        except Exception as e:
            return self._send(400, {"error": "bad tarball: %s" % str(e)[:200]})
        self._send(200, {"ok": True, "sha256": sha})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    H.port = a.port
    say("image %s | pod %s | key %s | reader %s | queue root %s | code off "
        "the volume every %gs (or POST /code on :%d) | idle stop %s, "
        "ceiling %s" % (
            image_version() or "unversioned",
            os.environ.get("RUNPOD_POD_ID") or "none",
            "set" if H.key else "UNSET: set LAYOUT_KEY on the pod",
            reader_model() or "loading", root(), LOOK, a.port,
            ("%d min with nothing to do" % IDLE_STOP) if IDLE_STOP
            else "never", ("%g h" % MAX_HOURS) if MAX_HOURS else "none"))
    try:
        write_heartbeat()                  # "booting", the first second
    except OSError as e:
        say("heartbeat not written: %s" % str(e)[:120])
    threading.Thread(target=beat, daemon=True).start()
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
