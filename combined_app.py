#!/usr/bin/env python3
"""
Combined Inference Suite
========================

Serves BOTH tools from a single process on a single port:

    /            -> landing page with a card for each tool
    /crop/...    -> Person Crop Balancer   (crop_balancer_app.py)
    /webapp/...  -> Inference Web App PPE+SM (web_app.py)

Instead of rewriting ~4500 lines of two mature Flask apps, we compose them
with Werkzeug's application dispatcher (the standard Flask "Application
Dispatching" pattern). Each app keeps its own routes, state and threads, but
now lives under a URL prefix on one port.

The only wrinkle is that both apps emit *root-absolute* URLs in their HTML/JS
(e.g. fetch("/api/status"), <img src="/stream.mjpg">, href="/balancer"). Those
would resolve against the site root and miss the mounted sub-app. Because every
one of those URLs is a quoted string literal that only ever appears inside the
served HTML page (never in the JSON API bodies, which carry the real filesystem
paths), we fix them with a narrow after_request pass that rewrites the known URL
prefixes to include the app's mount point. request.script_root gives us that
mount point, so the same hook works for either app.

Run:  python3 combined_app.py       (or: npm run dev)
"""
import os
import threading
import time
from pathlib import Path

from flask import Flask, redirect, request
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

import web_app
import crop_balancer_app

APP_DIR = Path(__file__).resolve().parent
LANDING_HTML = (APP_DIR / "scripts" / "landing.html").read_text(encoding="utf-8")
PORT = int(os.environ.get("PORT", "8000"))

# A token unique to this process. When the dev launcher restarts the server on a
# file change, the token changes and the injected live-reload script refreshes
# every open page automatically — so edits show up without a manual reload.
START_TOKEN = str(time.time())
LIVERELOAD_SNIPPET = (
    "<script>(function(){var t=null;function c(){"
    "fetch('/__livereload__',{cache:'no-store'}).then(function(r){return r.text();})"
    ".then(function(id){if(t===null){t=id;}else if(id!==t){location.reload();}})"
    ".catch(function(){});}setInterval(c,1000);c();})();</script>"
)


def _with_livereload(body):
    """Insert the live-reload poller just before </body> (or append it)."""
    if "</body>" in body:
        return body.replace("</body>", LIVERELOAD_SNIPPET + "</body>", 1)
    return body + LIVERELOAD_SNIPPET

# Quoted absolute-URL prefixes each sub-app emits in its HTML/JS. Each entry is a
# (quote-char-included) string that, when found in an HTML response, gets the
# app's mount prefix injected right after the opening quote. Listing them
# explicitly (rather than blindly rewriting every "/...") keeps us from touching
# unrelated strings. The union is safe to apply to both apps: a prefix an app
# never emits simply never matches.
URL_LITERAL_PREFIXES = (
    '"/api/', "'/api/",          # every fetch()/img.src API call in both apps
    '"/stream.mjpg', "'/stream.mjpg",   # web_app MJPEG <img> stream
    '"/download/', "'/download/",       # web_app "download video"
    '"/balancer', "'/balancer",         # crop app nav links
    '"/disagreement', "'/disagreement",
    '"/review', "'/review",
    'href="/"', "href='/'",             # crop app "home" links
)


def _rewrite_absolute_urls(app):
    """Register an after_request hook that prefixes the sub-app's own absolute
    URLs with its mount point, so links/fetches resolve to the mounted app."""

    @app.after_request
    def _prefix_urls(resp):
        root = request.script_root  # e.g. "/crop" or "/webapp"
        # Only HTML pages carry these link/fetch literals. Skip JSON (real file
        # paths live there), images, and the multipart MJPEG stream — rewriting
        # or buffering those would corrupt them or hang the stream.
        if not root or resp.mimetype != "text/html" or resp.direct_passthrough:
            return resp
        body = resp.get_data(as_text=True)
        for literal in URL_LITERAL_PREFIXES:
            if literal in body:
                quote = literal[len("href=")] if literal.startswith("href=") else literal[0]
                # Insert the mount root right after the opening quote.
                replacement = literal[: literal.index(quote) + 1] + root + literal[literal.index(quote) + 1 :]
                body = body.replace(literal, replacement)
        resp.set_data(_with_livereload(body))
        return resp

    return app


def _register_hooks():
    """Cheap: attach the URL-rewrite + live-reload after_request hooks. Safe to
    run in every process (the reloader parent and the serving child both import
    this module)."""
    _rewrite_absolute_urls(web_app.app)
    _rewrite_absolute_urls(crop_balancer_app.app)


def _start_workers():
    """Heavy: load settings, start the prerender thread, make output dirs. Runs
    ONLY in the process that actually serves requests — never in the reloader's
    watcher parent — so models/threads aren't spun up twice."""
    web_app.load_settings()
    web_app.configure_quiet_logging()
    threading.Thread(target=web_app.prerender_worker, daemon=True).start()
    crop_balancer_app.CROPS_ROOT.mkdir(parents=True, exist_ok=True)
    crop_balancer_app.REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    crop_balancer_app.DISAGREE_ROOT.mkdir(parents=True, exist_ok=True)


landing = Flask("inference_suite_landing")


@landing.get("/")
def _home():
    return LANDING_HTML


@landing.get("/home")
def _home_redirect():
    # Sub-apps link here to return to the app picker (see _with_livereload note:
    # "/" is rewritten to the mount root inside sub-apps, so they use "/home").
    return redirect("/")


@landing.get("/__livereload__")
def _livereload_token():
    return START_TOKEN, 200, {"Content-Type": "text/plain"}


@landing.after_request
def _landing_livereload(resp):
    if resp.mimetype == "text/html" and not resp.direct_passthrough:
        resp.set_data(_with_livereload(resp.get_data(as_text=True)))
    return resp


def build_application():
    _register_hooks()
    # Landing app handles "/"; the two tools are mounted under prefixes.
    return DispatcherMiddleware(
        landing,
        {
            "/webapp": web_app.app,
            "/crop": crop_balancer_app.app,
        },
    )


application = build_application()


def _open_browser_when_ready(url, delay_sec=1.0):
    if os.environ.get("NO_BROWSER"):
        return
    import time
    import webbrowser

    def _open():
        time.sleep(delay_sec)
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()


if __name__ == "__main__":
    # Auto-reload on any .py (or the landing template) change — the real
    # "npm run dev" behaviour. NO_RELOAD=1 disables it (e.g. for production).
    use_reload = os.environ.get("NO_RELOAD") != "1"
    # With the reloader, this script runs in two processes: a watcher parent and
    # the serving child (marked by WERKZEUG_RUN_MAIN). Do the heavy startup and
    # banner only in the child so nothing is loaded twice.
    serving = (not use_reload) or os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    if serving:
        _start_workers()
        url = f"http://127.0.0.1:{PORT}"
        print(f"\nInference Suite running on {url}   (auto-reloads on edits)")
        print(f"  Landing      {url}/")
        print(f"  Crop Balancer{url}/crop/")
        print(f"  Web App      {url}/webapp/\n")
        _open_browser_when_ready(url)
    run_simple(
        "0.0.0.0", PORT, application,
        threaded=True,
        use_reloader=use_reload,
        use_debugger=False,
        extra_files=[str(APP_DIR / "scripts" / "landing.html")],
    )
