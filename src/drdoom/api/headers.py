"""Response headers that limit what a page from this service can do in a browser.

The dashboard's defence against model text becoming script is the sanitiser: every value
from the api is set as text or passed through DOMPurify (see ``web/index.html``). The
Content Security Policy is the layer behind it. If markup ever got past the sanitiser, the
browser would still refuse to run inline script or script from anywhere but this origin
and the two pinned CDN files, to connect anywhere but this origin, and to load an image
from elsewhere, which is how injected markup would carry data out. The page cannot be
framed, so the approve button cannot be overlaid by another site.

FastAPI's interactive docs load their own scripts from another CDN and run inline code,
so they are left out of the policy. They render the api's own schema, not model output.
"""

from __future__ import annotations

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# The only scripts from elsewhere. web/index.html pins each by hash as well.
CDN_SCRIPTS = (
    "https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/dompurify/3.1.6/purify.min.js",
)

CONTENT_SECURITY_POLICY = "; ".join(
    [
        "default-src 'none'",
        "script-src 'self' " + " ".join(CDN_SCRIPTS),
        "style-src 'self'",
        "img-src 'self'",
        "connect-src 'self'",
        "base-uri 'none'",
        "form-action 'none'",
        "frame-ancestors 'none'",
    ]
)


class SecurityHeaders:
    """Add the policy, and two headers that cost nothing, to every http response.

    Written as plain ASGI rather than with ``BaseHTTPMiddleware`` so the investigation
    stream passes through untouched.
    """

    def __init__(self, app: ASGIApp, exempt: frozenset[str] = frozenset()) -> None:
        self.app = app
        self.exempt = exempt

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        with_policy = scope["path"] not in self.exempt

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                message.setdefault("headers", [])
                headers = MutableHeaders(scope=message)
                if with_policy:
                    headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
                headers.setdefault("X-Content-Type-Options", "nosniff")
                headers.setdefault("Referrer-Policy", "no-referrer")
            await send(message)

        await self.app(scope, receive, send_with_headers)
