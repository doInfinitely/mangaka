"""ExHentai HTTP client with auth, retry, rate limiting, and HTML parsing."""

from __future__ import annotations

import io
import logging
import os
import random
import re
import time
from urllib.parse import quote

import requests
from PIL import Image

from .models import GalleryMetadata, parse_gallery_url

logger = logging.getLogger(__name__)

_BASE = "https://exhentai.org"

# ---------------------------------------------------------------------------
# Header pools (anti-detection)
# ---------------------------------------------------------------------------
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) "
    "Gecko/20100101 Firefox/125.0",
]


def _random_headers() -> dict[str, str]:
    return {
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": _BASE + "/",
    }


def _image_headers() -> dict[str, str]:
    return {
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept": "image/avif,image/webp,image/png,image/svg+xml,image/*;q=0.8,*/*;q=0.5",
        "Referer": _BASE + "/",
    }


class ExHentaiClient:
    """Stateful HTTP client for ExHentai with cookie auth and rate limiting."""

    def __init__(
        self,
        cookies: str | None = None,
        rate_limit: float = 2.0,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.timeout = timeout
        self._last_request: float = 0

        self.session = requests.Session()
        self._setup_cookies(cookies or os.environ.get("EXHENTAI_COOKIES", ""))

    def _setup_cookies(self, cookie_str: str) -> None:
        """Parse ``ipb_member_id=xxx; ipb_pass_hash=yyy; igneous=zzz`` and set on session."""
        if not cookie_str:
            raise ValueError(
                "ExHentai cookies required. Set EXHENTAI_COOKIES env var "
                "with 'ipb_member_id=xxx; ipb_pass_hash=yyy; igneous=zzz'"
            )
        for part in cookie_str.split(";"):
            part = part.strip()
            if "=" not in part:
                continue
            name, value = part.split("=", 1)
            self.session.cookies.set(
                name.strip(), value.strip(), domain=".exhentai.org"
            )

    def _wait_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

    def _fetch(self, url: str, headers: dict[str, str] | None = None) -> str:
        """GET with retry, exponential backoff, and sadpanda detection."""
        hdrs = headers or _random_headers()
        for attempt in range(1, self.max_retries + 1):
            self._wait_rate_limit()
            self._last_request = time.monotonic()
            try:
                resp = self.session.get(url, headers=hdrs, timeout=self.timeout)
                resp.raise_for_status()
            except requests.RequestException as exc:
                wait = 2 ** attempt + random.random()
                logger.warning("Fetch %s attempt %d failed: %s (retry in %.1fs)", url, attempt, exc, wait)
                time.sleep(wait)
                continue
            text = resp.text
            if "This gallery has been removed" in text:
                raise ValueError(f"Gallery removed: {url}")
            if len(text) < 100 and ("sadpanda" in text.lower() or "<html>" not in text.lower()):
                raise PermissionError(
                    "Got sadpanda — cookies are invalid or expired"
                )
            return text
        raise ConnectionError(f"Failed to fetch {url} after {self.max_retries} attempts")

    def _fetch_bytes(self, url: str) -> bytes:
        """GET raw bytes (for images)."""
        hdrs = _image_headers()
        for attempt in range(1, self.max_retries + 1):
            self._wait_rate_limit()
            self._last_request = time.monotonic()
            try:
                resp = self.session.get(url, headers=hdrs, timeout=self.timeout)
                resp.raise_for_status()
                return resp.content
            except requests.RequestException as exc:
                wait = 2 ** attempt + random.random()
                logger.warning("Image fetch %s attempt %d failed: %s", url, attempt, exc)
                time.sleep(wait)
        raise ConnectionError(f"Failed to fetch image {url} after {self.max_retries} attempts")

    # -- public API -------------------------------------------------------------

    def fetch_metadata(self, gallery_url: str) -> GalleryMetadata:
        """Scrape gallery metadata from the main gallery page."""
        gallery_id, token = parse_gallery_url(gallery_url)
        url = f"{_BASE}/g/{gallery_id}/{token}/"
        html = self._fetch(url)

        # Title: <h1 id="gn">...</h1>
        title_m = re.search(r'<h1 id="gn">([^<]+)</h1>', html)
        title = title_m.group(1).strip() if title_m else ""

        # Artist tags: id="ta_artist:name"
        artists = re.findall(r'id="ta_artist:([^"]+)"', html)
        # URL-encoded names → human-readable
        artists = [a.replace("+", " ") for a in artists]

        # Language tag
        lang_m = re.search(r'id="ta_language:([^"]+)"', html)
        language = lang_m.group(1).replace("+", " ") if lang_m else ""

        # Page count: "123 pages"
        pages_m = re.search(r"(\d+) pages", html)
        page_count = int(pages_m.group(1)) if pages_m else 0

        return GalleryMetadata(
            gallery_id=gallery_id,
            token=token,
            title=title,
            artists=artists,
            language=language,
            page_count=page_count,
            url=gallery_url,
        )

    def get_page_urls(self, gallery_url: str) -> list[str]:
        """Return all ``/s/`` page URLs for a gallery (handles multi-page listings)."""
        gallery_id, token = parse_gallery_url(gallery_url)
        base = f"{_BASE}/g/{gallery_id}/{token}/"

        page_urls: list[str] = []
        listing_page = 0

        while True:
            url = base if listing_page == 0 else f"{base}?p={listing_page}"
            html = self._fetch(url)

            # Page links: href="https://exhentai.org/s/hash/galleryid-pagenum"
            links = re.findall(
                rf'href="(https://exhentai\.org/s/[a-f0-9]+/{gallery_id}-\d+)"',
                html,
            )
            if not links:
                break

            # Deduplicate while preserving order
            for link in links:
                if link not in page_urls:
                    page_urls.append(link)

            # Check for next listing page
            if f"?p={listing_page + 1}" in html:
                listing_page += 1
            else:
                break

        return page_urls

    def download_image(self, page_url: str) -> Image.Image:
        """Fetch a single page image from its ``/s/`` URL."""
        html = self._fetch(page_url)

        # Image URL: <img id="img" src="https://...">
        img_m = re.search(r'id="img"\s+src="([^"]+)"', html)
        if not img_m:
            # Try alternate pattern
            img_m = re.search(r'src="([^"]+)"\s+id="img"', html)
        if not img_m:
            raise ValueError(f"Could not find image URL on page {page_url}")

        img_url = img_m.group(1)
        data = self._fetch_bytes(img_url)
        return Image.open(io.BytesIO(data)).convert("RGB")

    def search_artist_galleries(
        self, artist_name: str, max_results: int = 20
    ) -> list[str]:
        """Search ExHentai for galleries by artist tag, return gallery URLs."""
        encoded = quote(artist_name, safe="")
        tag_url = f"{_BASE}/tag/artist:{encoded}"
        gallery_urls: list[str] = []
        page = 0

        while len(gallery_urls) < max_results:
            url = tag_url if page == 0 else f"{tag_url}/{page}"
            html = self._fetch(url)

            # Gallery links: href="https://exhentai.org/g/id/token/"
            links = re.findall(
                r'href="(https://exhentai\.org/g/\d+/[a-f0-9]+/?)"', html
            )
            if not links:
                break

            for link in links:
                canonical = link.rstrip("/") + "/"
                if canonical not in gallery_urls:
                    gallery_urls.append(canonical)
                    if len(gallery_urls) >= max_results:
                        break

            # Check for next page
            if f"/{page + 1}" in html and len(links) > 0:
                page += 1
            else:
                break

        return gallery_urls
