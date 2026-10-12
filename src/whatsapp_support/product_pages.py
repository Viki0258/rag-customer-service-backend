"""Bounded, credential-free manufacturer page reader with pinned public DNS."""
import asyncio
import http.client
from html.parser import HTMLParser
import ipaddress
import socket
import ssl
from urllib.parse import urlsplit, urlunsplit


DOMAINS = ('samsung.com', 'lg.com', 'sony.com', 'sony.co.uk', 'sony.com.cn',
           'tcl.com', 'tcl.com.cn', 'hisense.com', 'hisense.com.cn', 'mi.com')
MAX_PAGE_BYTES = 3_000_000


def manufacturer_url(url):
    try:
        p = urlsplit(url)
        host = p.hostname or ''
        if (len(url) > 2048 or p.scheme != 'https' or p.username or p.password
                or p.port not in (None, 443) or '\\' in url
                or any(ord(c) < 32 for c in url)
                or not any(host == d or host.endswith('.' + d) for d in DOMAINS)):
            return None
        return urlunsplit((p.scheme, p.netloc, p.path, p.query, ''))
    except (ValueError, TypeError):
        return None


class PageText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'noscript', 'svg', 'nav', 'header', 'footer'):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'noscript', 'svg', 'nav', 'header', 'footer'):
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden and data.strip():
            self.parts.append(' '.join(data.split()))


def public_addresses(host):
    addresses = list(dict.fromkeys(x[4][0] for x in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)))
    if not addresses or any(not ipaddress.ip_address(ip).is_global for ip in addresses):
        raise ValueError('Non-public page destination')
    return addresses


class PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, host, address):
        super().__init__(host, timeout=8, context=ssl.create_default_context())
        self.address = address

    def connect(self):
        # Connect to the validated IP, but verify the certificate for the original host.
        sock = socket.create_connection((self.address, 443), timeout=self.timeout)
        try:
            self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
        except Exception:
            sock.close()
            raise


def read_page(url):
    normalized = manufacturer_url(url)
    if not normalized:
        raise ValueError('Unsupported manufacturer URL')
    p = urlsplit(normalized)
    connection = PinnedHTTPS(p.hostname, public_addresses(p.hostname)[0])
    try:
        connection.request('GET', p.path + ('?' + p.query if p.query else '') or '/',
                           headers={'User-Agent': 'CustomerServiceResearch/1.0',
                                    'Accept': 'text/html', 'Accept-Encoding': 'identity'})
        response = connection.getresponse()
        # No redirects, cookies, browser credentials, or model-selected fetch tools.
        if response.status != 200 or 'text/html' not in response.getheader('Content-Type', ''):
            raise ValueError('Page unavailable')
        raw = response.read(MAX_PAGE_BYTES + 1)
        if len(raw) > MAX_PAGE_BYTES:
            raise ValueError('Page too large')
        parser = PageText()
        parser.feed(raw.decode(response.headers.get_content_charset() or 'utf-8', errors='replace'))
        text = '\n'.join(parser.parts)
        if len(text) < 100:
            raise ValueError('No usable page text')
        return text[:24000]
    finally:
        connection.close()


async def fetch_product_page(url):
    return await asyncio.wait_for(asyncio.to_thread(read_page, url), timeout=15)
