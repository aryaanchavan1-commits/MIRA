"""Security boundary tests for consent-gated web access."""
from __future__ import annotations

import socket
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.websearch import fetch_page_text, is_public_http_url, validate_public_url, web_allowed


def _address(value: str):
    return (socket.AF_INET, socket.SOCK_STREAM, 6, "", (value, 80))


def test_web_url_rejects_private_and_non_http_targets():
    for url in ("file:///etc/passwd", "http://localhost/", "https://127.0.0.1/"):
        with patch("tools.websearch.socket.getaddrinfo", return_value=[_address("127.0.0.1")]):
            assert not is_public_http_url(url)
    try:
        validate_public_url("https://user:pass@example.com/")
    except ValueError:
        pass
    else:
        raise AssertionError("credential-bearing URLs must be rejected")


def test_web_url_accepts_public_dns_and_fetch_validates_before_network():
    with patch("tools.websearch.socket.getaddrinfo", return_value=[_address("93.184.216.34")]):
        assert is_public_http_url("https://example.com/article")
    with patch("tools.websearch.urllib.request.build_opener") as opener:
        try:
            fetch_page_text("http://127.0.0.1/private",
                            config={"offline": False}, allow_web=True)
        except ValueError:
            pass
        else:
            raise AssertionError("private fetch targets must be rejected")
    opener.assert_not_called()


def test_offline_mode_is_fail_closed_and_boolean_consent_is_strict():
    assert web_allowed({"offline": True, "web_search": {"enabled": True}}, True) is False
    assert web_allowed({"offline": False, "web_search": {"enabled": False}}, True) is True
    assert web_allowed({"offline": False, "web_search": {"enabled": True}}, False) is False
    try:
        web_allowed({"offline": False}, "true")
    except ValueError:
        pass
    else:
        raise AssertionError("non-boolean consent must be rejected")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")
