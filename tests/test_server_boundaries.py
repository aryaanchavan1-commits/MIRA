"""HTTP boundary tests that do not require a running server."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_websearch_requires_explicit_consent_when_global_search_is_off():
    import server

    original_context = server._ctx
    server._ctx = type("Context", (), {
        "cfg": {"offline": False, "web_search": {"enabled": False}},
    })()
    try:
        with patch.object(server, "web_search_fn") as search:
            result = server.websearch({"query": "private query"})
            assert result["results"] == []
            search.assert_not_called()
            server.websearch({"query": "private query", "allow_web": True})
            search.assert_called_once()
    finally:
        server._ctx = original_context


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")
