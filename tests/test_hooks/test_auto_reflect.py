"""Test auto-reflect at session stop."""
from unittest.mock import patch, MagicMock


def test_auto_reflect_finds_contradictions():
    """Auto-reflect should call find_contradictions and store results."""
    from omega.hooks.session_stop import _auto_reflect

    mock_store = MagicMock()
    mock_result = {
        "topic": "recent decisions",
        "memories_analyzed": 15,
        "contradictions": [
            {
                "memory_a_id": "mem-aaa",
                "memory_a_content": "Use MySQL",
                "memory_b_id": "mem-bbb",
                "memory_b_content": "Use PostgreSQL",
                "confidence": 0.85,
                "signals": ["negation"],
                "reason": "Conflicting database choices",
            }
        ],
    }

    with patch("omega.hooks.session_stop.find_contradictions", return_value=mock_result) as mock_find, \
         patch("omega.hooks.session_stop._get_reflect_store", return_value=mock_store):

        result = _auto_reflect("test-session", "/test/project")

        mock_find.assert_called_once()
        assert result["contradictions_found"] == 1


def test_auto_reflect_no_contradictions():
    """When no contradictions found, result shows 0."""
    from omega.hooks.session_stop import _auto_reflect

    mock_store = MagicMock()
    mock_result = {
        "topic": "recent decisions",
        "memories_analyzed": 10,
        "contradictions": [],
    }

    with patch("omega.hooks.session_stop.find_contradictions", return_value=mock_result), \
         patch("omega.hooks.session_stop._get_reflect_store", return_value=mock_store):

        result = _auto_reflect("test-session", "/test/project")

        assert result["contradictions_found"] == 0


def test_auto_reflect_handles_import_error():
    """Auto-reflect should handle missing modules gracefully."""
    from omega.hooks.session_stop import _auto_reflect

    with patch("omega.hooks.session_stop._get_reflect_store", side_effect=ImportError("no module")):
        result = _auto_reflect("test-session", "/test/project")
        assert result["contradictions_found"] == 0
