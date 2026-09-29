"""The Stop hook sends nothing anywhere (audit finding G3).

session_stop.py used to post session usage (project paths, commits, costs)
to whatever SUPABASE_URL and service key it found in the environment or in
~/.omega/secrets.json. The shipped wiring never ran it, but an install that
wired the script directly would have. The upload is gone.
"""
import inspect

from omega.hooks import session_stop


def test_session_stop_has_no_network_upload():
    source = inspect.getsource(session_stop)

    assert "_capture_usage_to_supabase" not in source
    assert "urllib" not in source
    assert "SUPABASE" not in source
