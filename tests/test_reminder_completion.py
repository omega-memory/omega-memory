"""A completed task may dismiss the reminder it completes, and nothing else.

Bug-team audit 2026-09-29, finding B6: reminder auto-dismiss never ran (it
called a store method that does not exist and swallowed the AttributeError).
Its rules, had they run, were far too loose: any decision or completion in any
project at cosine 0.40, or any three shared words, would dismiss a pending
reminder and also retire checkpoints. These tests pin the narrow version.
"""

import math

import pytest

from omega.sqlite_store import EMBEDDING_DIM

REMINDER = "Upgrade Node to version 22 on the build server"
COMPLETION = "Upgraded Node to version 22 on the build server and verified the pipeline end to end."
PLAN = "Decided to upgrade Node to version 22 on the build server next sprint, after the freeze."


def _vec(seed: float) -> list:
    raw = [math.sin(seed * (i + 1)) for i in range(EMBEDDING_DIM)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def _near(base: list, amount: float = 0.3) -> list:
    noise = _vec(99.0)
    mixed = [b * (1 - amount) + n * amount for b, n in zip(base, noise)]
    norm = math.sqrt(sum(x * x for x in mixed))
    return [x / norm for x in mixed]


@pytest.fixture
def bridge(_reset_bridge, monkeypatch):
    """Deterministic embeddings: the reminder and its completion are near neighbours."""
    import omega.bridge as bridge_module
    import omega.embedding as embedding

    near_pairs = {
        REMINDER: _vec(3.0),
        COMPLETION: _near(_vec(3.0)),
        PLAN: _near(_vec(3.0), 0.25),
        "Rotate the staging database credentials": _vec(5.0),
        "Rotated the production database credentials and updated the vault entries for all services.":
            _near(_vec(5.0)),
    }

    def fake(text, *args, **kwargs):
        for prefix, vector in near_pairs.items():
            if text.startswith(prefix):
                return vector
        return _vec(float(sum(map(ord, text[:50])) % 997) + 7.0)

    monkeypatch.setattr(embedding, "generate_embedding", fake)
    return bridge_module


def _status(bridge, reminder_id):
    node = bridge._get_store().get_node(reminder_id, track_access=False)
    return node.metadata.get("reminder_status")


def _remind(bridge, text, project="/work/alpha"):
    return bridge.create_reminder(text, "2d", project=project)["reminder_id"]


def test_matching_completion_dismisses_and_reports(bridge):
    reminder_id = _remind(bridge, REMINDER)

    result = bridge.store(COMPLETION, event_type="task_completion", project="/work/alpha")

    assert _status(bridge, reminder_id) == "dismissed"
    assert reminder_id in result
    node = bridge._get_store().get_node(reminder_id, track_access=False)
    assert node.metadata["dismissed_by"] == result.split()[1]
    assert not node.metadata.get("superseded")  # dismissed, like a manual dismissal


def test_completion_in_another_project_leaves_it_pending(bridge):
    reminder_id = _remind(bridge, REMINDER, project="/work/alpha")
    bridge.store(COMPLETION, event_type="task_completion", project="/work/beta")
    assert _status(bridge, reminder_id) == "pending"


def test_a_decision_is_not_a_completion(bridge):
    reminder_id = _remind(bridge, REMINDER)
    bridge.store(PLAN, event_type="decision", project="/work/alpha")
    assert _status(bridge, reminder_id) == "pending"


def test_hook_captured_completion_never_dismisses(bridge):
    reminder_id = _remind(bridge, REMINDER)
    bridge.auto_capture(
        content=COMPLETION, event_type="task_completion",
        metadata={"source": "auto_capture_hook"}, project="/work/alpha",
    )
    assert _status(bridge, reminder_id) == "pending"


def test_similar_but_different_task_stays_pending(bridge):
    """Near embeddings alone are not enough: staging vs production credentials."""
    reminder_id = _remind(bridge, "Rotate the staging database credentials")
    bridge.store(
        "Rotated the production database credentials and updated the vault entries for all services.",
        event_type="task_completion", project="/work/alpha",
    )
    assert _status(bridge, reminder_id) == "pending"


def test_unrelated_completion_dismisses_nothing(bridge):
    ids = [
        _remind(bridge, text)
        for text in (
            "Renew the TLS certificate for api.example.com",
            "Email the accountant about Q3 invoices",
            "Review the pull request for the search indexer",
            REMINDER,
        )
    ]
    bridge.store(
        "Finished migrating the product catalog search from Elasticsearch to Postgres full-text search.",
        event_type="task_completion", project="/work/alpha",
    )
    assert [_status(bridge, i) for i in ids] == ["pending"] * 4
