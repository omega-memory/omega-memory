"""The UserPromptSubmit capture hook stores statements, never questions or secrets.

Bug-team audit 2026-09-29, finding B2: the hook saved questions ("let's use
Redis? actually wait, can you ...") and a live API key verbatim as decisions,
and those captures could then retire real decisions.
"""

import pytest

from omega.hooks import auto_capture as prompt_hook
from omega.redaction import redact_secrets

pytestmark = pytest.mark.usefixtures("_reset_bridge")


@pytest.fixture(autouse=True)
def _fresh_capture_counts():
    prompt_hook._captures_by_session.clear()
    yield
    prompt_hook._captures_by_session.clear()


def _captured(event_type=None):
    from omega.bridge import _get_store

    rows = _get_store()._conn.execute("SELECT event_type, content FROM memories").fetchall()
    return [content for et, content in rows if event_type in (None, et)]


def _submit(prompt):
    prompt_hook.run({"prompt": prompt, "session_id": "s1", "cwd": "/work/shop"})


# A fake key assembled at runtime so no secret-shaped literal sits in the repo.
FAKE_STRIPE_KEY = "sk_" + "live_" + "51Hxyz" + "0123456789" + "abcdefABCDEF"


class TestQuestionsAreNotCaptured:
    @pytest.mark.parametrize(
        "prompt",
        [
            "let's use Redis? actually wait, can you first show me how the cache module is wired",
            "Can you remember that I have a dentist appointment at 3pm, then look at the failing test",
            "we should use Postgres; hmm, what do you think? is MySQL better for this?",
            "We should switch to pnpm for the monorepo so installs are faster and disk use drops, right?",
        ],
    )
    def test_question_prompt_stores_nothing(self, prompt):
        _submit(prompt)
        assert _captured() == []

    def test_statement_followed_by_a_question_keeps_the_statement(self):
        _submit(
            "Let's go with Postgres for the orders service because we need row-level locking "
            "and mature migrations. Any concerns before I start?"
        )
        [content] = _captured("decision")
        assert "Let's go with Postgres for the orders service because" in content
        assert "Any concerns" not in content

    @pytest.mark.parametrize(
        "prompt",
        [
            "Do not deploy on Fridays from now on; releases wait until Monday morning after standup.",
            "Will do, and from now on run the full suite before saying something is fixed or done.",
        ],
    )
    def test_statement_opening_with_an_auxiliary_is_captured(self, prompt):
        """Only an auxiliary followed by a subject ("can you", "is it") asks."""
        _submit(prompt)
        assert len(_captured("decision")) == 1

    def test_plain_decision_is_still_captured(self):
        _submit("from now on please run the full test suite before you tell me that something is done or fixed")
        [content] = _captured("decision")
        assert "run the full test suite" in content

    def test_lesson_is_still_captured(self):
        _submit(
            "turns out the flaky test was due to a timezone assumption in tests/test_dates.py, "
            "the fix was to use UTC everywhere"
        )
        assert len(_captured("lesson_learned")) == 1


class TestSecretsAreRedacted:
    def test_api_key_in_prompt_is_not_stored(self):
        _submit(f"remember that the staging API key is {FAKE_STRIPE_KEY} and do not share it")

        [content] = _captured("decision")
        assert FAKE_STRIPE_KEY not in content
        assert "[REDACTED]" in content

    def test_hook_redaction_covers_every_hook_source(self):
        """Redaction lives in bridge.auto_capture, so assistant/session hooks get it too."""
        from omega.bridge import auto_capture

        auto_capture(
            content=f"Lesson: the deploy failed until we exported STRIPE_KEY={FAKE_STRIPE_KEY} in CI.",
            event_type="lesson_learned",
            metadata={"source": "auto_assistant_capture"},
            project="/work/shop",
        )
        [content] = _captured("lesson_learned")
        assert FAKE_STRIPE_KEY not in content

    def test_direct_store_is_left_verbatim(self):
        """An explicit omega_store is the caller's deliberate choice."""
        from omega.bridge import store

        store(f"Test fixture key for the local Stripe mock: {FAKE_STRIPE_KEY}", event_type="memory",
              project="/work/shop")
        assert any(FAKE_STRIPE_KEY in c for c in _captured("memory"))


class TestRedactSecrets:
    @pytest.mark.parametrize(
        "secret",
        [
            FAKE_STRIPE_KEY,
            "AKIA" + "ABCDEFGHIJKLMNOP",
            "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8",
            "sk-ant-" + "api03-" + "a1B2c3D4e5F6g7H8i9J0k1L2",
            "xoxb-" + "123456789012-" + "a1B2c3D4e5F6g7H8i9J0",
            "eyJhbGciOiJIUzI1NiJ9" + "." + "eyJzdWIiOiIxMjM0NTY3ODkwIn0" + "." + "a1B2c3D4e5F6g7H8i9J0",
            "Zm9vYmFyQmF6UXV4" + "MTIzNDU2Nzg5MGFiY2RlZg",  # generic mixed-case base64
        ],
    )
    def test_secret_shapes_are_redacted(self, secret):
        text, count = redact_secrets(f"use {secret} for the call")
        assert secret not in text
        assert count == 1

    @pytest.mark.parametrize(
        "text",
        [
            "commit a54af6742f0c1b2d3e4f5a6b7c8d9e0f1a2b3c4d fixed it",  # git SHA
            "request 3f2b8c1e-9d4a-4c6b-8e2f-1a2b3c4d5e6f failed",  # UUID
            "the token is valid until midnight",
            "set the password rotation to 90 days",
            # Identifiers, filenames and setting names in captured lessons.
            "the fix was renaming getOAuth2AccessTokenFromCache to a plain getter",
            "build emitted main.a1B2c3D4e5F6g7H8i9J0k1L2.js twice",
            "set password = settings.DB_PASSWORD_PATH before the migration",
            "the api key is STRIPE_SECRET_KEY in the environment",
        ],
    )
    def test_ordinary_text_is_left_alone(self, text):
        assert redact_secrets(text) == (text, 0)

    def test_password_assignment_value_is_redacted(self):
        text, count = redact_secrets("the db password is hunter22 on staging")
        assert "hunter22" not in text
        assert count == 1

    def test_url_credentials_are_redacted(self):
        url = "postgres" + "://app:" + "s3cretPass" + "@db.internal:5432/orders"
        text, _ = redact_secrets(f"connect with {url}")
        assert "s3cretPass" not in text
        assert "db.internal:5432/orders" in text
