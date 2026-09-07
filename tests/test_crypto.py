"""Tests for omega.crypto — encryption at rest, key management, passthrough."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omega.crypto import (
    is_enabled,
    encrypt,
    decrypt,
    encrypt_line,
    decrypt_line,
    reset_crypto_state,
    _get_or_create_key,
    _key_path,
)


@pytest.fixture(autouse=True)
def _reset_crypto():
    """Reset crypto state before and after each test."""
    reset_crypto_state()
    yield
    reset_crypto_state()


# ============================================================================
# is_enabled
# ============================================================================


class TestIsEnabled:
    def test_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("OMEGA_ENCRYPT", raising=False)
        assert is_enabled() is True

    def test_enabled_with_1(self, monkeypatch):
        monkeypatch.setenv("OMEGA_ENCRYPT", "1")
        assert is_enabled() is True

    def test_enabled_with_true(self, monkeypatch):
        monkeypatch.setenv("OMEGA_ENCRYPT", "true")
        assert is_enabled() is True

    def test_enabled_with_yes(self, monkeypatch):
        monkeypatch.setenv("OMEGA_ENCRYPT", "yes")
        assert is_enabled() is True

    def test_disabled_with_0(self, monkeypatch):
        monkeypatch.setenv("OMEGA_ENCRYPT", "0")
        assert is_enabled() is False

    def test_disabled_with_false(self, monkeypatch):
        monkeypatch.setenv("OMEGA_ENCRYPT", "false")
        assert is_enabled() is False

    def test_disabled_with_no(self, monkeypatch):
        monkeypatch.setenv("OMEGA_ENCRYPT", "no")
        assert is_enabled() is False

    def test_enabled_with_empty(self, monkeypatch):
        monkeypatch.setenv("OMEGA_ENCRYPT", "")
        assert is_enabled() is True


# ============================================================================
# Plaintext passthrough (encryption disabled)
# ============================================================================


class TestPlaintextPassthrough:
    def test_encrypt_returns_plaintext_when_disabled(self, monkeypatch):
        monkeypatch.setenv("OMEGA_ENCRYPT", "0")
        assert encrypt("hello world") == "hello world"

    def test_decrypt_returns_plaintext_without_prefix(self):
        assert decrypt("just plain text") == "just plain text"

    def test_encrypt_line_returns_line_when_disabled(self, monkeypatch):
        monkeypatch.setenv("OMEGA_ENCRYPT", "0")
        line = '{"content": "test"}'
        assert encrypt_line(line) == line

    def test_decrypt_line_strips_whitespace(self):
        assert decrypt_line("  plain text  \n") == "plain text"


# ============================================================================
# Key management
# ============================================================================


class TestKeyManagement:
    def test_key_created_on_first_use(self, tmp_omega_dir):
        key_path = tmp_omega_dir / ".key"
        assert not key_path.exists()
        key = _get_or_create_key()
        assert key_path.exists()
        assert len(key) > 0

    def test_key_reused_on_second_call(self, tmp_omega_dir):
        key1 = _get_or_create_key()
        key2 = _get_or_create_key()
        assert key1 == key2

    def test_key_path_uses_omega_home(self, tmp_omega_dir):
        kp = _key_path()
        assert str(tmp_omega_dir) in str(kp)


# ============================================================================
# Encrypt/decrypt roundtrip (requires cryptography package)
# ============================================================================


class TestEncryptDecryptRoundtrip:
    @pytest.fixture(autouse=True)
    def _enable_encryption(self, monkeypatch, tmp_omega_dir):
        monkeypatch.setenv("OMEGA_ENCRYPT", "1")
        reset_crypto_state()

    def _has_cryptography(self):
        try:
            import cryptography  # noqa: F401
            return True
        except ImportError:
            return False

    def test_roundtrip(self):
        if not self._has_cryptography():
            pytest.skip("cryptography package not installed")
        original = "sensitive memory content"
        encrypted = encrypt(original)
        assert encrypted.startswith("ENC:")
        assert encrypted != original
        decrypted = decrypt(encrypted)
        assert decrypted == original

    def test_roundtrip_unicode(self):
        if not self._has_cryptography():
            pytest.skip("cryptography package not installed")
        original = "Unicode content: cafe\u0301 \u2603 \U0001f680"
        encrypted = encrypt(original)
        assert decrypt(encrypted) == original

    def test_encrypt_line_decrypt_line_roundtrip(self):
        if not self._has_cryptography():
            pytest.skip("cryptography package not installed")
        line = '{"content": "test data", "type": "lesson"}'
        encrypted = encrypt_line(line)
        assert encrypted.startswith("ENC:")
        decrypted = decrypt_line(encrypted)
        assert decrypted == line


# ============================================================================
# reset_crypto_state
# ============================================================================


class TestResetCryptoState:
    def test_reset_clears_state(self, monkeypatch, tmp_omega_dir):
        monkeypatch.setenv("OMEGA_ENCRYPT", "1")
        try:
            import cryptography  # noqa: F401
        except ImportError:
            pytest.skip("cryptography package not installed")

        # Initialize fernet
        encrypted = encrypt("test")
        assert encrypted.startswith("ENC:")

        # Reset and verify re-initialization works
        reset_crypto_state()
        encrypted2 = encrypt("test2")
        assert encrypted2.startswith("ENC:")


class TestMissingBackendBehaviour:
    """What happens when encryption is on but `cryptography` is unavailable.

    `cryptography` is an optional extra while `is_enabled()` defaults to True,
    so on a default install encryption is nominally on with no backend behind
    it. Exports were then written in plaintext with no signal, leaving data
    unprotected while the caller believed otherwise.
    """

    @staticmethod
    def _without_backend(monkeypatch):
        import omega.crypto as crypto

        monkeypatch.setattr(crypto, "_get_fernet", lambda: None)

    def test_explicit_opt_in_refuses_to_write_plaintext(self, monkeypatch):
        from omega.crypto import encrypt as _encrypt

        monkeypatch.setenv("OMEGA_ENCRYPT", "1")
        self._without_backend(monkeypatch)

        with pytest.raises(RuntimeError, match="backend is unavailable"):
            _encrypt("sensitive")

    def test_inherited_default_falls_back_to_plaintext(self, monkeypatch):
        from omega.crypto import encrypt as _encrypt

        monkeypatch.delenv("OMEGA_ENCRYPT", raising=False)
        self._without_backend(monkeypatch)

        # The layer is documented as optional, so a default install without the
        # extra is not an error state -- but it must be visible, which is what
        # is_encryption_active and the export's `encrypted` field are for.
        assert _encrypt("sensitive") == "sensitive"

    def test_explicitly_disabled_never_raises(self, monkeypatch):
        from omega.crypto import encrypt as _encrypt

        monkeypatch.setenv("OMEGA_ENCRYPT", "0")
        self._without_backend(monkeypatch)

        assert _encrypt("sensitive") == "sensitive"

    def test_is_encryption_active_reports_the_truth(self, monkeypatch):
        from omega.crypto import is_encryption_active

        monkeypatch.delenv("OMEGA_ENCRYPT", raising=False)
        self._without_backend(monkeypatch)
        assert is_encryption_active() is False

        monkeypatch.setenv("OMEGA_ENCRYPT", "0")
        assert is_encryption_active() is False
