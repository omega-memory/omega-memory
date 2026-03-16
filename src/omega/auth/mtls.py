"""Mutual-TLS Certificate Authority for OMEGA agent authentication.

The ``cryptography`` library is an optional dependency.  If it is not
installed, instantiating ``CertificateAuthority`` raises ``RuntimeError``.
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Any

try:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    _HAS_CRYPTO = True
except ImportError:  # pragma: no cover
    _HAS_CRYPTO = False


def _require_crypto() -> None:
    if not _HAS_CRYPTO:
        raise RuntimeError(
            "The 'cryptography' package is required for mTLS support.  "
            "Install it with: pip install cryptography"
        )


class CertificateAuthority:
    """A lightweight CA for issuing and verifying agent client certificates.

    Parameters
    ----------
    ca_key_path:
        Path to the PEM-encoded CA private key.
    ca_cert_path:
        Path to the PEM-encoded CA certificate.
    """

    def __init__(self, ca_key_path: str | Path, ca_cert_path: str | Path) -> None:
        _require_crypto()
        self._ca_key_path = Path(ca_key_path)
        self._ca_cert_path = Path(ca_cert_path)

        with open(self._ca_key_path, "rb") as f:
            self._ca_key = serialization.load_pem_private_key(f.read(), password=None)
        with open(self._ca_cert_path, "rb") as f:
            self._ca_cert = x509.load_pem_x509_certificate(f.read())

        self._revoked_serials: set[str] = set()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def initialize(cls, storage_path: str | Path) -> CertificateAuthority:
        """Generate a new self-signed CA and return an instance.

        Creates ``ca.key`` and ``ca.crt`` inside *storage_path*.
        """
        _require_crypto()
        storage = Path(storage_path)
        storage.mkdir(parents=True, exist_ok=True)

        key = ec.generate_private_key(ec.SECP384R1())
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "OMEGA Agent CA"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "OMEGA"),
        ])
        now = datetime.datetime.now(datetime.timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=3650))
            .add_extension(
                x509.BasicConstraints(ca=True, path_length=0),
                critical=True,
            )
            .sign(key, hashes.SHA384())
        )

        key_path = storage / "ca.key"
        cert_path = storage / "ca.crt"
        key_path.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ),
        )
        # Restrict key permissions
        os.chmod(key_path, 0o600)
        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

        return cls(ca_key_path=key_path, ca_cert_path=cert_path)

    # ------------------------------------------------------------------
    # Issue
    # ------------------------------------------------------------------

    def issue_agent_cert(
        self,
        agent_name: str,
        user_id: str,
        ttl_days: int = 365,
    ) -> dict[str, str]:
        """Issue a client certificate for an agent.

        Returns a dict with keys ``cert_pem``, ``key_pem``, ``ca_pem``.
        """
        agent_key = ec.generate_private_key(ec.SECP384R1())
        now = datetime.datetime.now(datetime.timezone.utc)

        subject = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, agent_name),
            x509.NameAttribute(NameOID.USER_ID, user_id),
        ])
        serial = x509.random_serial_number()

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(self._ca_cert.subject)
            .public_key(agent_key.public_key())
            .serial_number(serial)
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=ttl_days))
            .add_extension(
                x509.BasicConstraints(ca=False, path_length=None),
                critical=True,
            )
            .add_extension(
                x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH]),
                critical=False,
            )
            .sign(self._ca_key, hashes.SHA384())
        )

        cert_pem = cert.public_bytes(serialization.Encoding.PEM).decode()
        key_pem = agent_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ).decode()
        ca_pem = self._ca_cert.public_bytes(serialization.Encoding.PEM).decode()

        return {
            "cert_pem": cert_pem,
            "key_pem": key_pem,
            "ca_pem": ca_pem,
            "serial_number": str(serial),
        }

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------

    def verify_client_cert(self, cert_pem: str) -> dict[str, str]:
        """Verify a client certificate and extract identity attributes.

        Returns a dict with ``agent_name`` and ``user_id``.

        Raises ``ValueError`` on invalid / revoked / expired certificates.
        """
        cert = x509.load_pem_x509_certificate(cert_pem.encode())

        # Check issuer matches our CA
        if cert.issuer != self._ca_cert.subject:
            raise ValueError("Certificate was not issued by this CA")

        # Check revocation
        serial = str(cert.serial_number)
        if serial in self._revoked_serials:
            raise ValueError("Certificate has been revoked")

        # Check expiry
        now = datetime.datetime.now(datetime.timezone.utc)
        if cert.not_valid_after_utc < now:
            raise ValueError("Certificate has expired")
        if cert.not_valid_before_utc > now:
            raise ValueError("Certificate is not yet valid")

        # Verify signature
        try:
            self._ca_key.public_key()  # type: ignore[union-attr]
        except AttributeError:
            pass
        # Use the CA public key to verify
        self._ca_cert.public_key().verify(
            cert.signature,
            cert.tbs_certificate_bytes,
            ec.ECDSA(cert.signature_hash_algorithm),  # type: ignore[arg-type]
        )

        # Extract attributes
        subject = cert.subject
        agent_name = subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
        user_id = subject.get_attributes_for_oid(NameOID.USER_ID)[0].value

        return {"agent_name": str(agent_name), "user_id": str(user_id)}

    # ------------------------------------------------------------------
    # Revoke
    # ------------------------------------------------------------------

    def revoke(self, serial_number: str) -> None:
        """Mark a certificate serial number as revoked (in-memory)."""
        self._revoked_serials.add(serial_number)
