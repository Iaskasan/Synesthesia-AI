"""Standard-library scrypt password hashing and opaque server-side sessions."""
import hashlib
import hmac
import secrets


def password_hash(password):
    salt = secrets.token_hex(16)
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()
    return f"scrypt${salt}${digest}"


def password_matches(password, encoded):
    _, salt, expected = encoded.split("$")
    actual = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()
    return hmac.compare_digest(actual, expected)


def token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()
