import hashlib

def prompt_digest(prompt: str, seed: int) -> str:
    key = f"{prompt}::seed={seed}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()[:16]
