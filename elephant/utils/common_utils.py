import hashlib
from typing import Dict


def get_signature(params: Dict):
    """
    Calculates md5 hash for a dict object.
    """
    def md5_helper(obj):
        return hashlib.md5(str(obj).encode()).hexdigest()

    signature = dict()
    for key, val in params.items():
        signature[key] = str(val)

    md5 = md5_helper(list(signature.items()))

    return md5, signature
