"""Stores uploaded audio and images under random ids until they are searched."""
import re
import uuid
from pathlib import Path

_ID = re.compile(r"^[0-9a-f]{32}$")


class UploadStore:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, data: bytes, suffix: str) -> str:
        upload_id = uuid.uuid4().hex
        (self.directory / f"{upload_id}.{suffix}").write_bytes(data)
        return upload_id

    def path(self, upload_id: str) -> Path:
        if not isinstance(upload_id, str) or not _ID.match(upload_id):
            raise KeyError(upload_id)
        matches = list(self.directory.glob(f"{upload_id}.*"))
        if not matches:
            raise KeyError(upload_id)
        return matches[0]
