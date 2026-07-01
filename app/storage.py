import re
import uuid
from pathlib import Path

_JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class Storage:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def new_job_id(self) -> str:
        return uuid.uuid4().hex

    @staticmethod
    def is_valid_job_id(job_id: str) -> bool:
        return bool(_JOB_ID_RE.match(job_id))

    def path_for(self, job_id: str) -> Path:
        return self.output_dir / f"{job_id}.png"

    def url_for(self, job_id: str) -> str:
        return f"/files/{job_id}.png"

    def exists(self, job_id: str) -> bool:
        return self.path_for(job_id).is_file()

    def save_png(self, job_id: str, image) -> Path:
        path = self.path_for(job_id)
        image.save(path, format="PNG")
        return path

    def delete(self, job_id: str) -> bool:
        p = self.path_for(job_id)
        if p.is_file():
            p.unlink()
            return True
        return False
