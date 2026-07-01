from pathlib import Path
from typing import List

from app.schemas import LoraInfo


class LoraRegistry:
    def __init__(self, lora_dir: str):
        self.lora_dir = Path(lora_dir)

    def list(self) -> List[LoraInfo]:
        if not self.lora_dir.is_dir():
            return []
        files = sorted(self.lora_dir.glob("*.safetensors"))
        return [LoraInfo(name=f.stem, filename=f.name) for f in files]

    def resolve(self, name: str) -> Path:
        for info in self.list():
            if info.name == name:
                return self.lora_dir / info.filename
        raise KeyError(name)
