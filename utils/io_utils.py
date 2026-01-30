from pathlib import Path
from typing import List, Optional
import json
def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def read_prompts(prompts_file: Optional[str], inline_prompts: List[dict]) -> List[dict]:
    if prompts_file:
        # read jsonl
        prompts: List[dict] = [json.loads(line) for line in open(prompts_file, "r", encoding="utf-8")]
        return prompts
    return inline_prompts

def image_exists(path: str) -> bool:
    return Path(path).exists()
