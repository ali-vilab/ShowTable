from clients.judge_client import JudgeClient
from config_schema import JudgeConfig
from PIL import Image
from typing import List
class JudgeStage:
    def __init__(self, cfg: JudgeConfig):
        self.cfg = cfg
        self.cli = JudgeClient(cfg)

    def run(self, image: Image.Image, topic: str, table: str, history: List[dict]) -> str:
        return self.cli.judge(image, topic, table, history)
