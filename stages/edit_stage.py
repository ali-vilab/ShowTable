from pathlib import Path
from typing import List
from PIL import Image
from clients.edit_client import EditClient
from config_schema import EditConfig
from utils.img_utils import save_image

class EditStage:
    def __init__(self, cfg: EditConfig, out_dir: str):
        self.cli = EditClient(cfg)
        self.out_dir = Path(out_dir)

    def output_edit_path(self, base_path: Path, round_idx: int) -> Path:
        new_path_str = str(base_path).replace("gen_images", "edit_images")
        new_path = Path(new_path_str)
        
        # 修改文件名添加后缀
        return new_path.with_stem(new_path.stem + f"_edit_{round_idx}")


    def run(self, image: Image.Image, judge_prompt: str, base_path: Path, round_idx: int) -> Path:
        image = self.cli.edit(image, judge_prompt)
        if image == "inappropriate":
            return "inappropriate"
        p = self.output_edit_path(base_path, round_idx)
        p.parent.mkdir(parents=True, exist_ok=True)
        save_image(image, str(p))
        return p
