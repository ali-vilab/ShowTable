from pathlib import Path
from clients.image_gen_client import ImageGenClient
from config_schema import ImageGenConfig
from utils.io_utils import image_exists
from utils.img_utils import save_image
from PIL import Image
from utils.hash_utils import prompt_digest
from utils.log_utils import setup_logging
logger = setup_logging(__name__, 'app.log')

class ImageGenStage:
    def __init__(self, cfg: ImageGenConfig, out_dir: str, exist_strategy: str):
        self.cli = ImageGenClient(cfg)
        self.out_dir = Path(out_dir)
        self.exist_strategy = exist_strategy
        self.seed = cfg.seed

    def output_path(self, prompt_id: str) -> Path:
        return self.out_dir / 'gen_images' / f"{prompt_id}.png"

    def run(self, prompt_id: str, prompt: str) -> Path | str:
        logger.info(f"Generating image for prompt {prompt_id}")
        path = self.output_path(prompt_id)
        if path.exists() and self.exist_strategy == "skip":
            return path
        img = self.cli.generate(prompt)
        if isinstance(img, Image.Image):
            path.parent.mkdir(parents=True, exist_ok=True)
            save_image(img, str(path))
            return path
        else:
            if img == 'inappropriate':
                return 'inappropriate'
            else:
                return 'error'
