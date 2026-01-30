from clients.rewrite_client import RewriteClient
from config_schema import RewriteConfig

class RewriteStage:
    def __init__(self, cfg: RewriteConfig):
        self.cli = RewriteClient(cfg)
        self.use_prompt_directly = cfg.use_prompt_directly
        self.enable = cfg.enable
        if not self.enable:
            if cfg.text_prompt_template is None:
                raise ValueError("Rewrite is not enabled but no template is provided")
            self.text_prompt_template = cfg.text_prompt_template
    def run(self, topic: str, table: str, prompt) -> str:
        if self.use_prompt_directly:
            prompt_from_file = prompt.get("prompt", None)
            assert prompt_from_file is not None, "prompt_from_file is None"
            return prompt_from_file
        return self.cli.rewrite(topic, table) if self.enable else self.text_prompt_template.format(topic=topic, table=table)
