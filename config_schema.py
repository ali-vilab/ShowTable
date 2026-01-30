from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal

class RunConfig(BaseModel):
    mode: Literal["serial", "eval"] = "serial"
    parallel_mode: Literal["process", "thread", "off"] = "thread"
    max_workers: int = 8

class DataConfig(BaseModel):
    prompts_file: Optional[str] = None
    gt_image_root: Optional[str] = None
    inline_prompts: List[dict] = []
    out_dir: str = "outputs"
    exist_strategy: Literal["skip", "overwrite"] = "skip"
    records_file: Optional[str] = "outputs/records.jsonl"
    success_records_file: Optional[str] = "outputs/visual_success_records.jsonl"

class RewriteServerCfg(BaseModel):
    model: str = "qwen3"
    api_base: str
    api_key: Optional[str] = "EMPTY"
    timeout: int = 60
    temperature: float = 0.0

class RewriteTfCfg(BaseModel):
    model_path: str
    temperature: float = 0.0

class RewriteConfig(BaseModel):
    enable: bool = True
    text_prompt_template: str = None
    use_prompt_directly: bool = False
    max_new_tokens: int = 4096
    retry_times: int = 1000
    provider: Literal["api", "server", "transformers"] = "server"
    api: List[RewriteServerCfg] = [
        RewriteServerCfg(api_base="http://localhost:23333/v1")
    ]
    server: List[RewriteServerCfg] = [
        RewriteServerCfg(api_base="http://localhost:23333/v1")
    ]
    transformers: RewriteTfCfg = RewriteTfCfg(model_path="./qwen3")

class GenProvider(BaseModel):
    model: str = ""
    api_base: str
    api_key: Optional[str] = None
    timeout: int = 60

class ImageGenConfig(BaseModel):
    provider: Literal[
        "qwen_image_server", "gpt", "wan", "pregen_bagel", "pregen_bagel_think",
        "pregen_omnigen2", "pregen_uniworld", "pregen_blip3onext", "pregen_xomni",
        "pregen_flux"
    ] = "qwen_image_server"
    qwen_image_server: List[GenProvider] = [
        GenProvider(api_base="http://localhost:3000")
    ]
    wan_pools: List[GenProvider] = [
        GenProvider(api_base="https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"),
    ]
    seed: int = 42
    retry_times: int = 10

class JudgeServerCfg(BaseModel):
    model: str = "qwen3"
    api_base: str
    api_key: Optional[str] = "EMPTY"
    timeout: int = 1800
    temperature: float = 0.0

class JudgeTfCfg(BaseModel):
    model_path: str
    temperature: float = 0.2

class JudgeConfig(BaseModel):
    provider: Literal["api", "server", "transformers"] = "server"
    max_new_tokens: int = 12284
    retry_times: int = 5
    rounds: int = 1
    use_history: bool = True
    api: List[JudgeServerCfg] = [
        JudgeServerCfg(api_base="http://localhost:23333/v1")
    ]
    server: List[JudgeServerCfg] = [
        JudgeServerCfg(api_base="http://localhost:23334/v1")
    ]
    transformers: JudgeTfCfg = JudgeTfCfg(model_path="./qwen2.5-vl")

class EditProvider(BaseModel):
    model: str = ""
    api_base: str
    api_key: Optional[str] = "EMPTY"
    timeout: int = 1800
    sample_weight: int = 8

class EditConfig(BaseModel):
    enable: bool = True
    provider: Literal["qwen_image_server", "qwen_image_diffsynth_server", "gpt", "wan"] = "qwen_image_server"
    qwen_image_server: List[EditProvider] = [
        EditProvider(api_base="http://localhost:4000")
    ]
    gpt_pools: List[EditProvider] = [
        EditProvider(api_base="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
    ]
    wan_pools: List[EditProvider] = [
        EditProvider(api_base="https://dashscope.aliyuncs.com/api/v1/services/aigc/image2image/image-synthesis"),
    ]
    seed: int = 42
    retry_times: int = 10


class EvalServerCfg(BaseModel):
    model: str = "qwen3"
    api_base: str
    api_key: Optional[str] = "EMPTY"
    timeout: int = 1800
    temperature: float = 0.0

class EvalConfig(BaseModel):
    enable: bool = True
    role: Literal["gt", "image_gen", "edit_0", "edit_1", "edit_2", "latest", "final"] = "gt" 
    provider: Literal["api", ] = "api"
    records_file: Optional[str] = "eval_records.jsonl"
    metric_file: Optional[str] = "eval_metric.json"
    max_new_tokens: int = 16384
    retry_times: int = 50
    use_history: bool = True
    api: List[JudgeServerCfg] = [
        EvalServerCfg(api_base="http://localhost:23333/v1")
    ]


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    

class AppConfig(BaseModel):
    run: RunConfig
    data: DataConfig
    rewrite: RewriteConfig
    image_gen: ImageGenConfig
    judge: JudgeConfig
    edit: EditConfig
    logging: LoggingConfig
    eval: EvalConfig
