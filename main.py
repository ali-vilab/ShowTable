import argparse
import yaml
from config_schema import AppConfig
from pipeline import run_pipeline
import shutil
from typing import Optional
from pathlib import Path
from datetime import datetime

from utils.log_utils import setup_logging
logger = setup_logging(__name__, 'app.log')

def load_config(path: str, prompts_file_overwrite: Optional[str] = None, eval_role_overwrite: Optional[str] = None, metric_file_overwrite: Optional[str] = None) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg_raw = AppConfig.model_validate(raw)
    if prompts_file_overwrite is not None:
        logger.info(f"overwriting 'data.prompts_file' from config ('{cfg_raw.data.prompts_file}') with command-line value ('{prompts_file_overwrite}').")
        cfg_raw.data.prompts_file = prompts_file_overwrite
    if eval_role_overwrite is not None:
        logger.info(f"overwriting 'eval.role' from config ('{cfg_raw.eval.role}') with command-line value ('{eval_role_overwrite}').")
        cfg_raw.eval.role = eval_role_overwrite
    if metric_file_overwrite is not None:
        logger.info(f"overwriting 'eval.metric_file' from config ('{cfg_raw.eval.metric_file}') with command-line value ('{metric_file_overwrite}').")
        cfg_raw.eval.metric_file = metric_file_overwrite
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if cfg_raw.rewrite.provider == "server":
        rewrite_model_name = cfg_raw.rewrite.server[0].model
    else:
        rewrite_model_name = cfg_raw.rewrite.api[0].model

    if cfg_raw.judge.provider == "server":
        judge_model_name = cfg_raw.judge.server[0].model
    else:
        judge_model_name = cfg_raw.judge.api[0].model
    if cfg_raw.edit.enable == True:
        edit_model_name = cfg_raw.edit.provider
    else:
        edit_model_name = 'False'
    # 构建新的目录名
    if cfg_raw.rewrite.use_prompt_directly:
        dir_suffix = (
            f"Direct_prompt_"
            f"{cfg_raw.rewrite.provider}_model_{rewrite_model_name}_"
            f"Gen_{cfg_raw.image_gen.provider}_"
            f"Judge_{cfg_raw.judge.provider}_model_{judge_model_name}_use_history_{cfg_raw.judge.use_history}_"
            f"Edit_{edit_model_name}_"
            f"Max_Rounds_{cfg_raw.judge.rounds}"
            # f"{timestamp}"
        )
    else:
        dir_suffix = (
            f"Rewrite_{cfg_raw.rewrite.enable}_"
            f"{cfg_raw.rewrite.provider}_model_{rewrite_model_name}_"
            f"Gen_{cfg_raw.image_gen.provider}_"
            f"Judge_{cfg_raw.judge.provider}_model_{judge_model_name}_use_history_{cfg_raw.judge.use_history}_"
            f"Edit_{edit_model_name}_"
            f"Max_Rounds_{cfg_raw.judge.rounds}"
            # f"{timestamp}"
        )
    
    # 修改配置
    cfg_raw.data.out_dir = str(Path(cfg_raw.data.out_dir) / dir_suffix)
    cfg_raw.data.records_file = str(Path(cfg_raw.data.out_dir) / cfg_raw.data.records_file)
    cfg_raw.data.success_records_file = str(Path(cfg_raw.data.out_dir) / cfg_raw.data.success_records_file)
    cfg_raw.eval.records_file = str(Path(cfg_raw.data.out_dir) / f"{cfg_raw.eval.role}_{cfg_raw.eval.records_file}")
    cfg_raw.eval.metric_file = str(Path(cfg_raw.data.out_dir) / f"{cfg_raw.eval.role}_{cfg_raw.eval.metric_file}")
    Path(cfg_raw.data.out_dir).mkdir(parents=True, exist_ok=True)
    config_backup_path = Path(cfg_raw.data.out_dir) / "config.yaml"
    shutil.copy2(path, config_backup_path)
    logger.info(f"Config file backed up to: {config_backup_path}")
    return cfg_raw
    
def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/example.yaml")
    ap.add_argument("--prompts-file", type=str, default=None, help="Path to the ground truth file.")
    ap.add_argument("--eval-role", type=str, default=None, help="overwrite the eval.role from the config file.")
    ap.add_argument("--metric-file", type=str, default=None, help="Path to the eval metric file.")
    args = ap.parse_args()

    cfg = load_config(
        args.config, prompts_file_overwrite=args.prompts_file,
        eval_role_overwrite=args.eval_role, metric_file_overwrite=args.metric_file
    )
    logger.info(f"Loaded config: {cfg}")
    run_pipeline(cfg)

if __name__ == "__main__":
    cli()
