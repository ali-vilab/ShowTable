import argparse
import yaml
from config_schema import AppConfig
import shutil
from pathlib import Path
from datetime import datetime
import json
import os
from utils.log_utils import setup_logging
from utils.io_utils import ensure_dir
logger = setup_logging(__name__, 'reuse.log')

def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg_raw = AppConfig.model_validate(raw)
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
    raw_output_dir = cfg_raw.data.out_dir
    # 修改配置
    cfg_raw.data.out_dir = str(Path(cfg_raw.data.out_dir) / dir_suffix)
    cfg_raw.data.records_file = str(Path(cfg_raw.data.out_dir) / cfg_raw.data.records_file)
    cfg_raw.data.success_records_file = str(Path(cfg_raw.data.out_dir) / cfg_raw.data.success_records_file)
    cfg_raw.eval.records_file = str(Path(cfg_raw.data.out_dir) / f"{cfg_raw.eval.role}_{cfg_raw.eval.records_file}")
    cfg_raw.eval.metric_file = str(Path(cfg_raw.data.out_dir) / f"{cfg_raw.eval.role}_{cfg_raw.eval.metric_file}")
    logger.info(f"Output dir: {cfg_raw.data.success_records_file}")
    assert os.path.exists(cfg_raw.data.success_records_file) == False, cfg_raw.data.success_records_file

    jsonl_records = find_records_by_prefix(raw_output_dir, cfg_raw.rewrite.enable, cfg_raw.rewrite.provider, rewrite_model_name)
    if len(jsonl_records) != 0:
        ensure_dir(os.path.dirname(cfg_raw.data.records_file))
        rewrite_reuse_prompt_id2json = {}
        for jsonl_record in jsonl_records:
            with open(jsonl_record, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    if record.get('steps', [])[-1]['role'] == 'rewrite':
                        rewrite_reuse_prompt_id2json[record['prompt_id']] = record
        print("Rewrite reuse records: ", len(rewrite_reuse_prompt_id2json))
        with open(cfg_raw.data.records_file, "a", encoding="utf-8") as f:
            for json_record in rewrite_reuse_prompt_id2json.values():
                f.write(json.dumps(json_record) + "\n")
    
    
    dst_image_base = os.path.join(cfg_raw.data.out_dir)
    jsonl_records = find_records_by_prefix(raw_output_dir, cfg_raw.rewrite.enable, cfg_raw.rewrite.provider, rewrite_model_name, cfg_raw.image_gen.provider)
    if len(jsonl_records) != 0:
        ensure_dir(os.path.dirname(cfg_raw.data.records_file))
        image_gen_reuse_prompt_id2json = {}
        for jsonl_record in jsonl_records:
            if str(dst_image_base) in str(jsonl_record):
                continue
            with open(jsonl_record, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    if record.get('steps', [])[-1]['role'] == 'image_gen':
                        image_gen_reuse_prompt_id2json[record['prompt_id']] = (record, os.path.dirname(jsonl_record))
        print("Image Generation reuse records: ", len(image_gen_reuse_prompt_id2json))
        with open(cfg_raw.data.records_file, "a", encoding="utf-8") as f:
            for json_record, origin_image_gen_path in image_gen_reuse_prompt_id2json.values():
                dst_image_gen_path = os.path.join(dst_image_base, "gen_images")
                src_image_gen_path = os.path.join(origin_image_gen_path, "gen_images")
                if os.path.exists(src_image_gen_path):
                    if not os.path.exists(dst_image_gen_path):
                        shutil.copytree(src_image_gen_path, dst_image_gen_path)
                else:
                    src_image_gen_path = json_record['steps'][-1]['content']
                    dst_image_gen_path = os.path.join(dst_image_base, "gen_images", f"{json_record['prompt_id']}.png")
                    ensure_dir(os.path.dirname(dst_image_gen_path))
                    assert os.path.exists(src_image_gen_path)
                    if not os.path.exists(dst_image_gen_path):
                        shutil.copy(src_image_gen_path, dst_image_gen_path)
                json_record['steps'][-1]['content'] = json_record['steps'][-1]['content'].replace(src_image_gen_path, dst_image_gen_path)
                f.write(json.dumps(json_record) + "\n")
    return cfg_raw


def find_records_by_prefix(base_dir: str, rewrite_enable: bool, provider: str, model_name: str, image_gen_model: str | None = None) -> list:
    """
    根据前缀查找匹配的文件夹，并返回其中的 records.jsonl 文件路径
    
    Args:
        base_dir: 基础目录
        rewrite_enable: rewrite 是否启用
        provider: provider 名称
        model_name: 模型名称
        image_gen_model: 图片生成模型名称，如果为 None，则不进行图片Gen判断
    
    Returns:
        匹配的 records.jsonl 文件路径列表
    """
    base_path = Path(base_dir)
    
    # 构建搜索前缀
    prefix = os.path.join(f"Rewrite_{rewrite_enable}_{provider}_model_{model_name}")
    if image_gen_model:
        prefix = os.path.join(f"Rewrite_{rewrite_enable}_{provider}_model_{model_name}_Gen_{image_gen_model}")
    logger.info(f"Searching for folders with prefix: {prefix}")
    
    # 查找匹配的文件夹
    matching_folders = []
    if base_path.exists():
        for folder in base_path.iterdir():
            if folder.is_dir() and folder.name.startswith(prefix):
                if image_gen_model:
                    if folder.name.split(prefix)[1].startswith("_Judge"):   
                        matching_folders.append(folder)
                else:
                    if folder.name.split(prefix)[1].startswith("_Gen"):
                        matching_folders.append(folder)
    
    # 查找每个匹配文件夹中的 records.jsonl
    records_files = []
    for folder in matching_folders:
        records_file = folder / "records.jsonl"
        if records_file.exists():
            records_files.append(records_file)
            logger.info(f"Found records file: {records_file}")
        else:
            logger.warning(f"Folder {folder} matched but no records.jsonl found")
    
    if not records_files:
        logger.warning(f"No records.jsonl files found with prefix: {prefix}")
    
    return records_files

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/example.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # logger.info(f"Loaded config: {cfg}")

if __name__ == "__main__":
    cli()
