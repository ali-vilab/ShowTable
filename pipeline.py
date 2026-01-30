from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from contextlib import nullcontext
from multiprocessing import Pool
from queue import Queue
from threading import Thread
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json
import time
import os
from utils.hash_utils import prompt_digest

from config_schema import AppConfig
from utils.io_utils import ensure_dir, read_prompts
from utils.img_utils import base64_to_pil
from utils.io_utils import image_exists
from stages.rewrite_stage import RewriteStage
from stages.image_gen_stage import ImageGenStage
from stages.judge_stage import JudgeStage
from stages.edit_stage import EditStage
from stages.eval_stage import EvalStage
from config_schema import AppConfig
from utils.jsonl_utils import (
    load_latest_steps, append_snapshot,
    steps_has_role, steps_get_content, steps_done,
    steps_last_completed_round, steps_pending_edit_round, latest_image_path,
    steps_finish_reason, check_answer_at_end_strip
)
from utils.log_utils import setup_logging
from utils.img_utils import pil_to_base64


logger = setup_logging(__name__, 'app.log')

def _load_image(path: Path) -> Image.Image:
    return Image.open(str(path)).convert("RGB")

import re as _re
_DONE_RE_LOCAL = _re.compile(r"<\s*answer\s*>\s*done\s*<\s*/\s*answer\s*>", _re.I)
def re_search_done(text: str) -> bool:
    return bool(_DONE_RE_LOCAL.search(text or ""))

def _single_infer_task(cfg: AppConfig, rw: RewriteStage, im: ImageGenStage, jd: JudgeStage, ed: EditStage, prompt: dict, lock):
    """
    断点续跑 + 单行快照（每个 prompt_id 一个 steps list）
    steps 的元素格式严格为：{role: str, content: str, round?: int}
       - rewrite: {role:'rewrite', content: rewritten_prompt}
       - image_gen: {role:'image_gen', content: image_path}
       - judge_k: {role:'judge_<k>', content: judge_response_text, round:k}
       - edit_k:  {role:'edit_<k>',  content: image_path, round:k}
    完成条件：任意 judge_k 的 content 含 <answer>done</answer> 或达到最大轮数。
    """
    prompt_id = prompt.get("id")
    topic: str = prompt.get("topic")
    table: str = prompt.get("table")
    if prompt_id is None:
        raise ValueError("prompt dict must contain 'id'")

    records_file = cfg.data.records_file
    t0 = time.monotonic()

    # 1) 读取该 prompt_id 的最新快照（steps 列表）
    steps = load_latest_steps(records_file, str(prompt_id))

    # 2) 若已完成（任意 judge_k 含 <answer>done</answer>，或者edit 为false，且最后一个是judge_1），直接返回
    finished, reason, next_round = steps_finish_reason(steps, cfg.judge.rounds, cfg)
    if finished:
        logger.info(f"[resume] prompt_id={prompt_id} already FINISHED by {reason}. skip.")
        success_record = load_latest_steps(cfg.data.success_records_file, str(prompt_id))
        if len(success_record) == 0:
            if reason in ['Input data may contain inappropriate content']:
                stage = "error"
            else:
                stage = "success"
            append_snapshot(lock, cfg.data.success_records_file, str(prompt_id), steps,
                            extra={"stage": stage, "reason": reason, "topic": topic, 'table': table,
                                        "total_sec": round(time.monotonic()-t0, 4)})
        return

    # 3) rewrite
    if not steps_has_role(steps, "rewrite"):
        t_r0 = time.monotonic()
        rewritten = rw.run(topic, table, prompt)  # 你的签名
        steps.append({"role": "rewrite", "content": rewritten})
        append_snapshot(lock, records_file, str(prompt_id), steps, extra={"stage": "rewrite", "latency_sec": round(time.monotonic()-t_r0, 4)})
        logger.info(f"Rewrite for prompt_id={prompt_id} completed! rewritten={rewritten}")
    else:
        logger.info(f"[resume] prompt_id={prompt_id} already REWRITTEN. skip.")

    rewritten = steps_get_content(steps, "rewrite")

    # 4) image_gen
    if not steps_has_role(steps, "image_gen"):
        t_i0 = time.monotonic()
        base_path = im.run(str(prompt_id), rewritten)  # 你的签名，内部可自跳过
        if isinstance(base_path, str):
            logger.error(f'Image Generate for prompt_id={prompt_id} failed!')
            steps.append({"role": "image_gen", "content": "Input data may contain inappropriate content."})
            append_snapshot(lock, records_file, str(prompt_id), steps, extra={"stage": "error", "latency_sec": round(time.monotonic()-t_i0, 4)})
            if cfg.data.success_records_file:
                append_snapshot(lock, cfg.data.success_records_file, str(prompt_id), steps,
                                extra={"stage": "error", "latency_sec": round(time.monotonic()-t_i0, 4)})
            return 
        
        steps.append({"role": "image_gen", "content": str(base_path)})
        append_snapshot(lock, records_file, str(prompt_id), steps, extra={"stage": "image_gen", "latency_sec": round(time.monotonic()-t_i0, 4)})
        logger.info(f"Image Generate for prompt_id={prompt_id} completed! Image_path={base_path}")
    else:
        logger.info(f"[resume] prompt_id={prompt_id} already IMAGE_GENERATED. skip.")

    # 5) 恢复当前应处理的轮次
    #    如果存在 pending_judge_round（judge_k 已有但 edit_k 没有），先补 edit_k；
    #    否则从 last_completed_round+1 开始新一轮 judge。
    for i in range(len(steps)):
        if steps[i]['role'] == 'image_gen':
            base_path = steps[i]['content']
            break
    base_base64_image = pil_to_base64(_load_image(Path(base_path)))
    user_prompt = f"Your primary goal is to generate a high-quality prompt for an image editing model if, and only if, the infographic for \"{topic}\" has errors. First, find and analyze the errors. Then, craft the image editing tool prompt. If no errors are found, your only final output after analysis should be \"<answer>done</answer>\".\n Table data: {table}\nThink first in <think> and </think> tag. Your output must strictly follow the format: <think>your thinking</think><assessment>...</assessment><analysis>...</analysis><tool_call>...</tool_call> or <think>your thinking</think><assessment>...</assessment><analysis>...</analysis><answer>done</answer>."

    history = [{
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": jd.cli.SYSTEM_MESSAGE,
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base_base64_image}"
                }
            },
            {
                "type": "text",
                "text": user_prompt,
            },
        ],
    }]

    for i in range(len(steps)):
        step = steps[i]
        if 'judge' in step['role']:
            history.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": step['content'],
                    },
                ],
            })
            logger.info(f"[resume] prompt_id={prompt_id} already JUDGED_{step['round']}. skip.")
        elif 'edit' in step['role']:
            img_content = step['content']
            edit_base64_image = pil_to_base64(_load_image(Path(img_content)))
            history.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{edit_base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                ],
            })
            logger.info(f"[resume] prompt_id={prompt_id} already EDITED_{step['round']}. skip.")
    while True:
        finished, reason, next_round = steps_finish_reason(steps, cfg.judge.rounds, cfg)
        # 确定当前图：优先最新编辑图，否则 base 图
        img_path = latest_image_path(steps)
        if not img_path:
            raise RuntimeError("No image found to proceed.")
        img = _load_image(Path(img_path))

        # 5.1 若该轮没有 judge_k，先做 judge
        j_role = f"judge_{next_round}"
        if not steps_has_role(steps, j_role):
            t_j0 = time.monotonic()
            judge_json, response = jd.run(img, topic, table, history)
            if judge_json.get('status', '') == 'failed':
                logger.error(f'{j_role} for prompt_id={prompt_id} failed!')
                steps.append({"role": j_role, "content": "Input data may contain inappropriate content."})
                append_snapshot(lock, records_file, str(prompt_id), steps, extra={"stage": "error", "latency_sec": round(time.monotonic()-t_j0, 4)})
                if cfg.data.success_records_file:
                    append_snapshot(lock, cfg.data.success_records_file, str(prompt_id), steps,
                                    extra={"stage": "error", "latency_sec": round(time.monotonic()-t_j0, 4)})
                return 
            history.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": response,
                    },
                ],
            })
            steps.append({"role": j_role, "content": response, "round": next_round})
            append_snapshot(lock, records_file, str(prompt_id), steps, extra={"stage": "judge", "round": next_round, "latency_sec": round(time.monotonic()-t_j0, 4)})
            logger.info(f"Judge_{next_round} for prompt_id={prompt_id} completed! Response={response}")

            # DONE 信号：文本中出现 <answer>done</answer>
            if response and check_answer_at_end_strip(response):
                logger.info(f"[finish] prompt_id={prompt_id} completed. Reason=<answer>done</answer> is appear.")
                if cfg.data.success_records_file:
                    append_snapshot(lock, cfg.data.success_records_file, str(prompt_id), steps,
                                    extra={"stage": "success", "reason": "answer_done", "topic": topic, 'table': table,
                                        "total_sec": round(time.monotonic()-t0, 4)})
                break
        else:
            logger.info(f"[resume] prompt_id={prompt_id} already {j_role}. skip.")


        if not cfg.edit.enable:
            logger.info(f"[finish] prompt_id={prompt_id} completed. Because edit is false.")
            if cfg.data.success_records_file:
                append_snapshot(lock, cfg.data.success_records_file, str(prompt_id), steps,
                                extra={"stage": "success", "reason": "judge_once", "topic": topic, 'table': table,
                                    "total_sec": round(time.monotonic()-t0, 4)})
            break

        # 5.2 若该轮尚无 edit_k，则做 edit
        e_role = f"edit_{next_round}"
        assert not steps_has_role(steps, e_role)

        judge_output = steps_get_content(steps, j_role) or ""
        judge_json = json.loads(judge_output.split("<tool_call>")[-1].split("</tool_call>")[0])
        try:
            judge_text = judge_json['arguments']['prompt']
        except:
            if cfg.data.success_records_file:
                append_snapshot(lock, cfg.data.success_records_file, str(prompt_id), steps,
                                extra={"stage": "error", "reason": "inappropriate",
                                    "total_sec": round(time.monotonic()-t0, 4)})
            return
        t_e0 = time.monotonic()
        out_path = ed.run(img, judge_text, Path(steps_get_content(steps, "image_gen")), next_round) 
        if out_path == "inappropriate":
            logger.error('Image Generate for prompt_id={prompt_id} failed!')
            steps.append({"role": e_role, "content": "Input data may contain inappropriate content.", "round": next_round})
            append_snapshot(lock, records_file, str(prompt_id), steps, extra={"stage": "error", "latency_sec": round(time.monotonic()-t_e0, 4)})
            if cfg.data.success_records_file:
                append_snapshot(lock, cfg.data.success_records_file, str(prompt_id), steps,
                                extra={"stage": "error", "reason": "inappropriate",
                                    "total_sec": round(time.monotonic()-t0, 4)})
            return 
        steps.append({"role": e_role, "content": str(out_path), "round": next_round})
        append_snapshot(lock, records_file, str(prompt_id), steps, extra={"stage": "edit", "round": next_round, "latency_sec": round(time.monotonic()-t_e0, 4)})
        logger.info(f"Edit_{next_round} for prompt_id={prompt_id} Edit_prompt={judge_text} completed!")
        

        finished, reason, _ = steps_finish_reason(steps, cfg.judge.rounds, cfg)
        if finished:
            logger.info(f"[finish] prompt_id={prompt_id} completed. Reason={reason}.")
            if cfg.data.success_records_file:
                append_snapshot(lock, cfg.data.success_records_file, str(prompt_id), steps,
                                extra={"stage": "success", "reason": reason, "topic": topic, 'table': table,
                                    "total_sec": round(time.monotonic()-t0, 4)})
            break


def _single_eval_task(cfg: AppConfig, ev: EvalStage, prompt: dict, lock):
    prompt_id = prompt['id']
    all_eval_tasks = [
        "data_accuracy",
        "text_rendering",
        "relative_relationship",
        "additional_info_accuracy",
        "aesthetic_quality",
    ]
    eval_steps = load_latest_steps(cfg.eval.records_file, str(prompt_id))
    # [{'task': 'xx', 'score': xx}]
    evaled_tasks = [eval_step['task'] for eval_step in eval_steps]
    miss_eval_tasks = [k for k in all_eval_tasks if k not in evaled_tasks]
    if len(miss_eval_tasks) == 0:
        return

    eval_role = cfg.eval.role

    if eval_role != "gt":
        steps, stage = load_latest_steps(cfg.data.success_records_file, str(prompt_id), return_stage=True)
        finished, reason, next_round = steps_finish_reason(steps, cfg.judge.rounds, cfg)
        if not finished:
            if eval_role == "image_gen" and steps_has_role(steps, "image_gen"):
                pass
            else:
                logger.info(f"prompt_id={prompt_id} not finished. Reason={reason}")
                return
        # assert finished
        # if stage == "error":
        #     return
        assert reason in ["answer_done", "max_rounds", 'Input data may contain inappropriate content', 'judge_once', None], f"Unknown reason: {reason}"

        image_roles = ["image_gen"] + [f"edit_{k}" for k in range(cfg.judge.rounds)]
        valid_image_steps = []
        for step in steps:
            if step['role'] in image_roles and step['content'] != "Input data may contain inappropriate content.":
                valid_image_steps.append(step)
        vaild_image_roles = [step['role'] for step in valid_image_steps]

        if len(valid_image_steps) == 0:
            logger.info(f"prompt_id={prompt_id} no valid image step. skip. steps[1:]={steps[1:]}")
            return

    if eval_role == 'gt':
        # image_path = prompt['image_url']
        image_path = os.path.join(cfg.data.gt_image_root, prompt['image_name'])
    elif eval_role == 'latest':
        image_path = valid_image_steps[-1]['content']
    elif eval_role == 'image_gen':
        # if eval_role not in vaild_image_roles:
        #     logger.info(f"prompt_id={prompt_id} no valid image_gen step. skip. valid_image_roles={vaild_image_roles}")
        #     return
        # 'image_gen' must be valid here as we checked it before
        image_path = steps_get_content(steps, "image_gen")
    elif eval_role.startswith('edit_'):
        if eval_role in vaild_image_roles:
            image_path = steps_get_content(steps, eval_role)
        else:
            image_path = valid_image_steps[-1]['content']
    else:
        raise ValueError(f"Invalid eval_role={eval_role}")
    
    if image_path.startswith("http"):
        pass
    elif not os.path.exists(image_path):
        raise FileNotFoundError(f"[prompt_id] {prompt_id}: Image path={image_path} does not exist.")
        # return

    logger.info(f"Evaluating prompt_id={prompt_id} image_path={image_path}")

    t0 = time.monotonic()
    unsuccessful_tasks = []
    for task in miss_eval_tasks:
        ev_result = ev.run(image_path, prompt['topic'], prompt['table'], task)
        if ev_result is None:
            unsuccessful_tasks.append(task)
            continue
        ev_record = {"task": task, }
        ev_record.update(ev_result)
        eval_steps.append(ev_record)
        append_snapshot(
            lock, cfg.eval.records_file, str(prompt_id), eval_steps,
            extra={"topic": prompt['topic'], "table": prompt['table'], "stage": "success", "latency_sec": round(time.monotonic()-t0, 4)}
        )

    # if len(unsuccessful_tasks) == 0:
    #     append_snapshot(lock, cfg.eval.success_records_file, str(prompt_id), eval_steps, extra={"stage": "success", "latency_sec": round(time.monotonic()-t0, 4)})


def _single_task(args):
    cfg, rw, im, jd, ed, ev, prompt, lock = args
    if rw is not None and im is not None and jd is not None and ed is not None:
        _single_infer_task(cfg, rw, im, jd, ed, prompt, lock)
    
    if cfg.eval.enable and ev is not None:
        _single_eval_task(cfg, ev, prompt, lock)

    return 1


def _serial_mode(cfg: AppConfig, prompts: List[dict]):
    rw, im = RewriteStage(cfg.rewrite), ImageGenStage(cfg.image_gen, cfg.data.out_dir, cfg.data.exist_strategy)
    jd, ed = JudgeStage(cfg.judge), EditStage(cfg.edit, cfg.data.out_dir)
    ev = EvalStage(cfg.eval)
    if cfg.run.parallel_mode == "off":
        lock = nullcontext()
        all_args = [(cfg, rw, im, jd, ed, ev, p, lock) for p in prompts]
        for args in all_args:
            _single_task(args)
    elif cfg.run.parallel_mode == "thread":
        lock = threading.Lock()
        all_args = [(cfg, rw, im, jd, ed, ev, p, lock) for p in prompts]
        with ThreadPoolExecutor(max_workers=cfg.run.max_workers) as ex:
            futs = [ex.submit(_single_task, args) for args in all_args]
            for f in tqdm(as_completed(futs), total=len(futs), desc="Processing"):
                f.result()
    elif cfg.run.parallel_mode == "process":
        raise NotImplementedError("Process mode not implemented")
        with Pool(processes=cfg.run.max_workers) as pool:
            results = list(tqdm(pool.imap_unordered(_single_task, all_args), total=len(all_args), desc="Processing"))
    else:
        raise NotImplementedError(f"Parallel mode {cfg.run.parallel_mode} not implemented")
    
    if cfg.eval.enable and ev is not None:
        ev.report_scores(prompts)
        logger.info(f"Report saved to {cfg.eval.metric_file}")

def _eval_mode(cfg: AppConfig, prompts: List[dict]):
    ev = EvalStage(cfg.eval)

    if cfg.eval.role == "final":
        ev.report_final_scores(prompts)
        logger.info(f"Report saved to {cfg.eval.metric_file}")
        return
    
    if cfg.run.parallel_mode == "off":
        lock = nullcontext()
        all_args = [(cfg, None, None, None, None, ev, p, lock) for p in prompts]
        for args in all_args:
            _single_task(args)
    elif cfg.run.parallel_mode == "thread":
        lock = threading.Lock()
        all_args = [(cfg, None, None, None, None, ev, p, lock) for p in prompts]
        with ThreadPoolExecutor(max_workers=cfg.run.max_workers) as ex:
            futs = [ex.submit(_single_task, args) for args in all_args]
            for f in tqdm(as_completed(futs), total=len(futs), desc="Processing"):
                f.result()
    elif cfg.run.parallel_mode == "process":
        raise NotImplementedError("Process mode not implemented")
        with Pool(processes=cfg.run.max_workers) as pool:
            results = list(tqdm(pool.imap_unordered(_single_task, all_args), total=len(all_args), desc="Processing"))
    else:
        raise NotImplementedError(f"Parallel mode {cfg.run.parallel_mode} not implemented")

    ev.report_scores_accelerate(prompts)
    logger.info(f"Report saved to {cfg.eval.metric_file}")

def run_pipeline(cfg: AppConfig):
    ensure_dir(cfg.data.out_dir)
    prompts = read_prompts(cfg.data.prompts_file, cfg.data.inline_prompts)
    if not prompts:
        logger.warning("No prompts found. Nothing to do.")
        return
    logger.info(f"Loaded {len(prompts)} prompts. Mode={cfg.run.mode}")

    if cfg.run.mode == "serial":
        _serial_mode(cfg, prompts)
    elif cfg.run.mode == "eval":
        _eval_mode(cfg, prompts)
    else:
        raise NotImplementedError(f"Mode {cfg.run.mode} not implemented")