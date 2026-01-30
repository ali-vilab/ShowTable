import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from config_schema import AppConfig

_DONE_RE = re.compile(r"<\s*answer\s*>\s*done\s*<\s*/\s*answer\s*>", re.I)

def _now_ts() -> float:
    return time.time()

def read_jsonl(file_path: str):
    with open(file_path, 'r') as f:
        data = [json.loads(line.strip()) for line in f.readlines()]
    return data


def append_snapshot(lock, record_path: str, prompt_id: str, steps: List[Dict[str, Any]], extra: Optional[Dict[str, Any]] = None) -> None:
    """
    以“快照”形式追加一行：{"prompt_id": "...", "steps": [...], "ts": 1739900000.123, ...}
    - steps: [{role: str, content: str, round?: int}]
    - extra: 可选的其它审计字段（如 status、latency 等）
    """
    p = Path(record_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    obj = {"prompt_id": str(prompt_id), "steps": steps, "ts": _now_ts()}
    if extra:
        obj.update(extra)
    with lock:
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_latest_steps(record_path: str, prompt_id: str, return_stage: bool = False) -> List[Dict[str, Any]]:
    """
    扫描 JSONL，返回该 prompt_id 的“最新快照”的 steps（若不存在则空列表）。
    复杂度 O(N)，但简单稳妥；大规模可换索引/DB。
    """
    p = Path(record_path)
    if not p.exists():
        if return_stage:
            return [], None
        return []
    latest: Optional[List[Dict[str, Any]]] = None
    stage = None
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("prompt_id") == str(prompt_id):
                latest = obj.get("steps", [])
                stage = obj.get("stage", None)
    if return_stage:
        return latest or [], stage
    return latest or []

def steps_has_role(steps: List[Dict[str, Any]], role: str) -> bool:
    return any(s.get("role") == role for s in steps)

def steps_get_content(steps: List[Dict[str, Any]], role: str) -> Optional[str]:
    for s in reversed(steps):
        if s.get("role") == role:
            return s.get("content")
    return None


def check_answer_at_end_strip(response_content):
    """判断去除末尾空白后，是否以目标标签结尾"""
    target = '<answer>done</answer>'
    return response_content.rstrip().endswith(target)

def steps_done(steps: List[Dict[str, Any]]) -> bool:
    """只用内容文本判断是否出现 <answer>done</answer>。"""
    for s in steps:
        c = s.get("content") or ""
        if check_answer_at_end_strip(c):
            return True
    return False

def steps_last_completed_round(steps: List[Dict[str, Any]]) -> int:
    """
    计算“已完整完成到的最后一轮”（judge_k + edit_k 都存在）。
    若一轮只有 judge_k 没有 edit_k，不算完成。
    """
    rounds_j = set()
    rounds_e = set()
    for s in steps:
        role = s.get("role", "")
        if role.startswith("judge_"):
            try:
                rounds_j.add(int(role.split("_", 1)[1]))
            except Exception:
                pass
        elif role.startswith("edit_"):
            try:
                rounds_e.add(int(role.split("_", 1)[1]))
            except Exception:
                pass
    r = -1
    while (r + 1) in rounds_j and (r + 1) in rounds_e:
        r += 1
    return r

def steps_pending_edit_round(steps: List[Dict[str, Any]]) -> Optional[int]:
    """若存在某一轮有 judge_k 但无 edit_k，返回最小的那一轮编号；否则 None。"""
    rounds_j = {}
    rounds_e = set()
    for s in steps:
        role = s.get("role", "")
        if role.startswith("judge_"):
            try:
                rounds_j[int(role.split("_", 1)[1])] = True
            except Exception:
                pass
        elif role.startswith("edit_"):
            try:
                rounds_e.add(int(role.split("_", 1)[1]))
            except Exception:
                pass
    for k in sorted(rounds_j.keys()):
        if k not in rounds_e:
            return k
    return None

def latest_image_path(steps: List[Dict[str, Any]]) -> Optional[str]:
    """优先返回最新 edit_k 的图片；否则返回 image_gen 的图片；否则 None。"""
    # 找 edit 最大轮
    best = None
    best_r = -1
    for s in steps:
        role = s.get("role", "")
        if role.startswith("edit_"):
            try:
                r = int(role.split("_", 1)[1])
            except Exception:
                continue
            if r >= best_r:
                best_r = r
                best = s.get("content")
    if best:
        return best
    # 否则用 base
    return steps_get_content(steps, "image_gen")

# 放在 jsonl_utils.py 末尾（已有工具的基础上）
def steps_finish_reason(steps: List[Dict[str, Any]], max_rounds: int, cfg: AppConfig) -> tuple[bool, str | None, int | None]:
    """
    返回 (finished, reason, next_round)
      - finished: 是否完成
      - reason: "answer_done" 或 "max_rounds" 或 None
      - next_round: 若未完成，下一轮应该做的轮次；若完成返回 None
    规则：
      - 若任意 step 文本含 <answer>done</answer> -> answer_done
      - 否则计算 next_round：
          pending = 存在 judge_k 无 edit_k 的最小 k
          last_full = 已完整完成到的最后一轮（judge_k + edit_k 皆存在）
          next_round = pending if pending is not None else last_full + 1
        若 next_round >= max_rounds -> max_rounds 完成
    """
    if steps_done(steps):
        return True, "answer_done", None

    if len(steps) > 0 and (steps[-1]['role'] == 'image_gen' or steps[-1]['role'].startswith("edit")) and steps[-1]['content'] == "Input data may contain inappropriate content.":
        return True, "Input data may contain inappropriate content", None

    last_full = steps_last_completed_round(steps)
    pending = steps_pending_edit_round(steps)
    if cfg.edit.enable == False and pending == 0:
        return True, "judge_once", None
    next_round = pending if pending is not None else (last_full + 1)
    if next_round >= max_rounds:
        return True, "max_rounds", None
    return False, None, next_round
