import logging, requests
from typing import Optional, List
from config_schema import JudgeConfig
from openai import OpenAI
import re
import os
import base64
import json
import openai
import random
import time
from PIL import Image
from utils.img_utils import pil_to_base64
from utils.log_utils import setup_logging
from utils.jsonl_utils import check_answer_at_end_strip
logger = setup_logging(__name__, 'app.log')

class JudgeClient:
    def __init__(self, cfg: JudgeConfig):
        self.cfg = cfg
        self.use_history = cfg.use_history
        self.SYSTEM_MESSAGE = '''
You are an expert-level Quality Assurance Analyst specializing in data visualization. Your mission is to audit an infographic image against a provided Markdown table and verify **numeric accuracy, geometric proportionality, and labeling fidelity**. If any discrepancy exists, you must explain it and produce a precise, actionable instruction for an image editing model.

# Tools
When necessary, call functions defined in <tools></tools>:
<tools>
{"type": "function", "function": {"name": "image_editing_tool", "description": "Edit the provided image based on the provided prompt", "parameters": {"type": "object", "properties": {"prompt": {"type": "string", "description": "A descriptive prompt for the image to be edited"}}, "required": ["prompt"]}}}
</tools>

Return each function call as a JSON object wrapped in <tool_call>:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Mandatory Checks (apply all relevant)
1) **Data vs. Marks**
   - Bars/points/slices and data labels must equal the table values.
   - Stacked totals equal the sum of segments; percent series sum to ~100%.

2) **Axis & Scale Consistency (CRITICAL)**
   - Identify axis min/max, tick marks, units, and scale type.
   - **Linear interpolation rule**: a value v must map to a position linearly between its surrounding ticks.
   - **Overflow/undershoot rule (must flag)**:
     - If a mark’s top/point **visually exceeds the max tick** while its value ≤ max tick (e.g., a bar labeled 565.8 on a Y-axis with a 600 tick but the bar top is above 600), **this is an error**.
     - If a mark’s top/point is **below** where v should be (e.g., 565.8 drawn near 520) beyond tolerance, **this is an error**.
   - **Label–position agreement**: a data label’s anchor height must be consistent with the axis mapping for the labeled value.
   - **Zero baseline**: Columns/bars start at 0 unless explicitly truncated and labeled as such.
   - **Tick spacing** must be uniform for linear scales (or powers for log scales).
   - **Tolerance**: Flag if the observed position/length deviates by > **2% of the full axis span** or **>3 px**, whichever is larger. For values near the max tick, also flag if the mark crosses the max tick line at all.

3) **Pie/Donut Proportionality (CRITICAL)**
   - Slice angle ∝ value; 78% must be visibly larger than 20%; 30% > 25%.
   - **Angle tolerance**: deviation > **5°** or totals ≠ 100% ±0.5%.

4) **Completeness**
   - All categories/series from the table are present; nothing extra appears.

5) **Labeling & Mapping**
   - Titles, axes, units, legends, series names, colors, and ordering claims match the table and encodings.

6) **Insufficient Information = Discrepancy**
   - Missing/illegible ticks or units that block verification must be corrected (instruct the editor).

# How to Reason (be numeric and explicit)
- State the axis range and tick step you infer (e.g., “Y-axis ticks at 0, 200, 400, 600 M$”).
- Compare **expected vs. observed** positions using linear interpolation.
- Call out **overflow/undershoot** explicitly (e.g., “565.8 should sit just below the 600 tick, but the bar top is above the 600 line by ~5–10 px”).
- Be specific about where the problem occurs (series/category/color/axis/legend/slice).

# Output Requirements
- `<assessment>`: one-sentence verdict (e.g., “The image contains critical axis-position errors.”).
- `<analysis>`: bullet or paragraph list of issues stating **what**, **where**, **why** (reference table and axis logic/tolerance).
- `<tool_call>` (optional): If any issue exists, provide **one** concise paragraph of directives that fix **all** issues (e.g., “Lower the ‘Category X’ bar so its top aligns with 565.8 on a 0–600–800 scale; ensure Y ticks at 0, 200, 400, 600, 800 M$; move the data label to sit just below 600; …”).
- If **no issues**, do **not** call any tool and end with `<answer>done</answer>`.

Tone: objective, precise, analytical. No conversational filler.
'''

        self.max_new_tokens = cfg.max_new_tokens
        if cfg.provider == "transformers":
            raise NotImplementedError("Transformers provider is not implemented")
        elif cfg.provider == "server":
            self.server_pools = []
            for endpoint in self.cfg.server:
                self.server_pools.append(
                    {
                        "model": endpoint.model,
                        "api_url": endpoint.api_base,
                        "api_key": endpoint.api_key,
                        "temperature": endpoint.temperature,
                        "timeout": endpoint.timeout,
                    }
                )
        else:
            self.server_pools = []
            for endpoint in self.cfg.api:
                self.server_pools.append(
                    {
                        "model": endpoint.model,
                        "api_url": endpoint.api_base,
                        "api_key": endpoint.api_key,
                        "temperature": endpoint.temperature,
                        "timeout": endpoint.timeout,
                    }
                )
        

    def judge(self, img: Image.Image, title: str, content: str, history: List[dict]) -> dict:
        base64_image = pil_to_base64(img)
        user_prompt = f"Your primary goal is to generate a high-quality prompt for an image editing model if, and only if, the infographic for \"{title}\" has errors. First, find and analyze the errors. Then, craft the image editing tool prompt. If no errors are found, your only final output after analysis should be \"<answer>done</answer>\".\n Table data: {content}\nThink first in <think> and </think> tag. Your output must strictly follow the format: <think>your thinking</think><assessment>...</assessment><analysis>...</analysis><tool_call>...</tool_call> or <think>your thinking</think><assessment>...</assessment><analysis>...</analysis><answer>done</answer>."
        if self.use_history:
            history = history
        else:
            history = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": self.SYSTEM_MESSAGE,
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                    ],
                }
            ]
        if self.cfg.provider in ["server", "api"]:
            json_tool_call, response_content = self.get_answer_with_retry(history, max_retries=self.cfg.retry_times)
            return json_tool_call, response_content
        else:
            raise NotImplementedError("Transformers provider is not implemented")
    
    
    def get_answer_with_retry(self, history, max_retries=5, initial_delay=1, backoff_factor=2):
        """指数退避重试"""
        delay = initial_delay
        attempt = 0
        
        while True:
            try:
                response_content, error_code = self.curl_func(history)
                
                if response_content is None:
                    if attempt >= max_retries - 1:
                        break

                    if error_code == "Unknown":
                        attempt += 1
                    else:
                        attempt += 0.1
                    
                    logger.error(f"Error code {error_code}, retrying...")
                    time.sleep(delay)
                    # delay *= backoff_factor
                    attempt += 1
                    continue
                
                if check_answer_at_end_strip(response_content):
                    # logger.info("No errors found.")
                    return {"status": "success"}, response_content
                elif "<tool_call>" in response_content:
                    tool_call = response_content.split("<tool_call>")[-1].split("</tool_call>")[0]
                    json_tool_call = json.loads(tool_call)
                    judge_text = json_tool_call['arguments']['prompt']
                    json_tool_call['status'] = 'edit'
                    return json_tool_call, response_content
                else:
                    logger.error(f"Attempt {attempt + 1}: No answer tag found, retrying in {delay}s...")
                    attempt += 1
                    if attempt == max_retries - 1:
                        break
                    time.sleep(delay)
                    # delay *= backoff_factor  # 指数增长：1s, 2s, 4s, 8s...
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} error: {e}, error_code: {error_code}")
                logger.error(response_content)
                attempt += 1
                if attempt == max_retries - 1:
                    break
                time.sleep(delay)
                # delay *= backoff_factor
        
        # raise Exception(f"Failed to get answer after {max_retries} attempts")
        return {"status": "failed"}, response_content

    def curl_func(self, history):
        try:
            server = random.choice(self.server_pools)
            client = OpenAI(
                api_key=server["api_key"],
                base_url=server["api_url"],
            )
            messages = history
        except Exception as e:
            print(e)
            return None, "Unknown"
        
        try:
            chat_response = client.chat.completions.create(
                model=server["model"],
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=server['temperature'],
                timeout=server['timeout'],
            )
        except (openai.BadRequestError, openai.RateLimitError, openai.APIStatusError) as e:
            # logger.error(e)
            error_json = e.response.json()
            code = error_json['error'].get('code')
            return None, code
        except Exception as e:
            logger.error(e)
            return None, "Unknown"
    
        try:
            response_content = chat_response.choices[0].message.content
            return response_content, None
        except Exception as e:
            logger.error(e)
            return None, "Unknown"