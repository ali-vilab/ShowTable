import logging, requests
from typing import Optional, List, Union
from config_schema import EvalConfig
from openai import OpenAI
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
import re
import os
import ast
import base64
import json
import openai
import random
import time
import torch
import numpy as np
from PIL import Image
from utils.img_utils import pil_to_base64
from utils.log_utils import setup_logging
logger = setup_logging(__name__, 'app.log')


class EvalClient:
    def __init__(self, cfg: EvalConfig, system_prompt: str, check_keys: List[str], strict_check: bool = True):
        self.cfg = cfg

        self.system_prompt = system_prompt
        self.max_new_tokens = cfg.max_new_tokens
        self.server_pools = []
        self.check_keys = check_keys
        self.strict_check = strict_check
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

    def eval(self, image: Union[Image.Image, str, bytes], user_prompt: str) -> dict:
        if isinstance(image, str) and image.startswith("http"):
            base64_image = image
        elif isinstance(image, Image.Image):
            base64_image = pil_to_base64(image)
            base64_image = f"data:image;base64,{base64_image}"
        elif isinstance(image, str) and os.path.isfile(image):
            base64_image = pil_to_base64(Image.open(image).convert("RGB"))
            base64_image = f"data:image;base64,{base64_image}"
        elif isinstance(image, bytes):
            encoded_image = base64.b64encode(image)
            encoded_image_text = encoded_image.decode("utf-8")
            base64_image = f"data:image;base64,{encoded_image_text}"

        message = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": self.system_prompt},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": base64_image}},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        scores = self.get_answer_with_retry(message, max_retries=self.cfg.retry_times)
        return scores

    def get_answer_with_retry(self, history, max_retries=5, initial_delay=1, backoff_factor=2):
        """指数退避重试"""
        delay = initial_delay
        attempt = 0
        # for attempt in range(max_retries):
        while True:
            try:
                response_content, error_code = self.curl_func(history)
                
                if response_content is None:
                    if attempt >= max_retries - 1:
                        break

                    if error_code == "Unknown":
                        attempt += 1
                    else:
                        attempt += 0.01
                    
                    logger.error(f"Error code {error_code}, retrying...")
                    time.sleep(delay)
                    # delay *= backoff_factor
                    continue
                
                struct_response_content = self.parse_response(response_content)
                return struct_response_content
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} error: {e}, error_code: {error_code}")
                logger.error(response_content)
                attempt += 1
                if attempt == max_retries - 1:
                    break
                time.sleep(delay)
                # delay *= backoff_factor
        
        logger.error(f"Failed to get answer after {max_retries} attempts")
        return None

    def parse_response(self, response):
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```python"):
            response = response[9:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        try:
            response = json.loads(response)
        except Exception:
            try:
                response = ast.literal_eval(response)
            except Exception as e:
                raise ValueError("e")
        assert isinstance(response, dict)
        # valid_check_keys = self.check_keys if self.strict_check else self.check_keys[:len(response.keys())]
        if self.strict_check:
            assert all(key in response for key in self.check_keys)
        else:
            assert all(key in self.check_keys for key in response.keys())
        return response
        

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
                temperature=server["temperature"],
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


class AestheticEvalClient:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        # load model and preprocessor
        model, preprocessor = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model = model.to(torch.bfloat16).cuda()
        self.model = model
        self.preprocessor = preprocessor
    
    def eval(self, image: Union[Image.Image, str, bytes]) -> dict:
        if isinstance(image, str) and image.startswith("http"):
            raise NotImplementedError("Not implemented yet")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        elif isinstance(image, str) and os.path.isfile(image):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            raise NotImplementedError("Not implemented yet")
        
        # preprocess image
        pixel_values = (
            self.preprocessor(images=image, return_tensors="pt")
            .pixel_values.to(torch.bfloat16)
            .cuda()
        )

        # predict aesthetic score
        with torch.inference_mode():
            score = self.model(pixel_values).logits.squeeze().float().cpu().item()
        
        return {"score": score}

