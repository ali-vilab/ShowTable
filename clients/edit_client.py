import logging
import time
import random
import io
import requests
from typing import List, Optional
from urllib.parse import urljoin
import bentoml
from PIL import Image
from config_schema import EditConfig
from utils.img_utils import pil_to_base64, base64_to_pil

from utils.log_utils import setup_logging
logger = setup_logging(__name__, 'app.log')

class EditClient:
    def __init__(self, cfg: EditConfig):
        self.cfg = cfg
        self.server_pools = []
        self.provider = cfg.provider
        if self.provider in ['qwen_image_server', 'qwen_image_diffsynth_server']:
            self.server_pools_weights = []
            for endpoint in self.cfg.qwen_image_server:
                self.server_pools.append(
                    {
                        "api_base": endpoint.api_base,
                        "timeout": endpoint.timeout,
                    }
                )
                self.server_pools_weights.append(endpoint.sample_weight)
                self.model = endpoint.model
        elif self.provider == 'gpt':
            for endpoint in self.cfg.gpt_pools:
                self.server_pools.append(
                    {
                        "api_base": endpoint.api_base,
                        "model": endpoint.model,
                        "api_key": endpoint.api_key,
                        "timeout": endpoint.timeout,
                    }
                )
                self.model = endpoint.model
        elif self.provider == 'wan':
            for endpoint in self.cfg.wan_pools:
                self.server_pools.append(
                    {
                        "api_base": endpoint.api_base,
                        "model": endpoint.model,
                        "api_key": endpoint.api_key,
                        "timeout": endpoint.timeout,
                    }
                )
                self.model = endpoint.model
        else:
            raise ValueError("Invalid provider")

    def wan_post(self, image: Image.Image, edit_prompt: str) -> Optional[Image.Image | str]:
        generated_image_b64: str = pil_to_base64(image)
        task_id, api_key = self.call_server_submit(edit_prompt, generated_image_b64)
        if task_id == 'limit':
            raise Exception("超过并发上限，请稍后再试！")
        if task_id == 'inappropriate':
            logger.error("警告：输入数据可能包含不适当内容。请检查输入数据。")
            return 'inappropriate'
        edited_image = self.call_server_get(task_id, api_key)
        if edited_image is None:
            return 'inappropriate'
        return edited_image

    def gpt_post(self, image: Image.Image, edit_prompt: str) -> Optional[Image.Image | str]:
        generated_image_b64: str = pil_to_base64(image)
        input_args = {
            "model": self.model,
            "image": [
                f"data:image;base64,{generated_image_b64}"
            ],
            "prompt": edit_prompt,
            "background": "auto",
            "n": 1,
            "quality": "auto",
            "user": "user123456"
        }
        server = random.choice(self.server_pools)
        headers = {
            'Content-Type': 'application/json',
            'Authorization': server["api_key"]
        }
        try:
            response = requests.post(server['api_base'], json=input_args, headers=headers)
            response.raise_for_status()
            image_url = response.json()['data'][0]['b64_json']
        except requests.exceptions.RequestException as e:
            logger.error(f"\n请求失败: {e}")
            # 如果有响应内容，也打印出来，方便排查问题
            if e.response is not None:
                logger.error(f"状态码: {e.response.status_code}, 响应内容: {e.response.text}")
            if 'Requests rate limit exceeded' not in e.response.text:
                logger.error("输入内容可能包含不适当内容，请重新输入")
                return "inappropriate"
            image_url = None

        if image_url is not None:
            try:
                image_response = requests.get(image_url, timeout=server['timeout'])
                image_response.raise_for_status()
                pil_image = Image.open(io.BytesIO(image_response.content))
            except requests.exceptions.RequestException as e:
                logger.error(f"错误：下载图片失败。URL: {image_url}, 错误信息: {e}")
            except Exception as e:
                logger.error(f"下载图片发生未知错误: {e}")
            edited_image = pil_image
            return edited_image

    def qwen_post(self, image: Image.Image, edit_prompt: str) -> Optional[Image.Image]:
        generated_image_b64: str = pil_to_base64(image)
        server = random.choices(self.server_pools, weights=self.server_pools_weights, k=1)[0]
        input_args = {
            "image": generated_image_b64,
            "prompt": [edit_prompt],
            "num_inference_steps": 8,
            "true_cfg_scale": 1.0,
            "seed": self.cfg.seed,
            "height": 1024,
            "width": 1024,
        }
        try:
            with bentoml.SyncHTTPClient(server['api_base'], timeout=server['timeout']) as client:
                images = client.txt2img(
                    input_args=input_args,
                )
            edited_image = base64_to_pil(images[0])
            return edited_image
        except Exception as e:
            logger.error(f"错误：调用服务失败。URL: {server['api_base']}, 错误信息: {e}")
            return None
        

    def qwen_diffsynth_post(self, image: Image.Image, edit_prompt: str) -> Optional[Image.Image]:
        generated_image_b64: str = pil_to_base64(image)
        server = random.choices(self.server_pools, weights=self.server_pools_weights, k=1)[0]
        input_args = {
            "prompt": edit_prompt,
            "edit_image": generated_image_b64,
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 40,
            "seed": self.cfg.seed,
        }
        try:
            with bentoml.SyncHTTPClient(server['api_base'], timeout=server['timeout']) as client:
                images = client.txt2img(
                    input_args=input_args,
                )
            edited_image = base64_to_pil(images[0])
            return edited_image
        except Exception as e:
            logger.error(f"错误：调用服务失败。URL: {server['api_base']}, 错误信息: {e}")
            return None

    def edit(self, image: Image.Image, edit_prompt: str) -> Optional[Image.Image | str]:
        max_retries = self.cfg.retry_times
        initial_delay = 1
        backoff_factor = 2
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                if self.provider == 'wan':
                    img = self.wan_post(image, edit_prompt)
                elif self.provider == 'gpt':
                    img = self.gpt_post(image, edit_prompt)
                elif self.provider == 'qwen_image_server':
                    img = self.qwen_post(image, edit_prompt)
                elif self.provider == 'qwen_image_diffsynth_server':
                    img = self.qwen_diffsynth_post(image, edit_prompt)
                else:
                    raise ValueError("Invalid provider")
                if img == "inappropriate":
                    return "inappropriate"
                if img is None:
                    logger.error(f"Attempt {attempt + 1}: No Image Return found, retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= backoff_factor  # 指数增长：1s, 2s, 4s, 8s...
                else:
                    return img
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} error: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay)
                # delay *= backoff_factor
        raise Exception(f"Failed to get answer after {max_retries} attempts")

    

    def call_server_submit(self, prompt: str, generated_image_b64: str):
        input_args = {
            "model": self.model,
            "input": {
                "prompt": prompt,
                "images": [
                    f"data:image;base64,{generated_image_b64}"
                ]
            },
            "parameters": {
                "n": 1,
                "size": "1024*1024"
            }
        }

        server = random.choice(self.server_pools)
        headers = {
            'X-DashScope-Async': 'enable',
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {server['api_key']}"
        }
        try:
            response = requests.post(server['api_base'], json=input_args, headers=headers)
            response.raise_for_status()
            task_id = response.json()['output']['task_id']
        except requests.exceptions.RequestException as e:
            logger.error(f"\n请求失败: {e}")
            if e.response is not None:
                logger.error(f"状态码: {e.response.status_code}, 响应内容: {e.response.text}")
            task_id = None
            if 'Requests rate limit exceeded' in e.response.text:
                logger.error("超过并发上限，可以重试多次！")
                return 'limit', None
            elif 'Input data may contain inappropriate content.' in e.response.text:
                logger.error("输入数据可能包含不appropriate内容。请检查输入数据。")
                return 'inappropriate', None
            elif 'url error' in e.response.text:
                logger.error("url error, retry")
                return 'limit', None
        return task_id, server['api_key']


    def call_server_get(self, task_id, api_key):
        task_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
        headers = {'Authorization': f"Bearer {api_key}"}
        for i in range(10000):
            try:
                response = requests.get(task_url, headers=headers, timeout=30)
                response.raise_for_status()
                response_data = response.json()
                if response_data['output']['task_status'] in ['PENDING', 'RUNNING']:
                    time.sleep(10)
                    continue
                if response_data['output']['task_status'] == 'SUCCEEDED':
                    image_url = response_data['output']['results'][0]['url']
                else:
                    image_url = None
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"\n请求失败: {e}")
                if e.response is not None:
                    logger.error(f"状态码: {e.response.status_code}, 响应内容: {e.response.text}")
                if 'Requests rate limit exceeded' in e.response.text:
                    logger.error("超过并发上限，延时重试！")
                    time.sleep(10 + random.random() * 10)
                    continue
                else:
                    image_url = None
                    break
        
        if image_url is None:
            return None
        
        try:
            image_response = requests.get(image_url, timeout=30)
            image_response.raise_for_status()
            pil_image = Image.open(io.BytesIO(image_response.content))
            return pil_image
        except requests.exceptions.RequestException as e:
            logger.error(f"错误：下载图片失败。URL: {image_url}, 错误信息: {e}")
            return None
        except Exception as e:
            logger.error(f"下载图片发生未知错误: {e}")
            return None