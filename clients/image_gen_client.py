import logging
import random
import requests
import io
import bentoml
from typing import Dict, Optional
import time
from PIL import Image
from config_schema import ImageGenConfig
from utils.img_utils import base64_to_pil, pil_to_base64
from utils.log_utils import setup_logging
logger = setup_logging(__name__, 'app.log')


class ImageGenClient:
    def __init__(self, cfg: ImageGenConfig):
        self.cfg = cfg
        self.server_pools = []
        self.provider = cfg.provider
        if self.provider == 'qwen_image_server':
            for endpoint in self.cfg.qwen_image_server:
                self.server_pools.append(
                    {
                        "api_base": endpoint.api_base,
                        "timeout": endpoint.timeout,
                    }
                )
                self.model = endpoint.model
        elif self.provider == 'wan':
            for endpoint in self.cfg.wan_pools:
                self.server_pools.append(
                    {
                        "api_url": endpoint.api_base,
                        "model": endpoint.model,
                        "api_key": endpoint.api_key,
                    }
                )
                self.model = endpoint.model

    def wan_post(self, prompt: str) -> Optional[Image.Image | str]:
        task_id, api_key = self.call_server_submit(prompt)
        if task_id == 'limit':
            raise Exception("超过并发上限，请稍后再试！")
        if task_id == 'inappropriate':
            logger.error("警告：输入数据可能包含不适当内容。请检查输入数据。")
            return 'inappropriate'
        gen_image = self.call_server_get(task_id, api_key)
        if gen_image is None:
            return 'inappropriate'
        return gen_image

    def qwen_post(self, prompt: str) -> Optional[Image.Image]:
        server = random.choice(self.server_pools)
        input_args = {
            "prompt": [prompt],
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 8,
            "true_cfg_scale": 1.0,
            "seed": self.cfg.seed,
        }
        try:
            with bentoml.SyncHTTPClient(server['api_base'], timeout=server['timeout']) as client:
                images = client.txt2img(
                    input_args=input_args,
                )
            gen_image = base64_to_pil(images[0])
            return gen_image
        except Exception as e:
            logger.error(f"错误：调用服务失败。URL: {server['api_base']}, 错误信息: {e}")
            return None
    
    def generate(self, prompt: str) -> Image.Image | str:
        max_retries = self.cfg.retry_times
        initial_delay = 2
        backoff_factor = 1
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                if self.provider == 'wan':
                    gen_image = self.wan_post(prompt)
                elif self.provider == 'qwen_image_server':
                    gen_image = self.qwen_post(prompt)
                else:
                    raise ValueError("Invalid provider")

                if isinstance(gen_image, Image.Image):
                    return gen_image
                else:
                    raise Exception("Failed to get answer")
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} error: {e}")
                if attempt == max_retries - 1:
                    return 'inappropriate'
                time.sleep(delay)
                delay *= backoff_factor
        # raise Exception(f"Failed to get answer after {max_retries} attempts")
        return "inappropriate"


    def call_server_submit(self, prompt):
        input_args = {
            "model": self.model,
            "input": {
                "prompt": prompt,
            },
            "parameters": {
                "size": "1024*1024",
                "n": 1
            }
        }

        server = random.choice(self.server_pools)
        headers = {
            'X-DashScope-Async': 'enable',
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {server['api_key']}"
        }
        try:
            response = requests.post(server['api_url'], json=input_args, headers=headers)
            response.raise_for_status()
            task_id = response.json()['output']['task_id']
        except requests.exceptions.RequestException as e:
            logger.error(f"\n请求失败: {e}")
            # 如果有响应内容，也打印出来，方便排查问题
            if e.response is not None:
                logger.error(f"状态码: {e.response.status_code}, 响应内容: {e.response.text}")
            if 'Requests rate limit exceeded' in e.response.text:
                logger.error("超过并发上限，可以重试多次！")
                return 'limit', None
            elif 'Input data may contain inappropriate content.' in e.response.text:
                logger.error("输入数据可能包含不appropriate内容。请检查输入数据。")
                return 'inappropriate', None
            elif 'url error' in e.response.text:
                logger.error("url error, retry")
                return 'limit', None
            task_id = None
        return task_id, server['api_key']


    def call_server_get(self, task_id, api_key):
        task_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
        headers = {'Authorization': f"Bearer {api_key}"}
        for _ in range(10000):
            try:
                response = requests.get(task_url, headers=headers, timeout=30)
                response.raise_for_status()
                response_data = response.json()
                if response_data['output']['task_status'] in ['PENDING', 'RUNNING']:
                    logger.info(response_data['output']['task_status'])
                    time.sleep(8)
                    continue
                logger.info(response_data['output']['task_status'])
                if response_data['output']['task_status'] == 'SUCCEEDED':
                    image_url = response_data['output']['results'][0]['url']
                else:
                    image_url = None
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"\n请求失败: {e}")
                # 如果有响应内容，也打印出来，方便排查问题
                if e.response is not None:
                    logger.error(f"状态码: {e.response.status_code}, 响应内容: {e.response.text}")
                if 'Requests rate limit exceeded' in e.response.text:
                    logger.error("超过并发上限，延时重试！")
                    time.sleep(12)
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
