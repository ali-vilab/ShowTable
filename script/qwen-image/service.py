from __future__ import annotations

import os
import math
import queue
import base64
import asyncio
import torch
import bentoml

from io import BytesIO
from typing import List
from PIL.Image import Image
from concurrent.futures import ThreadPoolExecutor
from diffusers import QwenImagePipeline, QwenImageEditPipeline, DiffusionPipeline, FlowMatchEulerDiscreteScheduler
from pydantic import BaseModel


WORLD_SIZE = int(os.environ.get("SERVER_WORLD_SIZE", 1))


class BatchInput(BaseModel):
    prompt: str


@bentoml.service(
    name="qwen-image",
    image=bentoml.images.Image(python_version="3.11"),
    traffic={"timeout": 3000},
    # envs=[{"name": "HF_TOKEN"}],
    resources={"gpu": WORLD_SIZE},
    workers=1,
)
class QwenImage:
    @bentoml.on_startup
    def setup_pipeline(self) -> None:

        MODEL_PATH = "models/Qwen-Image"
        # LORA_MODEL_PATH = None
        LORA_MODEL_PATH = "models/Qwen-Image-Lightning/Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors"

        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),  # We use shift=3 in distillation
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),  # We use shift=3 in distillation
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,  # set shift_terminal to None
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        

        self.pipes = []
        self.gpu_queue = queue.Queue()

        for gpu_id in range(WORLD_SIZE):
            device = torch.device(f"cuda:{gpu_id}")
            
            if LORA_MODEL_PATH is not None:
                scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
                pipe = QwenImagePipeline.from_pretrained(
                    MODEL_PATH, scheduler=scheduler, torch_dtype=torch.bfloat16
                ).to(device=device)
                pipe.load_lora_weights(LORA_MODEL_PATH)
                pipe.fuse_lora()
            else:
                pipe = QwenImagePipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to(device=device)
            
            self.pipes.append(pipe)
            self.gpu_queue.put(gpu_id)
        
        self.executor = ThreadPoolExecutor(max_workers=WORLD_SIZE)

    # @bentoml.api
    def _generate_on_gpu(self, gpu_id: int, input_args: dict) -> List[str]:
        pipe = self.pipes[gpu_id]
        # print(f"使用GPU: {gpu_id}")

        if 'negative_prompt' not in input_args:
            input_args['negative_prompt'] = ' '
        if 'true_cfg_scale' not in input_args:
            input_args['true_cfg_scale'] = 1.0
        if 'num_inference_steps' not in input_args:
            input_args['num_inference_steps'] = 8
        if 'callback_on_step_end' not in input_args:
            input_args['callback_on_step_end'] = None
        
        seed = input_args.pop('seed', 42)
        input_args['generator'] = torch.Generator(device=pipe.device).manual_seed(seed)
        
        new_images = pipe(**input_args).images
        
        return [self.pil2base64(image) for image in new_images]

    @bentoml.api
    async def txt2img(self, input_args: dict):
        """异步API支持真正的并发"""
        loop = asyncio.get_event_loop()
        
        # 异步获取可用GPU
        gpu_id = await loop.run_in_executor(None, self.gpu_queue.get)
        
        try:
            # 在线程池中执行GPU推理
            result = await loop.run_in_executor(
                self.executor, 
                self._generate_on_gpu, 
                gpu_id, 
                input_args
            )
            
        finally:
            # 释放GPU回队列
            self.gpu_queue.put(gpu_id)
        
        return result
    def pil2base64(self, image):
        buffer = BytesIO()
        image.save(buffer, format="png")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return image_b64