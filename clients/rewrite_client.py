import logging, requests
from typing import Optional
from config_schema import RewriteConfig
import openai
import time
from openai import OpenAI
import re
import random
from utils.log_utils import setup_logging
logger = setup_logging(__name__, 'app.log')

class RewriteClient:
    def __init__(self, cfg: RewriteConfig):
        self.cfg = cfg
        self.max_new_tokens = cfg.max_new_tokens
        self.SYSTEM_MESSAGE = "You are a helpful assistant. Your task is to carefully understand the user’s input structured data (e.g., Markdown tables) and transform it into a more detailed, precise, and well-structured description, making it fully optimized for image generation models to produce outputs that strictly align with the user’s requirements. When presented with a table, you deeply interpret its content, then craft a natural, narrative think process that envisions how the information should be transformed into a compelling visual representation. This includes reasoning about the suitable visualization style, layout structure, color palette, typography, iconography, and compositional balance, while ensuring data accuracy and aesthetic clarity."
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

    def rewrite(self, title: str, content: str) -> str:
        user_prompt = f"You are an expert prompt engineer. Your client wants an image about '{title}', and has provided this data. Write the perfect prompt to get a stunning result from a model like Stable Diffusion.\n{content}\nThink first in <think> tag, then directly output your prompt in <answer> tag. Format strictly as:  <think>your thinking</think><answer>put your refined prompt here.</answer>"
        if self.cfg.provider in ["server", "api"]:
            result = self.get_answer_with_retry(user_prompt, max_retries=self.cfg.retry_times)
            return result
        else:
            raise NotImplementedError("Transformers provider is not implemented")
    
    def get_answer_with_retry(self, user_prompt, max_retries=5, initial_delay=1, backoff_factor=2):
        """指数退避重试"""
        delay = initial_delay
        
        for attempt in range(max_retries):
            try:
                response_content, error_code = self.curl_func(self.SYSTEM_MESSAGE, user_prompt)
                
                if response_content is None:
                    logger.error(f"Error code {error_code}, retrying...")
                    time.sleep(delay)
                    # delay *= backoff_factor
                    continue
                
                
                answer = response_content.split("<answer>")[-1].split("</answer>")[0]
                
                if answer:
                    return answer
                
                logger.error(f"Attempt {attempt + 1}: No answer tag found in, retrying in {delay}s...")
                logger.info(f"Response content: {response_content}")
                time.sleep(delay)
                # delay *= backoff_factor  # 指数增长：1s, 2s, 4s, 8s...
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} error: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay)
                # delay *= backoff_factor
        
        raise Exception(f"Failed to get answer after {max_retries} attempts")
    def curl_func(self, system_prompt, text_prompt):
        try:
            server = random.choice(self.server_pools)
            client = OpenAI(
                api_key=server["api_key"],
                base_url=server["api_url"],
            )


            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text_prompt,
                        },
                    ],
                },
            ]
        except Exception as e:
            print(e)
            return None, "Unknown"
        
        try:
            chat_response = client.chat.completions.create(
                model=server["model"],
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=server["temperature"],
                timeout=server["timeout"],
            )
        except (openai.BadRequestError, openai.RateLimitError, openai.APIStatusError) as e:
            # logger.error(e)
            error_json = e.response.json()
            code = error_json['error'].get('code')
            return None, code
        except Exception as e:
            print(e)
            return None, "Unknown"
        
        try:
            response_content = chat_response.choices[0].message.content
            return response_content, None
        except Exception as e:
            print(e)
            return None, "Unknown"