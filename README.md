# ShowTable

**ShowTable** is a novel pipeline that synergizes Multimodal Large Language Models (MLLMs) with diffusion models via a progressive self-correcting process for **Creative Table Visualization**. This task challenges models to generate infographics that are both aesthetically pleasing and faithful to the data points in a given table.

This repository contains the official **inference** and **evaluation** implementation of the paper:
> **ShowTable: Unlocking Creative Table Visualization with Collaborative Reflection and Refinement**


## 🚀 Repository Overview

The ShowTable pipeline consists of four key stages, we implement all inference process via API or Sever Calling.

1.  **Rewriting**: The MLLM reasons over the tabular data to plan an aesthetic visual sketch and rewrites the user prompt. You can find the code in ```stages/rewrite_stage.py``` and ```clients/rewrite_client.py```.
2.  **Generation**: The diffusion model creates an initial figure based on the MLLM's sketch. You can find the code in ```stages/image_gen_stage.py``` and ```clients/image_gen_client.py```.
3.  **Reflection**: The MLLM assesses the generated output, identifying inconsistencies and formulating precise editing instructions. You can find the code in ```stages/judge_stage.py``` and ```clients/judge_client.py```.
4.  **Refinement**: The diffusion model edits the figure based on the reflective feedback to produce the final high-fidelity visualization. You can find the code in ```stages/edit_stage.py``` and ```clients/edit_client.py```.

The repository also supports the evaluation on our proposed TableVisBench. You can find the code in ```stages/eval_stage.py``` and ```clients/eval_client.py```.


## 🛠️ Getting Started

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/ali-vilab/ShowTable.git
cd ShowTable
pip install -r requirements.txt
```

### 2. Model Server Preparation

The pipeline requires several models to be running. You can start them using the provided scripts in the `script/` directory.

For example, to start the Qwen-Image service:

```bash
bash script/start_qwen_image.sh
```

Ensure you have the necessary model weights and environment configurations set up as per the scripts.

### 3. Configuration

Configuration is managed via YAML files. A sample configuration is provided in `configs/example.yaml`.

Example `configs/example.yaml` modification:

```yaml
data:
  prompts_file: "data/eval_data.jsonl"
  out_dir: "work_dirs/eval_outputs"

rewrite:
  enable: true
  provider: "server"
  server:
    - model: "qwen3"
      api_base: "http://your-server-ip:port/v1"

image_gen:
  provider: "qwen_image_server"
  qwen_image_server:
    - api_base: "http://your-server-ip:port"
```

### 4. Running the Pipeline

To run the pipeline, use the `main.py` script with your configuration file:

```bash
python main.py -c configs/example.yaml
```


## 📊 Benchmark (TableVisBench)

The repository supports evaluation on **TableVisBench**. Ensure your `data.prompts_file` points to the benchmark dataset and `run.mode` is set to `eval`. The evaluation metrics include:
*   **DA**: Data Accuracy
*   **TR**: Text Rendering
*   **RR**: Relative Relationship
*   **AA**: Additional Information Accuracy
*   **AQ**: Aesthetic Quality

## 📝 Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{liu2025showtable,
  title={ShowTable: Unlocking Creative Table Visualization with Collaborative Reflection and Refinement},
  author={Liu, Zhihang and Bao, Xiaoyi and Li, Pandeng and Zhou, Junjie and Liao, Zhaohe and He, Yefei and Jiang, Kaixun and Xie, Chen-Wei and Zheng, Yun and Xie, Hongtao},
  journal={arXiv preprint arXiv:2512.13303},
  year={2025}
}
```

