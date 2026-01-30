from clients.eval_client import EvalClient, AestheticEvalClient
from config_schema import EvalConfig
from PIL import Image
from typing import List, Union
from utils.jsonl_utils import load_latest_steps
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from collections import Counter


class EvalPrompts:
    def __init__(self):
        self.data_accuracy_system_prompt = \
'''You are an expert-level Quality Assurance Analyst specializing in data visualization. Your mission is to audit an infographic image against a provided Markdown table with the topic. The judgement is splited into some dimensions, and you should only focus on the aspect of the given evaluation dimension.

Your evaluation must be objective, precise, and strictly follow the definitions and output format below, provide a structured, multi-faceted critique.

**Evaluation Dimension Definition:**

**Data Accuracy**
    *   **Focus:** Verify that every single data point from the source table is correctly represented in the image, and not missing, rendered as raw table data, or containing errors.
    *   **Process:** For each row/data point in the table, verify its presence, rendering, and correctness in the image. This includes the numerical value, the associated label (e.g., category name), and its mapping to legends, series names, or colors. If the image only simply prints the raw table data without any data visualization or visual elements, you should consider this as a failure.
    *   **Exclusion:** Do NOT consider the relative size/position in this step. Only focus on the existence and correctness of the data labels, values, and series mapping.


**Mandatory Output Format:**

You MUST provide your response as a single JSON object within a Markdown code block. Do not add any explanatory text outside of the JSON structure.

```json
{
    "total_data_points": <integer>, // The total number of data points in the source table.
    "incorrect_data_points": <integer>, // The number of data points that are missing, have wrong values, or incorrect labels/legends in the image.
    "detailed_explain": "<string>" // Output your specific reason or detailed explain here.
}
```

'''
        self.data_accuracy_user_prompt = "As an expert-level Quality Assurance Analyst, please follow the detailed instructions and structured JSON output format provided in the system prompt. Analyze the provided image based on the data in the Markdown table below and perform a comprehensive evaluation on Data Accuracy.\nTopic:{}\nTable:{}\n\nGenerate your analysis. Your entire response must be only the final JSON object inside a Markdown code block."

        self.text_rendering_system_prompt = \
'''You are an expert-level Quality Assurance Analyst specializing in data visualization. Your mission is to audit an infographic image against a provided Markdown table with the topic. The judgement is splited into some dimensions, and you should only focus on the aspect of the given evaluation dimension.

Your evaluation must be objective, precise, and strictly follow the definitions and output format below, provide a structured, multi-faceted critique.

**Evaluation Dimension Definition:**

**Text Rendering**
    *   **Focus:** Checks the correctness of all textual elements in the image.
    *   **Process:** First, identify and list ALL text visible in the image, including titles, annotations, labels, numbers, and legends. Then, compare this text against common knowledge and the source table to identify any errors (e.g., typos, misspellings, garbled characters, incorrect numbers). When listing incorrect texts, you should only list all error text characters in the image, rather than the whole string containing the error.


**Mandatory Output Format:**

You MUST provide your response as a single JSON object within a Markdown code block. Do not add any explanatory text outside of the JSON structure.

```json
{
    "all_text_in_image": [
        "<string>", // List all text strings found in the image.
        "<string>",
        ...
    ],
    "incorrect_text_in_image": [
        "<string>", // List only the text characters that are incorrect (typos, wrong values, garbled characters, etc.).
        ...
    ]
}
```

'''
        self.text_rendering_user_prompt = "As an expert-level Quality Assurance Analyst, please follow the detailed instructions and structured JSON output format provided in the system prompt. Analyze the provided image based on the data in the Markdown table below and perform a comprehensive evaluation on Text Rendering.\nTopic:{}\nTable:{}\n\nGenerate your analysis. Your entire response must be only the final JSON object inside a Markdown code block."

        self.relative_relationship_system_prompt = \
'''You are an expert-level Quality Assurance Analyst specializing in data visualization. Your mission is to audit an infographic image against a provided Markdown table with the topic. The judgement is splited into some dimensions, and you should only focus on the aspect of the given evaluation dimension.

Your evaluation must be objective, precise, and strictly follow the definitions and output format below, provide a structured, multi-faceted critique.

**Evaluation Dimension Definition:**

**Relative Relationship**
    *   **Focus:** Checks if the visual proportions between data points are correct.
    *   **Process:** For charts like bar, column, or line charts, a larger data value must correspond to a taller/longer bar or a higher point. For pie or donut charts, a larger value must correspond to a larger slice area. Check each data point against others to ensure its visual scale is logically correct relative to them.
    *   **Notation:** If the image only simply prints the raw table data without any data visualization or visual elements, you should consider this as a failure, beacause the relative relationship cannot be checked.

**Mandatory Output Format:**

You MUST provide your response as a single JSON object within a Markdown code block. Do not add any explanatory text outside of the JSON structure.

```json
{
    "total_data_points": <integer>, // The total number of data points in the source table.
    "incorrect_data_points": <integer>, // The number of data points whose visual size is incorrect relative to other data points.
    "detailed_explain": "<string>" // Output your specific reason or detailed explain here.
}
```

'''
        self.relative_relationship_user_prompt = "As an expert-level Quality Assurance Analyst, please follow the detailed instructions and structured JSON output format provided in the system prompt. Analyze the provided image based on the data in the Markdown table below and perform a comprehensive evaluation on Relative Relationship.\nTopic:{}\nTable:{}\n\nGenerate your analysis. Your entire response must be only the final JSON object inside a Markdown code block."

        self.additional_info_accuracy_system_prompt = \
r'''You are an expert-level Quality Assurance Analyst specializing in data visualization. Your mission is to audit an infographic image against a provided Markdown table with the topic. The judgement is splited into some dimensions, and you should only focus on the aspect of the given evaluation dimension.

Your evaluation must be objective, precise, and strictly follow the definitions and output format below, provide a structured, multi-faceted critique.

**Evaluation Dimension Definition:**

**Additional Information Accuracy**
    *   **Focus:** Checks the correctness of non-data elements (not appearing in the topic and table) that provide context, such as annotations, axes, ticks, gridlines, and some unreadable marks (including some markdown marks, table delimiters, \n mark).
    *   **Process:**
        *   **Existence:** First, determine if any additional information (like a Y-axis or X-axis with ticks, lines, additional annotations and marks) is present.
        *   **Axis Indicator/Label/Tick Logic:** If axis marks with scale indicators exist, check their labels. For a numerical axis, the tick mark labels must be logical and sequential (e.g., monotonically increasing: 0, 100, 200, 300, not 0, 200, 100, 300). Calculate the proportion of incorrect tick labels, and give the result ranged from 0 to 1 in the 'percentage_of_incorrect_indicator' field.
        *   **Data-to-Axis Alignment:** If axis marks with scale indicators exist, for each data point from the table, verify its visual alignment with the axis ticks. Complete the 'total_data_points' and 'misaligned_data_points_vs_axis' field based on the following criteria:
            *   An error occurs if a data mark (e.g., the top of a bar) is visually **above** a tick mark when its value is less than or equal to that tick's value (e.g., a bar for value 565 is drawn above the 600 tick).
            *   An error occurs if a data mark is drawn **significantly lower** than its value suggests on the scale (e.g., a value of 565.8 is drawn down near the 520 level on an axis that goes up to 600).
        *   **Additional Mark:** Except the label or tick above, if any additional annotations or marks (like markdown delimiters or \n) are present, judge whether they are appropriate to exist in the image. Give the percentage of how many of them are inappropriate to exist ranged from 0 to 1 in the 'percentage_of_inappropriate_mark' field.

**Mandatory Output Format:**

You MUST provide your response as a single JSON object within a Markdown code block. Do not add any explanatory text outside of the JSON structure.

```json
{
    "detailed_explain": "<string>" // Output your specific reason or detailed explain here.
    "has_additional_indicator": <integer>, // 1 if elements like axes or gridlines with scale indicators exist, otherwise 0.
    "percentage_of_incorrect_indicator": <float>, // (Only if has_additional_indicator is True) The percentage of axis tick labels that are logically incorrect (e.g., not sequential). E.g., 0.25 for 1 out of 4 wrong labels.
    "total_data_points": <integer>, // (Only if has_additional_indicator is True) The total number of data points in the source table.
    "misaligned_data_points_vs_axis": <integer>, // (Only if has_additional_indicator is True) The number of data points from the table whose visual position is incorrectly aligned with the axis scale.
    "has_additional_mark": <integer>, // 1 if additional marks or annotations (not including the indicators above) exist, otherwise 0.
    "percentage_of_inappropriate_mark": <float>, // (Only if has_additional_mark is True) The level of inappropriate marks, ranged from 0 to 1.0.
}
```

'''
        self.additional_info_accuracy_user_prompt = "As an expert-level Quality Assurance Analyst, please follow the detailed instructions and structured JSON output format provided in the system prompt. Analyze the provided image based on the data in the Markdown table below and perform a comprehensive evaluation on Additional Information Accuracy.\nTopic:{}\nTable:{}\n\nGenerate your analysis. Your entire response must be only the final JSON object inside a Markdown code block."

#         self.aesthetic_quality_system_prompt = \
# '''You are an expert-level Quality Assurance Analyst specializing in data visualization. Your mission is to audit an infographic image against a provided Markdown table with the topic. The judgement is splited into some dimensions, and you should only focus on the aspect of the given evaluation dimension.

# Your evaluation must be objective, precise, and strictly follow the definitions and output format below, provide a structured, multi-faceted critique.

# **Evaluation Dimension Definition:**

# **Aesthetic Quality**
#     *   **Focus:** A holistic assessment of the chart's visual appeal and professionalism.
#     *   **Process:** Provide a single score from 0 to 5 based on overall design quality. Consider factors like color scheme harmony, layout balance, clarity, typography choice, appropriate spacing, and overall polish.
#     *   **Scoring Guide:**
#         *   <1: Unusable or extremely unpleasant.
#         *   <2: Poor. Many significant aesthetic flaws.
#         *   <3: Mediocre. Functional but unappealing.
#         *   4: Good. Generally well-designed with minor flaws.
#         *   5: Excellent. Visually appealing, clear, and professional.
#     *   **Bad Case:**
#         *   Color choices make data interpretation difficult or misleading, give score around 1.
#         *   Layout is chaotic or visually distracting, give score around 1.5.
#         *   Poor spacing or alignment affects readability, give score around 2.

# **Mandatory Output Format:**

# You MUST provide your response as a single JSON object within a Markdown code block. Do not add any explanatory text outside of the JSON structure.

# ```json
# {
#     "score": <float> // A single float score from 0 to 5.
#     "detailed_explain": <string> // A detailed explanation of the score.
# }
# ```

# '''
#         self.aesthetic_quality_user_prompt = "As an expert-level Quality Assurance Analyst, please follow the detailed instructions and structured JSON output format provided in the system prompt. Analyze the provided image based on the data in the Markdown table below and perform a comprehensive evaluation on Aesthetic Quality.\nTopic:{}\nTable:{}\n\nGenerate your analysis. Your entire response must be only the final JSON object inside a Markdown code block."



class EvalStage:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.prompts = EvalPrompts()
        self.data_accuracy_cli = EvalClient(
            cfg, self.prompts.data_accuracy_system_prompt,
            check_keys=["total_data_points", "incorrect_data_points", "detailed_explain"],
            strict_check=True
        )
        self.text_rendering_cli = EvalClient(
            cfg, self.prompts.text_rendering_system_prompt,
            check_keys=["all_text_in_image", "incorrect_text_in_image"],
            strict_check=True
        )
        self.relative_relationship_cli = EvalClient(
            cfg, self.prompts.relative_relationship_system_prompt,
            check_keys=["total_data_points", "incorrect_data_points", "detailed_explain"],
            strict_check=True
        )
        self.additional_info_accuracy_cli = EvalClient(
            cfg, self.prompts.additional_info_accuracy_system_prompt,
            check_keys=["detailed_explain", "has_additional_indicator", "percentage_of_incorrect_indicator", "total_data_points", "misaligned_data_points_vs_axis", "has_additional_mark", "percentage_of_inappropriate_mark"],
            strict_check=False
        )
        # self.aesthetic_quality_cli = EvalClient(
        #     cfg, self.prompts.aesthetic_quality_system_prompt,
        #     check_keys=["score", "detailed_explain"],
        #     strict_check=True
        # )
        self.aesthetic_quality_cli = AestheticEvalClient(cfg)
    
    def cal_data_accuracy_score(self, judge_result: dict):
        if judge_result["total_data_points"] == 0:
            return 1
        return (judge_result["total_data_points"] - judge_result["incorrect_data_points"]) / judge_result["total_data_points"]
    
    def cal_text_rendering_score(self, judge_result: dict):
        total_len = len(''.join(judge_result["all_text_in_image"]))
        incorrect_len = len(''.join(judge_result["incorrect_text_in_image"]))
        if total_len == 0:
            return 0
        return (total_len - incorrect_len) / total_len
    
    def cal_relative_relationship_score(self, judge_result: dict):
        if judge_result["total_data_points"] == 0:
            return 1
        return (judge_result["total_data_points"] - judge_result["incorrect_data_points"]) / judge_result["total_data_points"]
    
    def cal_additional_info_accuracy_score(self, judge_result: dict):
        score_list = []
        has_additional_indicator = judge_result.get("has_additional_indicator", False)
        if has_additional_indicator:
            label_acc = 1 - judge_result["percentage_of_incorrect_indicator"]
            score_list.append(label_acc)
            if "total_data_points" in judge_result and "misaligned_data_points_vs_axis" in judge_result:
                if judge_result["total_data_points"] == 0:
                    data_axis_acc = 0
                else:
                    data_axis_acc = (judge_result["total_data_points"] - judge_result["misaligned_data_points_vs_axis"]) / judge_result["total_data_points"]
                score_list.append(data_axis_acc)

        has_additional_mark = judge_result.get("has_additional_mark", False)
        if has_additional_mark:
            mark_acc = 1 - judge_result["percentage_of_inappropriate_mark"]
            score_list.append(mark_acc)
        
        acc = None if len(score_list) == 0 else sum(score_list) / len(score_list)
        return acc
    
    def cal_aesthetic_quality_score(self, judge_result: dict):
        return judge_result["score"]

    def run_data_accuracy(self, image: Union[Image.Image, str], topic: str, table: str) -> dict:
        data_accuracy_user_prompt = self.prompts.data_accuracy_user_prompt.format(topic, table)
        data_accuracy_judge = self.data_accuracy_cli.eval(image, data_accuracy_user_prompt)
        if data_accuracy_judge is None:
            return None
        score = self.cal_data_accuracy_score(data_accuracy_judge)
        return {
            "score": score,
            "judge": data_accuracy_judge
        }

    def run_text_rendering(self, image: Union[Image.Image, str], topic: str, table: str) -> dict:
        text_rendering_user_prompt = self.prompts.text_rendering_user_prompt.format(topic, table)
        text_rendering_judge = self.text_rendering_cli.eval(image, text_rendering_user_prompt)
        if text_rendering_judge is None:
            return None
        score = self.cal_text_rendering_score(text_rendering_judge)
        return {
            "score": score,
            "judge": text_rendering_judge
        }
    
    def run_relative_relationship(self, image: Union[Image.Image, str], topic: str, table: str) -> dict:
        relative_relationship_user_prompt = self.prompts.relative_relationship_user_prompt.format(topic, table)
        relative_relationship_judge = self.relative_relationship_cli.eval(image, relative_relationship_user_prompt)
        if relative_relationship_judge is None:
            return None
        score = self.cal_relative_relationship_score(relative_relationship_judge)
        return {
            "score": score,
            "judge": relative_relationship_judge
        }
        
    def run_additional_info_accuracy(self, image: Union[Image.Image, str], topic: str, table: str) -> dict:
        additional_info_accuracy_user_prompt = self.prompts.additional_info_accuracy_user_prompt.format(topic, table)
        additional_info_accuracy_judge = self.additional_info_accuracy_cli.eval(image, additional_info_accuracy_user_prompt)
        if additional_info_accuracy_judge is None:
            return None
        score = self.cal_additional_info_accuracy_score(additional_info_accuracy_judge)
        return {
            "score": score,
            "judge": additional_info_accuracy_judge
        }
    
    def run_aesthetic_quality(self, image: Union[Image.Image, str], topic: str, table: str) -> dict:
        # aesthetic_quality_user_prompt = self.prompts.aesthetic_quality_user_prompt.format(topic, table)
        # aesthetic_quality_judge = self.aesthetic_quality_cli.eval(image, aesthetic_quality_user_prompt)
        aesthetic_quality_judge = self.aesthetic_quality_cli.eval(image)
        if aesthetic_quality_judge is None:
            return None
        score = self.cal_aesthetic_quality_score(aesthetic_quality_judge)
        return {
            "score": score,
            "judge": aesthetic_quality_judge
        }

    def run(self, image: Union[Image.Image, str], topic: str, table: str, task: str) -> dict:
        if task == "data_accuracy":
            return self.run_data_accuracy(image, topic, table)
        if task == "text_rendering":
            return self.run_text_rendering(image, topic, table)
        if task == "relative_relationship":
            return self.run_relative_relationship(image, topic, table)
        if task == "additional_info_accuracy":
            return self.run_additional_info_accuracy(image, topic, table)
        if task == "aesthetic_quality":
            return self.run_aesthetic_quality(image, topic, table)
    

    def load_prompt_id_2_steps(self, records_file) -> dict:
        p = Path(records_file)
        prompt_id_2_steps = dict()
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                prompt_id = obj.get("prompt_id")
                latest = obj.get("steps", [])
                prompt_id_2_steps[prompt_id] = latest
        return prompt_id_2_steps

    def report_scores_accelerate(self, prompts: List[dict]):
        # records_file = self.cfg.records_file
        self.prompt_id_2_steps = self.load_prompt_id_2_steps(self.cfg.records_file)
        scores_dict = {
            "data_accuracy": [],
            "text_rendering": [],
            "relative_relationship": [],
            "additional_info_accuracy": [],
            "aesthetic_quality": []
        }
        for prompt in tqdm(prompts, desc="Summarizing scores"):
            prompt_id = prompt['id']
            eval_steps = self.prompt_id_2_steps.get(str(prompt_id), [])
            for eval_step in eval_steps:
                task = eval_step['task']
                score = eval_step['score']
                if task == "additional_info_accuracy" and score is None:
                    continue
                scores_dict[task].append(score)
        
        report_dict = {}
        for task, scores in scores_dict.items():
            report_dict[task] = {
                "count": len(scores),
                "mean": np.mean(scores),
                "median": np.median(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "percentiles": {
                    "25": np.percentile(scores, 25),
                    "50": np.percentile(scores, 50),
                    "75": np.percentile(scores, 75),
                    "90": np.percentile(scores, 90),
                    "95": np.percentile(scores, 95),
                }
            }
        print(report_dict)
        
        with open(self.cfg.metric_file, "w") as f:
            json.dump(report_dict, f, indent=4)

    def report_scores(self, prompts: List[dict]):
        records_file = self.cfg.records_file
        scores_dict = {
            "data_accuracy": [],
            "text_rendering": [],
            "relative_relationship": [],
            "additional_info_accuracy": [],
            "aesthetic_quality": []
        }
        for prompt in tqdm(prompts, desc="Summarizing scores"):
            prompt_id = prompt['id']
            eval_steps = load_latest_steps(records_file, str(prompt_id))
            for eval_step in eval_steps:
                task = eval_step['task']
                score = eval_step['score']
                if task == "additional_info_accuracy" and score is None:
                    continue
                scores_dict[task].append(score)
        
        report_dict = {}
        for task, scores in scores_dict.items():
            report_dict[task] = {
                "count": len(scores),
                "mean": np.mean(scores),
                "median": np.median(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "percentiles": {
                    "25": np.percentile(scores, 25),
                    "50": np.percentile(scores, 50),
                    "75": np.percentile(scores, 75),
                    "90": np.percentile(scores, 90),
                    "95": np.percentile(scores, 95),
                }
            }
        print(report_dict)
        
        with open(self.cfg.metric_file, "w") as f:
            json.dump(report_dict, f, indent=4)
    
    def cal_single_avg_score(self, record):
        if not record:
            return -100
        task2score = {r['task']: r['score'] for r in record}
        if task2score.get("additional_info_accuracy") is None:
            score = (100 * (task2score['data_accuracy'] + task2score['text_rendering'] + task2score['relative_relationship'])
                 + 10 * task2score['aesthetic_quality']) / 4
        else:
            score = (100 * (task2score['data_accuracy'] + task2score['text_rendering'] + task2score['relative_relationship'] + task2score['additional_info_accuracy'])
                 + 10 * task2score['aesthetic_quality']) / 5
        return score

    def report_final_scores(self, prompts: List[dict]):
        infer_records_file = Path(self.cfg.records_file).parent / "visual_success_records.jsonl"
        gen_records_file = self.cfg.records_file.replace("final", "image_gen")
        edit0_records_file = self.cfg.records_file.replace("final", "edit_0")
        edit1_records_file = self.cfg.records_file.replace("final", "edit_1")
        latest_records_file = self.cfg.records_file.replace("final", "latest")
        assert Path(infer_records_file).exists(), f"{infer_records_file} not exists"
        assert Path(gen_records_file).exists(), f"{gen_records_file} not exists"
        assert Path(edit0_records_file).exists(), f"{edit0_records_file} not exists"
        assert Path(edit1_records_file).exists(), f"{edit1_records_file} not exists"
        assert Path(latest_records_file).exists(), f"{latest_records_file} not exists"

        self.prompt_id_2_infer_steps = self.load_prompt_id_2_steps(infer_records_file)
        self.prompt_id_2_gen_steps = self.load_prompt_id_2_steps(gen_records_file)
        self.prompt_id_2_edit0_steps = self.load_prompt_id_2_steps(edit0_records_file)
        self.prompt_id_2_edit1_steps = self.load_prompt_id_2_steps(edit1_records_file)
        self.prompt_id_2_latest_steps = self.load_prompt_id_2_steps(latest_records_file)
        scores_dict = {
            "data_accuracy": [],
            "text_rendering": [],
            "relative_relationship": [],
            "additional_info_accuracy": [],
            "aesthetic_quality": []
        }

        final_records = []
        final_roles = []
        unseccess_count = 0
        for prompt in tqdm(prompts, desc="Summarizing scores"):
            prompt_id = prompt['id']
            infer_steps = self.prompt_id_2_infer_steps.get(str(prompt_id), [])
            image_roles = ["image_gen"] + [f"edit_{k}" for k in range(3)]
            valid_image_steps = []
            for step in infer_steps:
                if step['role'] in image_roles and step['content'] != "Input data may contain inappropriate content.":
                    valid_image_steps.append(step)
            vaild_image_roles = [step['role'] for step in valid_image_steps]
            
            gen_steps = self.prompt_id_2_gen_steps.get(str(prompt_id), [])
            edit0_steps = self.prompt_id_2_edit0_steps.get(str(prompt_id), [])
            edit1_steps = self.prompt_id_2_edit1_steps.get(str(prompt_id), [])
            latest_steps = self.prompt_id_2_latest_steps.get(str(prompt_id), [])

            all_evaled_steps = [gen_steps, edit0_steps, edit1_steps, latest_steps]
            all_evaled_roles = ["image_gen", "edit_0", "edit_1", "latest"]
            if "edit_2" in vaild_image_roles:
                unseccess_count += 1
                scores = [self.cal_single_avg_score(steps) for steps in all_evaled_steps]
                max_index = np.argmax(scores)
                final_role = all_evaled_roles[max_index]
                final_steps = all_evaled_steps[max_index]
            else:
                final_role = vaild_image_roles[-1]
                final_steps = all_evaled_steps[all_evaled_roles.index(final_role)]
            
            final_roles.append(final_role)
            final_records.append({
                "prompt_id": prompt_id,
                "role": final_role,
                "steps": final_steps,
                "topic": prompt['topic'],
                "table": prompt['table'],
            })
            for eval_step in final_steps:
                task = eval_step['task']
                score = eval_step['score']
                if task == "additional_info_accuracy" and score is None:
                    continue
                scores_dict[task].append(score)

        print("Role summary:", Counter(final_roles))
        print("Unseccess count:", unseccess_count)

        with open(self.cfg.records_file, "w") as f:
            for record in final_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        report_dict = {}
        for task, scores in scores_dict.items():
            report_dict[task] = {
                "count": len(scores),
                "mean": np.mean(scores),
                "median": np.median(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "percentiles": {
                    "25": np.percentile(scores, 25),
                    "50": np.percentile(scores, 50),
                    "75": np.percentile(scores, 75),
                    "90": np.percentile(scores, 90),
                    "95": np.percentile(scores, 95),
                }
            }
        print(report_dict)
        
        with open(self.cfg.metric_file, "w") as f:
            json.dump(report_dict, f, indent=4)


