import os
import re
import torch
import fire
import logging
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import List
from datetime import datetime
from mantis.models.qwen2_vl import Qwen2VLForSequenceClassification
from transformers import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from utils_tools import _add_to_res_file,regression_query_template
from utils_conv import conv_templates


CONV_TEMPLATE = conv_templates["idefics_2"]
NUM_ASPECT = 5
ROUND_DIGIT = 3
FPS = 8.0
BENCH_NAMES=["video_feedback","eval_crafter","vbench","genaibench"]
REGRESSION_QUERY_TEMPLATE = regression_query_template()


def _read_video_frames(
    frame_paths:List[str], 
    max_frames:int,
):
    
    total_frames = len(frame_paths)
    indices = np.linspace(0, total_frames - 1, num=max_frames).astype(int)
    selected_frames = [np.array(Image.open(frame_paths[i])) for i in indices]
    return np.stack(selected_frames)


def _model_output(
    model: Qwen2VLForSequenceClassification,
    processor: Qwen2VLProcessor,
    video_prompt: str, 
    frames_path_list: List[str],
):
    messages = [
        {
            "role": "user",
            "content": [],
        }
    ]
    
    messages[0]["content"].append({"type": "video", "video": frames_path_list, "fps":FPS})
    messages[0]["content"].append({"type": "text", "text": REGRESSION_QUERY_TEMPLATE.format(text_prompt=video_prompt)})
    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    # print(inputs['input_ids'].shape)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    num_aspects = logits.shape[-1]

    aspect_scores = []
    for i in range(num_aspects):
        aspect_scores.append(round(logits[0, i].item(),ROUND_DIGIT))
    
    return aspect_scores


def main(
    model_repo_name: str="TIGER-Lab/VideoScore-Qwen2-VL",
    data_repo_name: str="TIGER-Lab/VideoScore-Bench",
    frames_dir: str="../data/video_feedback/test", 
    name_postfixs: List[str]=['video_feedback'], 
    result_file: str="./eval_results/video_feedback/eval_video_feedback_videoscore_qwen2_vl.json",
    bench_name: str="video_feedback",
):
    '''
    evalualte VideoScore-Qwen2-VL model on VideoScore-Bench which contains four benchmarks, save results to 'result_file' 
    and calculate spearman correlation coefficient between human-annotated references and model output.
    '''
    
    logging.basicConfig(level=logging.INFO)
    logger= logging.getLogger(__name__)
    date_time=datetime.now().strftime("%m-%d %H:%M:%S")
    log_file=f"./logs/{bench_name}/eval_videoscore_qwen2_vl_on_{bench_name}_{date_time}.log"
    os.makedirs(os.path.dirname(log_file),exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    processor = Qwen2VLProcessor.from_pretrained(model_repo_name)
    model = Qwen2VLForSequenceClassification.from_pretrained(
        model_repo_name, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("processor and model loaded")
    
    for source in name_postfixs:
        test_data=load_dataset(data_repo_name,name=source, split="test")
        
        curr_frames_dir=f"{frames_dir}/frames_{source}"
        
        for idx, item in tqdm(enumerate(test_data)):
            vid=item["id"]
            frame_path_list=[f"{curr_frames_dir}/{vid}/{img}" for img in item["images"]]
            
            human_text=item["conversations"][0]["value"]
            bot_text=item["conversations"][1]["value"]
            
            video_prompt=human_text.split("text prompt is \"")[1].split("\",\n")[0]
            assert bench_name in BENCH_NAMES, "benchmark name is not supported"
            if bench_name=="video_feedback":
                ref_scores=[int(item) for item in re.findall(r': (\d+)', bot_text)]
            else:
                ref_scores=item["score_list"]
            
            ans_scores=_model_output(model, processor, video_prompt, frame_path_list)

            logger.info(f"{idx} {vid} {ans_scores}")
            curr_compare_dict={
                "id":vid,
                "text":video_prompt,
                "ref":f"{ref_scores}",
                "ans":f"{ans_scores}"
            }
            _add_to_res_file(result_file,curr_compare_dict)
                

if __name__ == "__main__":
    fire.Fire(main)
