import enum
import json
import os
import re
import cv2
import copy
import random
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from datasets import Dataset
from benchmark.utils_tools import REGRESSION_QUERY_TEMPLATE
import av
import numpy as np
from typing import List
from PIL import Image
import torch
import time
from transformers import AutoProcessor
from mantis.models.idefics2 import Idefics2ForSequenceClassification
from mantis.models.qwen2_vl import Qwen2VLForSequenceClassification
from transformers import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info


def add_bad_t2v(src_path,dest_path,select_num):
    HTTP_PREFIX="https://huggingface.co/datasets/hexuan21/VideoFeedback-videos-mp4/blob/main"
    
    data=json.load(open(src_path,"r", encoding="utf-8"))
    all_prompt_list=[x["text prompt"] for x in data]
    
    print(len(data))
    print(data[0].keys())
    
    for idx,item in enumerate(data):
        curr_prompt=data[idx]["text prompt"]
        id=data[idx]["id"]
        data[idx]["video link"]=f"{HTTP_PREFIX}/{id[0]}/{id}.mp4"
        data[idx]["conversations"][0]["value"]=REGRESSION_QUERY_TEMPLATE.format(text_prompt=curr_prompt.strip())
        data[idx]["conversations"][0]["value"]+="\n"+"<image> "*len(data[idx]["images"])
    
    selected=random.sample(data,select_num)
    selected=[copy.deepcopy(x) for x in selected]
    
    for idx,item in tqdm(enumerate(selected)):
        if idx<int(select_num*0.25):
            selected[idx]["text prompt"]=""
            selected[idx]["text-to-video alignment"]=0
            selected[idx]["conversations"][0]["value"]=REGRESSION_QUERY_TEMPLATE.format(text_prompt="")
            selected[idx]["conversations"][0]["value"]+="\n"+"<image> "*len(selected[idx]["images"])
            model_output=selected[idx]["conversations"][1]["value"]
            selected[idx]["conversations"][1]["value"]=re.sub(r'(text-to-video alignment:)(..)', r'\1 0', model_output)
        else:
            curr_prompt=selected[idx]["text prompt"]
            random_prompt=random.choice(all_prompt_list)
            while random_prompt==curr_prompt:
                random_prompt=random.choice(all_prompt_list)
            
            selected[idx]["text prompt"]=random_prompt
            selected[idx]["text-to-video alignment"]=0
            selected[idx]["conversations"][0]["value"]=REGRESSION_QUERY_TEMPLATE.format(text_prompt=random_prompt.strip())
            selected[idx]["conversations"][0]["value"]+="\n"+"<image> "*len(selected[idx]["images"])
            model_output=selected[idx]["conversations"][1]["value"]
            selected[idx]["conversations"][1]["value"]=re.sub(r'(text-to-video alignment:)(..)', r'\1 0', model_output)
            
    os.makedirs(os.path.dirname(dest_path),exist_ok=True)
    with open(dest_path,"w") as f:
        json.dump(data+selected,f,indent=4)
    
    print(len(json.load(open(dest_path,"r"))))


def upload(path,remote_config_name):
    json_data=[]
    with open(path,"r") as f:
        json_data.extend(json.load(f))
    
    repo_id="hexuan21/VideoFeedback_new"
    split="train"
    token=os.environ["HF_TOKEN"]
    
    hf_dataset=Dataset.from_list(json_data,split=split)
    hf_dataset.push_to_hub(
        repo_id=repo_id,
        split=split,
        token=token,
        config_name=remote_config_name,
        commit_message=f"add {remote_config_name} {split} data",
        revision="main"
    )


def conv_to_video(input_dir,output_dir,fps=8):
    os.makedirs(output_dir,exist_ok=True)
    for subdir in tqdm(sorted(os.listdir(input_dir))):
        if "eval_crafter" in input_dir:
            if "gen2" in subdir or "pika" in subdir:
                fps=24
        
        image_folder=f'{input_dir}/{subdir}'
        images = sorted([img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))])
        
        if not images:
            print("No images found in the folder.")
            return

        first_image_path = os.path.join(image_folder, images[0])
        frame = cv2.imread(first_image_path)
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        output_path=f"{output_dir}/{subdir}.mp4"
        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for image in images:
            image_path = os.path.join(image_folder, image)
            frame = cv2.imread(image_path)
            video.write(frame)

        video.release()
        video = VideoFileClip(output_path)
        video.write_videofile(output_path, codec="libx264",fps=fps)





def _read_video_pyav(
    container,
    indices,
):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])



def run_inference_idefics2(video_path,video_prompt,max_frames):
    ROUND_DIGIT=3
    REGRESSION_QUERY_PROMPT = """
    Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
    please watch the following frames of a given video and see the text prompt for generating the video,
    then give scores from 5 different dimensions:
    (1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color
    (2) temporal consistency, both the consistency of objects or humans and the smoothness of motion or movements
    (3) dynamic degree, the degree of dynamic changes
    (4) text-to-video alignment, the alignment between the text prompt and the video content
    (5) factual consistency, the consistency of the video content with the common-sense and factual knowledge

    for each dimension, output a float number from 1.0 to 4.0,
    the higher the number is, the better the video performs in that sub-score, 
    the lowest 1.0 means Bad, the highest 4.0 means Perfect/Real (the video is like a real video)
    Here is an output example:
    visual quality: 3.2
    temporal consistency: 2.7
    dynamic degree: 4.0
    text-to-video alignment: 2.3
    factual consistency: 1.8

    For this video, the text prompt is "{text_prompt}",
    all the frames of video are as follows:
    """
    

    # sample uniformly 8 frames from the video
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    if total_frames > max_frames:
        indices = np.arange(0, total_frames, total_frames / max_frames).astype(int)
    else:
        indices = np.arange(total_frames)

    frames = [Image.fromarray(x) for x in _read_video_pyav(container, indices)]
    eval_prompt = REGRESSION_QUERY_PROMPT.format(text_prompt=video_prompt)
    num_image_token = eval_prompt.count("<image>")
    if num_image_token < len(frames):
        eval_prompt += "<image> " * (len(frames) - num_image_token)

    flatten_images = []
    for x in [frames]:
        if isinstance(x, list):
            flatten_images.extend(x)
        else:
            flatten_images.append(x)
    s_t=time.time()
    flatten_images = [Image.open(x) if isinstance(x, str) else x for x in flatten_images]
    # print(f"len(flatten_images): {len(flatten_images)}")
    inputs = processor(text=eval_prompt, images=flatten_images, return_tensors="pt")
    # print(f"len(inputs): {len(inputs)}")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    num_aspects = logits.shape[-1]

    aspect_scores = []
    for i in range(num_aspects):
        aspect_scores.append(round(logits[0, i].item(),ROUND_DIGIT))
    print(aspect_scores)

    # print(round(time.time()-s_t,ROUND_DIGIT))

    """
    model output on visual quality, temporal consistency, dynamic degree,
    text-to-video alignment, factual consistency, respectively

    [2.297, 2.469, 2.906, 2.766, 2.516]
    """

def run_inference_qwen2_vl(video_or_frames_path,video_prompt,fps):
    ROUND_DIGIT=3
    REGRESSION_QUERY_PROMPT = """
    Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
    please watch the following frames of a given video and see the text prompt for generating the video,
    then give scores from 5 different dimensions:
    (1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color
    (2) temporal consistency, both the consistency of objects or humans and the smoothness of motion or movements
    (3) dynamic degree, the degree of dynamic changes
    (4) text-to-video alignment, the alignment between the text prompt and the video content
    (5) factual consistency, the consistency of the video content with the common-sense and factual knowledge

    for each dimension, output a float number from 1.0 to 4.0,
    the higher the number is, the better the video performs in that sub-score, 
    the lowest 1.0 means Bad, the highest 4.0 means Perfect/Real (the video is like a real video)
    Here is an output example:
    visual quality: 3.2
    temporal consistency: 2.7
    dynamic degree: 4.0
    text-to-video alignment: 2.3
    factual consistency: 1.8

    For this video, the text prompt is "{text_prompt}",
    all the frames of video are as follows:
    """    

    # Messages containing a images list as a video and a text query
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_or_frames_path,
                    "fps": fps,
                },
                {"type": "text", "text": REGRESSION_QUERY_PROMPT.format(text_prompt=video_prompt)},
            ],
        }
    ]

    # image_list=[f"{video_or_frames_path}/{x}" for x in sorted(os.listdir(video_or_frames_path))]
    # # image_list=[f"video1/video1_{i:02d}.jpg" for i in range(24)]   

    # messages = [
    #     {
    #         "role": "user",
    #         "content": [],
    #     }
    # ]
    # messages[0]["content"].append({"type": "video", "video": image_list, "fps":8.0})
    # messages[0]["content"].append({"type": "text", "text": REGRESSION_QUERY_PROMPT.format(text_prompt=video_prompt)})

    s_t=time.time()
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
    print(aspect_scores)
    # print(round(time.time()-s_t,ROUND_DIGIT))



if __name__ == "__main__":
    # src_path="/data/xuan/video_eval/VideoFeedback/v24.06/data_annotated.json"
    # dest_path="/data/xuan/video_eval/VideoFeedback/v24.11/data_annotated.json"
    # select_num=4500
    
    # src_path="/data/xuan/video_eval/VideoFeedback/v24.06/data_real.json"
    # dest_path="/data/xuan/video_eval/VideoFeedback/v24.11/data_real.json"
    # select_num=500
    
    # add_bad_t2v(src_path,dest_path,select_num)
    
    
    
    # path="/data/xuan/video_eval/VideoFeedback/v24.11/data_annotated.json"
    # subset="annotated"
    # upload(path,subset)
            
    
    
    # video_feedback: 8fps 8,16,24
    # genaibench: 8fps 16
    # vbench: 8fps 16, 33
    # eval_crafter: 
    # floor33-16 8fps
    # ms-32 8fps
    # zs-35 8fps
    # zs-36 8fps
    # pika-72 24fps
    # gen2-96 24fps
    
    # bench_name="eval_crafter"
    # input_dir=f"/home/brantley/workdir/VideoScore/data/{bench_name}/test/frames_{bench_name}"
    # output_dir=f"/home/brantley/workdir/VideoScore/data/{bench_name}/test/videos"
    
    # conv_to_video(input_dir, output_dir, fps=8)
    
    
    
    
    video_path="examples/video2.mp4"
    # original_prompt="Near the Elephant Gate village, they approach the haunted house at night. Rajiv feels anxious, but Bhavesh encourages him. As they reach the house, a mysterious sound in the air adds to the suspense."
    original_prompt="elon musk smiling infront of camera  and having btc coin in hand"
    prompt_list=[original_prompt]+["",
                                   "elon musk smiling infront of camera",
                                   "albert einstein smiling infront of camera  and having btc coin in hand",
                                   "smiling infront of camera  and having btc coin in hand",
                                           "a horse is running ",
                                           "a girl surrounded by sparkling gold, clear circle on the surface with some ripples, C4d, minimalist stage design, soft and dreamy depictions, atmospheric serenity, linear illustration, light gold and white",
                                           "young beautiful wealthy woman with short blonde hair, standing with a glass of champagne in a huge dressing room with high ceilings of a townhouse, Art Nouveau interior, interior architect Alvar Aalto, bright cheerful photo ",
                                           "A cute aesthetic girl in a pretty dress playing arcade",
                                           "in a futuristic classroom with sun light windows, a young teacher is teaching in front of a large screen, she face to students, the screen shows robot design diagram.",
                                           "Albert Einstein talking for 30 seconds at 16:9 resolution with greensreen bakground",
                                           "film noir, cinematic, a woman and two men are sitting in a dark room, their faces are not visible, camera zoom out",
                                           "a mage doing a spell in the middle of the forest, with a big moon behind, realism style  ",]

    MAX_FRAMES=24
    # model_name="Mantis-VL/mantis-8b-idefics2-video-eval_5184_regression"
    # model_name="Mantis-VL/mantis-8b-idefics2-video-eval_6144_regression"
    model_name="TIGER-Lab/VideoScore-v1.1"
    processor = AutoProcessor.from_pretrained(model_name,torch_dtype=torch.bfloat16)
    model = Idefics2ForSequenceClassification.from_pretrained(model_name,torch_dtype=torch.bfloat16).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for video_prompt in prompt_list:
        run_inference_idefics2(video_path,video_prompt,max_frames=MAX_FRAMES)
    

    # FPS=8.0
    # # model_name="Mantis-VL/qwen2-vl-video-eval-241120_46080_regression"
    # # model_name="Mantis-VL/qwen2-vl-video-eval_49152_regression"
    # model_name="Mantis-VL/qwen2-vl-video-eval_55296_regression"
    # model = Qwen2VLForSequenceClassification.from_pretrained(
    #     model_name, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2"
    # )
    # processor = Qwen2VLProcessor.from_pretrained(model_name)
    # for video_prompt in prompt_list:
    #     run_inference_qwen2_vl(video_path,video_prompt,FPS)
    
    
    
    
    # from huggingface_hub import login,upload_file
    # repo_id="TIGER-Lab/VideoScore-v1.1"
    # HF_TOKEN='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'
    # login(token=HF_TOKEN)
    
    # dir="/home/brantley/.cache/huggingface/hub/models--Mantis-VL--mantis-8b-idefics2-video-eval_6144_regression/snapshots/5a68ff87a1dc86e14bb4d53bdb0a75bc7896b878/"
    # for f in os.listdir(dir):
    #     path=f"{dir}/{f}"
    #     upload_file(
    #         path_or_fileobj=path,
    #         path_in_repo=f,
    #         repo_id=repo_id,
    #         repo_type="model",   
    #     )