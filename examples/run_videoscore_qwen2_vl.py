"""
pip install qwen_vl_utils mantis-vl
"""
import torch
from mantis.models.qwen2_vl import Qwen2VLForSequenceClassification
from transformers import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info

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
video_path="video1.mp4"

# model_name="TIGER-Lab/VideoScore-Qwen2-VL"

video_prompt="Near the Elephant Gate village, they approach the haunted house at night. Rajiv feels anxious, but Bhavesh encourages him. As they reach the house, a mysterious sound in the air adds to the suspense."
# [3.609, 3.594, 3.719, 3.156, 3.703]
# video_prompt="a man is running in the forest"
# [3.109, 3.172, 3.391, 3.094, 3.109]
# video_prompt="" 
# [2.453, 2.562, 2.781, 2.375, 2.484]


model_name="Mantis-VL/qwen2-vl-video-eval_49152_regression"

# video_prompt="Near the Elephant Gate village, they approach the haunted house at night. Rajiv feels anxious, but Bhavesh encourages him. As they reach the house, a mysterious sound in the air adds to the suspense."
# [2.516, 2.656, 2.844, 2.453, 2.812]
# video_prompt="a man is running in the forest"
# [2.609, 2.641, 2.875, 2.375, 2.75]
# video_prompt="" 
# [2.609, 2.734, 2.719, 0.941, 2.844]


# default: Load the model on the available device(s)
model = Qwen2VLForSequenceClassification.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2"
)

# default processer
processor = Qwen2VLProcessor.from_pretrained(model_name)

# Messages containing a images list as a video and a text query
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                "fps": 24.0,
            },
            {"type": "text", "text": REGRESSION_QUERY_PROMPT.format(text_prompt=video_prompt)},
        ],
    }
]


# image_list=[f"video1/video1_{i:02d}.jpg" for i in range(24)]   
# frames_path_for_qwen=[f"file://{x}" for x in image_list] 

# messages = [
#     {
#         "role": "user",
#         "content": [],
#     }
# ]

# messages[0]["content"].append({"type": "video", "video": image_list, "fps":8.0})
# messages[0]["content"].append({"type": "text", "text": REGRESSION_QUERY_PROMPT.format(text_prompt=video_prompt)})

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
print(inputs['input_ids'].shape)

# Inference
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
num_aspects = logits.shape[-1]

aspect_scores = []
for i in range(num_aspects):
    aspect_scores.append(round(logits[0, i].item(),ROUND_DIGIT))
print(aspect_scores)


"""
model output on visual quality, temporal consistency, dynamic degree,
text-to-video alignment, factual consistency, respectively

[3.578, 3.594, 3.703, 3.156, 3.688]
"""








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




"""
Suppose you are an expert in judging and evaluating the quality of AI-generated videos, 
please watch the following frames of a given video and see the text prompt for generating the video, 
then give scores from 5 different dimensions:
(1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color
(2) temporal consistency, the consistency of objects or humans in video
(3) dynamic degree, the degree of dynamic changes
(4) text-to-video alignment, the alignment between the text prompt and the video content
(5) factual consistency, the consistency of the video content with the common-sense and factual knowledge

For each dimension, output a number from [1,2,3,4], 
in which '1' means 'Bad', '2' means 'Average', '3' means 'Good', 
'4' means 'Real' or 'Perfect' (the video is like a real video)
Here is an output example:
visual quality: 4
temporal consistency: 4
dynamic degree: 3
text-to-video alignment: 1
factual consistency: 2

For this video, the text prompt is ,
all the frames of video are as follows: "
"""
