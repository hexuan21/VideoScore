mkdir -p ./benchmark/eval_results/video_feedback

model_repo_name="TIGER-Lab/MantisScore"
data_repo_name="TIGER-Lab/VideoFeedback-Bench"
frames_dir="./data/video_feedback/test"
name_postfixs="['video_feedback']"
result_file='./benchmark/eval_results/video_feedback/eval_video_feedback_mantisscore.json'
python benchmark/eval_mantis_score.py --model_repo_name $model_repo_name --data_repo_name $data_repo_name --frames_dir $frames_dir  --name_postfixs $name_postfixs --result_file $result_file 