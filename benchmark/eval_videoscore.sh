# # ===== eval VideoScore on VideoFeedback-test =====
# mkdir -p ./eval_results/video_feedback

# model_repo_name="TIGER-Lab/VideoScore"
# data_repo_name="TIGER-Lab/VideoScore-Bench"
# bench_name="video_feedback"
# frames_dir="../data/${bench_name}/test"
# name_postfixs="[${bench_name}]"
# result_file="./eval_results/${bench_name}/eval_${bench_name}_videoscore.json"


# CUDA_VISIBLE_DEVICES=0 python eval_videoscore.py --model_repo_name $model_repo_name \
#     --data_repo_name $data_repo_name \
#     --frames_dir $frames_dir  \
#     --name_postfixs $name_postfixs \
#     --result_file $result_file \
#     --bench_name $bench_name \




# # ===== eval VideoScore on EvalCrafter =====
# mkdir -p ./eval_results/eval_crafter

# # we use this variant "TIGER-Lab/VideoScore-anno-only" except for VideoFeedback-test
# model_repo_name="TIGER-Lab/VideoScore-anno-only"
# data_repo_name="TIGER-Lab/VideoScore-Bench"
# bench_name="eval_crafter"
# frames_dir="../data/${bench_name}/test"
# name_postfixs="[${bench_name}]"
# result_file="./eval_results/${bench_name}/eval_${bench_name}_videoscore.json"

# CUDA_VISIBLE_DEVICES=1 python eval_videoscore.py --model_repo_name $model_repo_name \
    # --data_repo_name $data_repo_name \
    # --frames_dir $frames_dir \
    # --name_postfixs $name_postfixs \
    # --result_file $result_file \
    # --bench_name $bench_name




# # ===== eval VideoScore on GenAI-Bench =====
# mkdir -p ./eval_results/genaibench

# # we use this variant "TIGER-Lab/VideoScore-anno-only" except for VideoFeedback-test
# model_repo_name="TIGER-Lab/VideoScore-anno-only"
# data_repo_name="TIGER-Lab/VideoScore-Bench"
# bench_name="genaibench"
# frames_dir="../data/${bench_name}/test"
# name_postfixs="[${bench_name}]"
# result_file="./eval_results/${bench_name}/eval_${bench_name}_videoscore.json"


# CUDA_VISIBLE_DEVICES=2 python eval_videoscore.py --model_repo_name $model_repo_name \
#     --data_repo_name $data_repo_name \
#     --frames_dir $frames_dir \
#     --name_postfixs $name_postfixs \
#     --result_file $result_file \
#     --bench_name $bench_name




# ===== eval VideoScore on VBench =====
mkdir -p ./benchmark/eval_results/vbench

# we use this variant "TIGER-Lab/VideoScore-anno-only" except for VideoFeedback-test
model_repo_name="TIGER-Lab/VideoScore-anno-only"
data_repo_name="TIGER-Lab/VideoScore-Bench"
bench_name="vbench"
frames_dir="../data/${bench_name}/test"
name_postfixs="[${bench_name}]"
result_file="./eval_results/${bench_name}/eval_${bench_name}_videoscore.json"


python eval_videoscore.py --model_repo_name $model_repo_name \
    --data_repo_name $data_repo_name \
    --frames_dir $frames_dir \
    --name_postfixs $name_postfixs \
    --result_file $result_file \
    --bench_name $bench_name