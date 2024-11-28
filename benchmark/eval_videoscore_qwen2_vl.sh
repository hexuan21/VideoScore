
bench_name="video_feedback"
# bench_name="eval_crafter"
# bench_name="genaibench"
# bench_name="vbench"

eval_res="./eval_results_qwen2_vl_new_data_55296_video_frames"
mkdir -p "${eval_res}/${bench_name}"

# model_repo_name="Mantis-VL/qwen2-vl-video-eval-241120_46080_regression"
# model_repo_name="Mantis-VL/qwen2-vl-video-eval_49152_regression"
model_repo_name="Mantis-VL/qwen2-vl-video-eval_55296_regression"

data_repo_name="TIGER-Lab/VideoScore-Bench"
frames_dir="../data/${bench_name}/test"
name_postfixs="[${bench_name}]"
result_file="./${eval_res}/${bench_name}/eval_${bench_name}_videoscore_qwen2_vl.json"

CUDA_VISIBLE_DEVICES=0 python eval_videoscore_qwen2_vl.py --model_repo_name $model_repo_name \
    --data_repo_name $data_repo_name \
    --frames_dir $frames_dir  \
    --name_postfixs $name_postfixs \
    --result_file $result_file \
    --bench_name $bench_name


