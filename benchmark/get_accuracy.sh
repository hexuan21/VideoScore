bench_name="video_feedback"

result_dir="./eval_results/${bench_name}"
python get_accuracy.py --result_dir $result_dir --bench_name $bench_name