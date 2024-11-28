bench_name="video_feedback"
# bench_name="eval_crafter"

eval_res_dir="eval_results"

result_dir="./${eval_res_dir}/${bench_name}"
python get_spearman_corr.py --result_dir $result_dir --bench_name $bench_name