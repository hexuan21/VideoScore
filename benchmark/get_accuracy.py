import json
import numpy as np
import scipy.stats as stats
import os
import fire

ROUND_DIGIT=4

def cal_accuracy(
    result_dir: str="./benchmark/eval_results/video_feedback/",
    bench_name: str="video_feedback"
):
    for result_file in sorted(os.listdir(result_dir)):
        if not result_file.startswith("eval_"):
            continue
        # result_file example: eval_video_feedback_videoscore.json
        model_name=result_file.split(f"{bench_name}_")[1].split(".")[0]
        all_res=json.load(open(f"{result_dir}/{result_file}","r"))
        all_ref_scores=[eval(item["ref"]) for item in all_res]
        all_ans_scores=[eval(item["ans"]) for item in all_res]
        matched_num_list=[0 for _ in range(len(all_ref_scores[0]))]
        acc_list=[]
        # try:
        for ref_list, ans_list in zip(all_ref_scores,all_ans_scores):
            
            for i,(ref,ans) in enumerate(zip(ref_list,ans_list)):
                if abs(float(ref)-float(round(ans)))<1e-3:
                    matched_num_list[i]+=1            
        acc_list=[round(x/len(all_ref_scores),ROUND_DIGIT) for x in matched_num_list]
        # except Exception as e:
        #     print(e)
        #     acc_list=[None for _ in range(len(all_ref_scores[0]))]
        print(f"bench name:{bench_name}, model name: {model_name}, accuracy: {acc_list}")
        
        dirname=os.path.dirname(f"{result_dir}/{result_file}")
        acc_file=f"{dirname}/accuracy_{bench_name}.json"
        
        if not os.path.exists(acc_file):
            all_acc=[]
        else:
            all_acc=json.load(open(acc_file,"r"))
        all_acc.append({model_name:
                {
                "acc_list":acc_list,
                }
            })
        
        with open(acc_file,"w") as file:
            json.dump(all_acc,file,indent=4)
        
if __name__ == "__main__":
    fire.Fire(cal_accuracy)