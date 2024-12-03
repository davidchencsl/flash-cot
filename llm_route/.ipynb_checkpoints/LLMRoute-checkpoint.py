import pandas as pd
import re
from llm import batch_flash_cot, batch_inference
from .ArcClassify import prepare_arc_data, process_data, route_base_prompt
import json
import time

question_base_prompt = """
    Please choose one answer from the choices for the following question. You need to output the answer in the following format, the answer can be a letter or a number:
    <ANSWER>your answer</ANSWER>
"""

def parse_answer(text):
    match = re.search(r"<ANSWER>(.*?)</ANSWER>", text)
    answer = None
    if match:
        answer = match.group(1)
    return answer


def run_single(data_list, model):
    results = []

    tot_cnt = 0
    correct_cnt = 0
    start_time = None
    end_time = None
    
    for idx, (qid, q, c, a, label) in enumerate(data_list):
        item = f"Here is the question and choices: \nquestion:{q}\nchoices:\n{c}\n Please determine the answer."
        question_prompt = question_base_prompt + item
        answer_response = batch_inference(model, [question_prompt])[0]
        answer = parse_answer(answer_response)
        if idx == 0:
            start_time = time.time()
        if answer is None:
            # print(f"parse answer failed, id: {qid}, answer_text: {answer_response}")
            continue
        is_correct = (answer.lower() == a.lower())
        if is_correct:
            correct_cnt += 1
        tot_cnt += 1
        results.append(
            {
                "id": qid,
                "question": q,
                "answer": a,
                "predicted": answer,
                "is_correct": is_correct,
            }
        )
        if (idx + 1) % 100 == 0:
            print(f"processed {idx + 1} item, correct cnt: {correct_cnt}, valid cnt: {tot_cnt}, correct rate: {correct_cnt / tot_cnt}")
    end_time = time.time()
    time_slot = end_time - start_time
    print(f"processed {len(data_list)} item, correct cnt: {correct_cnt}, valid cnt: {tot_cnt}, correct rate: {correct_cnt / tot_cnt}, time: {time_slot}")
    return results


def run_router(data_list, route_model, easy_model, hard_model, mode='fusion'):
    results = []

    tot_cnt = 0
    correct_cnt = 0
    for idx, (qid, q, c, a, label) in enumerate(data_list):
        route_item = f"Here is the question and choices: \nquestion:{q}\nchoices:\n{c}\n Please determine the output."
        route_prompt = route_base_prompt + route_item
        route_sign = batch_inference(route_model, [route_prompt])[0]

        item = f"Here is the question and choices: \nquestion:{q}\nchoices:\n{c}\n Please determine the answer."
        question_prompt = question_base_prompt + item
        answer_response = None
        if '0' in route_sign:
            answer_response = batch_inference(easy_model, [question_prompt])[0]
        else:
            answer_response = call_hard_model(easy_model, hard_model, mode, question_prompt)

        answer = parse_answer(answer_response)
        if idx == 0:
            start_time = time.time()
        if answer is None:
            # print(f"parse answer failed, id: {qid}, answer_text: {answer_response}")
            continue
        
        is_correct = (answer.lower() == a.lower())
        if is_correct:
            correct_cnt += 1
        tot_cnt += 1
        results.append(
            {
                "id": qid,
                "question": q,
                "answer": a,
                "predicted": answer,
                "is_correct": is_correct,
            }
        )
        if (idx + 1) % 100 == 0:
            print(f"processed {idx + 1} item, correct cnt: {correct_cnt}, valid cnt: {tot_cnt}, correct rate: {correct_cnt / tot_cnt}")
    end_time = time.time()
    time_slot = end_time - start_time
    print(f"processed {len(data_list)} item, correct cnt: {correct_cnt}, valid cnt: {tot_cnt}, correct rate: {correct_cnt / tot_cnt}, time: {time_slot}")
    return results


def call_hard_model(easy_model, hard_model, mode, prompt):
    if mode != 'fusion':
        return batch_inference(hard_model, [prompt])[0]
    # 8b cot + 70b
    return batch_flash_cot(easy_model, hard_model, prompt)[0]

def export_results(path, data):
    fp = open(path, "w", encoding="utf8")
    json.dump(data, fp)
    fp.close()

def main():
    easy_df_list, hard_df_list = prepare_arc_data()
    running_data = process_data(easy_df_list, hard_df_list, 32)

    easy_prefix = "Qwen/"
    easy_model_name = "Qwen2.5-3B-Instruct"
    hard_prefix = "Qwen/"
    hard_model_name = "Qwen2.5-32B-Instruct"

    easy_model = easy_prefix + easy_model_name
    hard_model = hard_prefix + hard_model_name

    # results2 = run_single(running_data, model=hard_model)
    # output_path2 = f"llm_route/output/single_{hard_model_name}_results.json"
    # export_results(output_path2, results2)
    
    # results3 = run_router(running_data, route_model=easy_model, easy_model=hard_model, hard_model=hard_model)
    # output_path3 = f"llm_route/output/fuison_{easy_model_name}-{hard_model_name}_results.json"
    # export_results(output_path3, results3)

    results1 = run_single(running_data, model=easy_model)
    output_path1 = f"llm_route/output/single_{easy_model_name}_results.json"
    export_results(output_path1, results1)
    
    # results = run_llm(running_data, model='llama3.1:70b')
    # # export_results(results)
    # cal_acc(results)
    

if __name__ == '__main__':
    main()