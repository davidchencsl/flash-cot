import pandas as pd
import re
from llm import batch_flash_cot, batch_inference
from .ArcClassify import prepare_arc_data, process_data, route_base_prompt
import json


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


def run_router(data_list, route_model='Qwen/Qwen2.5-3B-Instruct', easy_model='Qwen/Qwen2.5-3B-Instruct', hard_model='Qwen/Qwen2.5-7B-Instruct', mode='fusion'):
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
        if answer is None:
            print(f"parse answer failed, id: {qid}, answer_text: {answer_response}")
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
    print(f"processed {len(data_list)} item, correct cnt: {correct_cnt}, valid cnt: {tot_cnt}, correct rate: {correct_cnt / tot_cnt}")
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
    results = run_router(running_data)
    output_path = "llm_route/output/fuison-results.json"
    export_results(output_path, results)
    
    # results = run_llm(running_data, model='llama3.1:70b')
    # # export_results(results)
    # cal_acc(results)
    

if __name__ == '__main__':
    main()