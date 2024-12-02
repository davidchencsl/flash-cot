import pandas as pd
import re
from LocalModel import call_openai, call_ollama
from ArcClassify import prepare_arc_data, process_data, route_base_prompt
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


def run_router(data_list, route_model='llama3.1:70b', easy_model='qwen2.5:7b', hard_model='fusion'):
    results = []

    tot_cnt = 0
    correct_cnt = 0
    for idx, (qid, q, c, a, label) in enumerate(data_list):
        route_item = f"Here is the question and choices: \nquestion:{q}\nchoices:\n{c}\n Please determine the output."
        route_prompt = route_base_prompt + route_item
        route_sign = call_ollama(route_model, route_prompt)

        item = f"Here is the question and choices: \nquestion:{q}\nchoices:\n{c}\n Please determine the answer."
        question_prompt = question_base_prompt + item
        answer_response = None
        if '0' in route_sign:
            answer_response = call_ollama(easy_model, question_prompt)
        else:
            answer_response = call_hard_model(hard_model, question_prompt)

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


def call_hard_model(model, prompt):
    if model != 'fusion':
        return call_ollama(model, prompt)
    
    # 8b cot + 70b
    return call_ollama(model, prompt)

def export_results(path, data):
    fp = open(path, "w", encoding="utf8")
    json.dump(data, fp)
    fp.close()

def main():
    easy_df_list, hard_df_list = prepare_arc_data()
    running_data = process_data(easy_df_list, hard_df_list)
    results = run_router(running_data)
    export_results(results)
    
    # results = run_llm(running_data, model='llama3.1:70b')
    # # export_results(results)
    # cal_acc(results)
    

if __name__ == '__main__':
    main()