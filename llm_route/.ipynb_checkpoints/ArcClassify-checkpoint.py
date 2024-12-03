import pandas as pd
import random
from .LocalModel import call_openai, call_ollama
from llm import batch_flash_cot, batch_inference
from datasets import load_dataset

route_base_prompt = """Given a problem, you need to determine whether the problem is easy to solve or more complex. Easy to sovle will be answered by a small language model and complex problems will be answered by a large language model.
    Simple questions can usually be answered directly through common sense or a single known scientific fact, question descriptions are usually succinct and clear, as well as having little need for reasoning, and questions tend to be of the factual memory type. 
    Complex questions usually require a combination of multiple knowledge points and reasoning steps, questions may contain implicit information, require a deep understanding of the problem statement, and require background knowledge, interdisciplinary thinking, and chains of reasoning.

```
Simple questions are like:
-  question: What gas do plants use in photosynthesis?
   choices:
	A. Oxygen
	B. Carbon Dioxide
	C. Nitrogen
	D. Hydrogen

- question: Which planet is known as the "Red Planet"?
  choices:
	A. Venus
	B. Mars
	C. Jupiter
	D. Saturn

And complex questions are like:
- question: Why does a can of soda pop when opened?
  choices:
	A. Air inside the can pushes the soda out.
	B. Carbon dioxide dissolved in the soda escapes rapidly.
	C. The metal of the can expands when opened.
	D. Soda gets warmer when the can is opened.

- question: A scientist places a thermometer into a jar of water and observes the temperature. He then puts a lid on the jar and shakes it vigorously. What happens to the water temperature?
  choices:
	A. It increases.
	B. It decreases.
	C. It stays the same.
	D. It first increases and then decreases.
```

Please strictly follow the output format without explanation: if it is an easy problem, output "0"; if it is a complex problem, output "1"."""


def prepare_arc_data():
    # easy_train_df = pd.read_parquet("dataset/ai2_arc/ARC-Easy/train-00000-of-00001.parquet")
    # easy_val_df = pd.read_parquet("dataset/ai2_arc/ARC-Easy/validation-00000-of-00001.parquet")
    # easy_test_df = pd.read_parquet("dataset/ai2_arc/ARC-Easy/test-00000-of-00001.parquet")
    # # print(easy_train_df.shape)
    # # print(easy_val_df.shape)
    # # print(easy_test_df.shape)

    # hard_train_df = pd.read_parquet("dataset/ai2_arc/ARC-Challenge/train-00000-of-00001.parquet")
    # hard_val_df = pd.read_parquet("dataset/ai2_arc/ARC-Challenge/validation-00000-of-00001.parquet")
    # hard_test_df = pd.read_parquet("dataset/ai2_arc/ARC-Challenge/test-00000-of-00001.parquet")
    # # print(hard_train_df.shape)
    # # print(hard_val_df.shape)
    # # print(hard_test_df.shape)

    # return (easy_train_df, easy_val_df, easy_test_df), (hard_train_df, hard_val_df, hard_test_df)
    dataset_easy = load_dataset("allenai/ai2_arc", 'ARC-Easy', split="test")
    dataset_hard = load_dataset("allenai/ai2_arc", 'ARC-Challenge', split="test")
    return dataset_easy, dataset_hard

def process_label(df, isHard):

    label_data = []
    for item in df:
        qid = item['id']
        q = item['question']
        choice = item['choices']
        text_list = choice['text']
        letter_list = choice['label']
        c = "\n".join(f"{letter}. {text}" for (letter, text) in zip(letter_list, text_list))
        a = item['answerKey']
        label_data.append((qid, q, c, a, isHard))
        
    # ids = df['id'].tolist()
    # questions = df["question"].tolist()
    # origin_choices = df["choices"].tolist()
    # choices = []
    # for choice in origin_choices:
    #     text_list = choice['text']
    #     letter_list = choice['label']
    #     choice_text = "\n".join(f"{letter}. {text}" for (letter, text) in zip(letter_list, text_list))
    #     choices.append(choice_text)
    # answers = df["answerKey"].tolist()
    # label_data = [(qid, q, c, a, isHard) for (qid, q, c, a) in zip(ids, questions, choices, answers)]
    label_data = label_data
    return label_data

def process_data(easy_ds, hard_ds, num_of_data=32):
    easy_data = process_label(easy_ds, 0)
    hard_data = process_label(hard_ds, 1)
    # print(easy_data[:10])
    # print(hard_data[:10])
    running_data = easy_data + hard_data
    random.seed(42)
    random.shuffle(running_data)

    num_of_data = min(num_of_data, len(running_data))
    running_data = running_data[:num_of_data]
    return running_data

def run_llm(data_list, model='Qwen/Qwen2.5-32B-Instruct'):
    
    # base_prompt = """Given a problem, you need to determine whether the problem is easy to solve or more complex. Easy to sovle will be answered by a small language model and complex problems will be answered by a large language model.
    # Simple questions can usually be answered directly through common sense or a single known scientific fact, question descriptions are usually succinct and clear, as well as having little need for reasoning, and questions tend to be of the factual memory type. 
    # Complex questions usually require a combination of multiple knowledge points and reasoning steps, questions may contain implicit information, require a deep understanding of the problem statement, and require background knowledge, interdisciplinary thinking, and chains of reasoning.
    # Please strictly follow the output format without explanation: if it is an easy problem, output "0"; if it is a complex problem, output "1".""" 
    
    results = []
    for idx, (qid, q, c, a, label) in enumerate(data_list):
        item = f"Here is the question and choices: \nquestion:{q}\nchoices:\n{c}\n Please determine the output."
        prompt = route_base_prompt + item
        # response = call_openai(prompt)
        response_list = batch_inference(model, [prompt])[0]
        results.append((q, c, a, label, response))
        if (idx + 1) % 100 == 0:
            print(f"already process {idx + 1} items.")
            cal_acc(results)
    return results


def export_results(results):
    df = pd.DataFrame(results, columns=['question', 'choices', 'answer', 'hard_label', 'hard_pred'])
    df.to_csv("llm_route/output/arc_test.csv")

def cal_acc(results):
    correct = 0
    tot = 0
    err = 0
    for _, _, _, label, pred in results:
        try:
            num_pred = eval(pred)
            if num_pred == label:
                correct += 1
            tot += 1
        except Exception:
            print(label, pred)
            err += 1
    print("correct: ", correct, "tot: ", tot, "err: ", err, "acc: ", correct / tot)

def main():
    easy_ds, hard_ds = prepare_arc_data()
    running_data = process_data(easy_ds, hard_ds)
    results = run_llm(running_data)
    cal_acc(results)
    export_results(results)
    

if __name__ == '__main__':
    main()