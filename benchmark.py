import json
import os
import re

from datasets import load_dataset
from llm import batch_inference, batch_flash_cot


NUM_TEST_SAMPLES = 16

def benchmark(model_id: str, dataset_id: str, use_cot: bool = False, batch_size: int = 4, summary_model_id: str = ""):
    dataset = load_dataset("allenai/ai2_arc", dataset_id, split="test")
    results = []

    for i in range(0, min(NUM_TEST_SAMPLES, len(dataset)), batch_size):
        prompts = []
        for j in range(batch_size):
            if i+j >= len(dataset):
                break
            entry = dataset[i+j]
            system_prompt = f"You are an expert on solving complex problems."
            cot_prompt = f"Think through the problem step by step." if use_cot else ""
            prompt = f"""{system_prompt}
Here is the question:
{entry["question"]}
{'\n'.join([label + ": " + text for text, label in zip(entry["choices"]["text"], entry["choices"]["label"])])}
Wrap your answer in <ANSWER> and </ANSWER>. For example <ANSWER>X</ANSWER>.
{cot_prompt}
"""
            prompts.append(prompt)
        if not summary_model_id:
            responses = batch_inference(model_id, prompts)
        else:
            responses = batch_flash_cot(model_id, summary_model_id, prompts)
        for j in range(batch_size):
            if i+j >= len(dataset):
                break
            entry = dataset[i+j]
            response = responses[j]
            is_correct = False
            prediction = ""
            try:
                pattern = r">(.*?)<"
                matches = re.findall(pattern, response)
                for predicted in matches:
                    if predicted.upper().strip() == entry["answerKey"].upper().strip():
                        is_correct = True
                        prediction = predicted
                        break
                print(f"Testcase {i+j}: {'Correct' if is_correct else 'Incorrect'}")
            except Exception as e:
                print(f"Testcase {i+j}: {e}")
                pass
            
            results.append({
                "id": entry["id"],
                "question": entry["question"],
                "response": response,
                "predicted": prediction,
                "expected": entry["answerKey"],
                "is_correct": is_correct,
            })
        json.dump(results, open(f"{dataset_id}-{model_id.split('/')[1]}-CoT-{use_cot}{'-'+summary_model_id.split('/')[1] if summary_model_id else ''}.json", "w"), indent=2)
    print(f"Accuracy: {sum([1 for result in results if result['is_correct']]) / len(results)}")
    # json.dump(results, open(f"{dataset_id}-{model_id.split('/')[1]}-CoT-{use_cot}.json", "w"), indent=2)
    return results


def main():
    #benchmark(model_id="fsaudm/Meta-Llama-3.1-70B-Instruct-INT8", dataset_id="ARC-Challenge", use_cot=False, batch_size=4)
    #benchmark(model_id="fsaudm/Meta-Llama-3.1-70B-Instruct-INT8", dataset_id="ARC-Challenge", use_cot=True, batch_size=1)
    #benchmark(model_id="meta-llama/Llama-3.1-8B-Instruct", dataset_id="ARC-Challenge", use_cot=False, batch_size=16) # 128
    #benchmark(model_id="meta-llama/Llama-3.1-8B-Instruct", dataset_id="ARC-Challenge", use_cot=True, batch_size=16) 

    benchmark(model_id="meta-llama/Llama-3.1-8B-Instruct", summary_model_id="fsaudm/Meta-Llama-3.1-70B-Instruct-INT8", dataset_id="ARC-Challenge", use_cot=True, batch_size=4) 

if __name__ == '__main__':
    main()
