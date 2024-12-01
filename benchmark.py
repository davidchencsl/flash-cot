import json
import os

from datasets import load_dataset
from llm import batch_inference


NUM_TEST_SAMPLES = 1000000

def benchmark(model_id: str, dataset_id: str, use_cot: bool = False, batch_size: int = 4):
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
Wrap your answer in <ANSWER> and </ANSWER>. For example <ANSWER>A</ANSWER>.
{cot_prompt}
"""
            prompts.append(prompt)
        responses = batch_inference(model_id, prompts)
        for j in range(batch_size):
            if i+j >= len(dataset):
                break
            entry = dataset[i+j]
            response = responses[j]
            is_correct = False
            try:
                predicted = response.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
                is_correct = predicted[0] == entry["answerKey"]
                results.append(
                    {
                        "id": entry["id"],
                        "question": entry["question"],
                        "answer": entry["answerKey"],
                        "predicted": predicted,
                        "is_correct": is_correct,
                    }
                )
                print(f"Testcase {i+j}: {'Correct' if is_correct else 'Incorrect'}")
            except Exception as e:
                print(f"Error: {e}")
    print(f"Accuracy: {sum([1 for result in results if result['is_correct']]) / len(results)}")
    json.dump(results, open(f"{dataset_id}-{model_id.split('/')[1]}-CoT-{use_cot}.json", "w"), indent=2)
    return results


        


def main():
    benchmark(model_id="meta-llama/Llama-3.1-8B-Instruct", dataset_id="ARC-Challenge", use_cot=False, batch_size=64)
    benchmark(model_id="meta-llama/Llama-3.1-8B-Instruct", dataset_id="ARC-Challenge", use_cot=True, batch_size=64)
    benchmark(model_id="meta-llama/Llama-3.1-70B-Instruct", dataset_id="ARC-Challenge", use_cot=False, batch_size=16)
    benchmark(model_id="meta-llama/Llama-3.1-70B-Instruct", dataset_id="ARC-Challenge", use_cot=True, batch_size=16)

if __name__ == '__main__':
    main()
