from llm import get_response
import json
import os
import ast

MAX=100000

def evaluate(model, cot=False, dataset='ARC-AGI/data/evaluation', max_testcases=MAX):
    print(f'Evaluating {model}{" with CoT" if cot else ""}')
    results = {}
    system_prompt = 'You solve puzzles by transforming the input into the output. You are expert at the PGM (Portable Graymap) format.'
    testcases = os.listdir(dataset)
    testcases.sort()
    testcases = testcases[:max_testcases]
    for i, testcase in enumerate(testcases):
        print(f'Evaluating {testcase} ({i+1}/{len(testcases)})')
        with open(f'{dataset}/{testcase}') as f:
            data = json.load(f)
        user_prompt = f"""# Transformations
## Transformation A - Input
```
{data['train'][0]["input"]}
```
## Transformation A - Output
```
{data['train'][0]["output"]}
```
## Transformation B - Input
```
{data['test'][0]["input"]}
```
## Transformation B - Output
```
# Predict this output
```
Wrap your output matrix in <Matrix> and </Matrix>.
"""
        cot_prompt = 'Please think it through step by step.' if cot else ''
        response = get_response(model, system_prompt, f"{user_prompt}{cot_prompt}")
        is_correct = False
        try:
            prediction = response.split('<Matrix>')[1].split('</Matrix>')[0].strip()
            prediction = ast.literal_eval(prediction)
            golden = data['test'][0]["output"]
            is_correct = prediction == golden
            # if not is_correct:
            #     print(f'Prediction: {prediction}')
            #     print(f'Golden: {golden}')
        except Exception as e:
            # print(f'Error: {e}')
            # print(f'Response: {response}')
            pass

        results[testcase] = {
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'response': response,
            'is_correct': is_correct
        }
        print(f'Correct: {is_correct}')

    correct_count = sum([1 for result in results.values() if result['is_correct']])
    print(f'Correct: {correct_count}/{len(results)} ({correct_count/len(results)*100:.2f}%)')
    with open(f'evaluation_{model}_CoT={cot}.json', 'w') as f:
        json.dump(results, f, indent=4)
    return results

def run():
    evaluate('llama3.1:8b')
    evaluate('llama3.1:8b', cot=True)
    evaluate('llama3.1:70b')
    evaluate('llama3.1:70b', cot=True)

if __name__ == '__main__':
    run()
