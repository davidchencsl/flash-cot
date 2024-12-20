from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_cache = {}

def batch_inference(model_id: str, prompts: list):
    """
    Perform batch inference with GPU acceleration, ensuring the prompt tokens are skipped.
    
    Args:
        model_id (str): Identifier for the model (Hugging Face model hub name).
        prompts (list): List of prompts (strings) for inference.
        
    Returns:
        list: Responses generated by the model for each prompt.
    """

    # Load the tokenizer and model
    if model_id not in model_cache:
        model_cache[model_id] = {
            "tokenizer": AutoTokenizer.from_pretrained(model_id, device_map="auto"),
            "model": AutoModelForCausalLM.from_pretrained(model_id, device_map="auto"),
        }
    tokenizer = model_cache[model_id]["tokenizer"]
    model = model_cache[model_id]["model"]

    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Tokenize prompts for batch processing
    inputs = tokenizer(prompts, padding="longest", return_tensors="pt")
    inputs = {key: val.to("cuda") for key, val in inputs.items()}  # Move inputs to GPU

    # Generate responses
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            pad_token_id=tokenizer.eos_token_id
        )

        responses = []
        # Decode responses and remove prompt text
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i, raw_response in enumerate(decoded):
            # Remove the prompt text from the start of the response
            response = raw_response[len(prompts[i]):].strip()
            responses.append(response)

        return responses
    except Exception as e:
        print(f"Error during inference: {e}")
        return []
    

def batch_flash_cot(draft_model_id, summary_model_id, prompts: list):
    cot_responses = batch_inference(draft_model_id, prompts)
    full_prompts = [a+"\n"+b+'\nUse the information above to answer the question. Wrap your answer in <ANSWER> and </ANSWER>. For example <ANSWER>X</ANSWER>.' for a, b in zip(prompts, cot_responses)]
    responses = batch_inference(summary_model_id, full_prompts)
    return responses
    