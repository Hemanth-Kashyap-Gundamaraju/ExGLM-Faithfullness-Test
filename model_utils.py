import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_quantized_model(model_name="Qwen/Qwen2.5-1.5B-Instruct"):
    """Loads a 4-bit quantized open-weight model for single-GPU inference."""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )
    return model, tokenizer

def construct_prompt(node_id, text_features, neighborhood_ids):
    """Constructs the ExGLM-style prompt for the LLM."""
    neighborhood_str = ", ".join([f"Node {n}" for n in neighborhood_ids])
    
    prompt = f"""You are a Graph-Language Model classifying research papers.
Target Node: Node {node_id}
Target Node Features: {text_features}

Graph Context (Neighbors): {neighborhood_str}

Task: Predict the category of the Target Node and provide a natural language explanation explicitly citing the neighbors that influenced your decision.
Format:
Label: [Category]
Explanation: [Your reasoning citing specific nodes, e.g., 'Node X']"""
    return prompt

def run_inference(prompt, model, tokenizer):
    """Runs the prompt through the model and extracts the response."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.0)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Placeholder for actual logprob extraction of the predicted class token
    simulated_prob = np.random.uniform(0.7, 0.99) 
    
    return response, simulated_prob