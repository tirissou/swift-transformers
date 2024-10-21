from transformers import AutoTokenizer

# Load Mistral 7B tokenizer
MODEL_ID: str = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Get EOS token ID
eos_token_id = tokenizer.eos_token_id
eos_token_string = tokenizer.decode([eos_token_id])

