import torch
from models.gpt2 import GPT2Model
from transformers import GPT2Tokenizer
from sonnet_generation import SonnetGPT, add_arguments
import argparse

# Setup args (you need to manually pass the same model args as training)
parser = argparse.ArgumentParser()
parser.add_argument("--model_size", type=str, default="gpt2")
args = parser.parse_args()

# Add model dimension arguments
args = add_arguments(args)

# Load checkpoint
checkpoint_path = "trained_model/9_10-1e-05-sonnet.pt"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Load model
model = SonnetGPT(checkpoint['args'])  # Uses saved training args
model.load_state_dict(checkpoint['model'])
model.eval()

# Move to device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Tokenizer (should be same as during training)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Prompt for generation
prompt = "The bustle in the red wood is totally different then what we used to feel"
encoding = tokenizer(prompt, return_tensors='pt').to(device)

# Generate
with torch.no_grad():
    _, generated_text = model.generate(
        encoding['input_ids'],
        temperature=1.2,
        top_p=0.9,
        max_length=150
    )

print("\nGenerated sonnet:")
print(generated_text)
