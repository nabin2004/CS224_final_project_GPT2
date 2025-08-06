# instruction_tune_sonnetgpt.py
'''
Instruction-tune your custom SonnetGPT model for poetic Q&A.
User: "Write a sonnet about the moon."
Assistant: [Generates a sonnet]
'''
import argparse
import random
import json
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange
import os

# -----------------------------
# 1. CONFIG & SEED
# -----------------------------
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

TQDM_DISABLE = False

# -----------------------------
# 2. DATASET
# -----------------------------
class InstructionDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line.strip())
                    prompt = obj["prompt"].strip()
                    response = obj["response"].strip()
                    # Format as dialog
                    text = f"User: {prompt}\nAssistant: {response}{tokenizer.eos_token}"
                    self.examples.append(text)
                except Exception as e:
                    print(f"Error parsing line: {line[:60]}... -> {e}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {"text": self.examples[idx]}

def collate_fn(batch, tokenizer, max_length=512):
    texts = [item["text"] for item in batch]
    # Tokenize: returns a dict of lists
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None  # Return Python lists
    )

    # Extract lists
    input_ids = [torch.tensor(seq) for seq in encoded["input_ids"]]
    attention_mask = [torch.tensor(seq) for seq in encoded["attention_mask"]]

    # Pad dynamically
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.eos_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

# -----------------------------
# 3. YOUR CUSTOM SonnetGPT MODEL
# -----------------------------
class SonnetGPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Load GPT-2 backbone
        from models.gpt2 import GPT2Model  # Adjust path if needed
        self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_projection = nn.Linear(args.d, self.tokenizer.vocab_size, bias=False)

        for param in self.gpt.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        output = self.gpt(input_ids, attention_mask)
        hidden_states = output['last_hidden_state'] if isinstance(output, dict) else output
        logits = self.vocab_projection(hidden_states)
        return logits

    def get_device(self):
        for param in self.gpt.parameters():
            return param.device

    @torch.no_grad()
    def generate(self, prompt_text, temperature=0.85, top_p=0.9, max_length=200):
        self.eval()
        inputs = self.tokenizer(prompt_text, return_tensors='pt', truncation=True, max_length=400)
        token_ids = inputs['input_ids'].to(self.get_device())
        attention_mask = inputs['attention_mask'].to(self.get_device())

        for _ in range(max_length):
            logits = self.forward(token_ids, attention_mask)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            top_p_mask = cumulative_probs <= top_p
            top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()
            top_p_mask[..., 0] = True

            filtered_probs = sorted_probs * top_p_mask
            filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)

            sampled_idx = torch.multinomial(filtered_probs, 1)
            next_token = sorted_indices.gather(dim=-1, index=sampled_idx)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            token_ids = torch.cat([token_ids, next_token], dim=1)
            new_mask = torch.ones((1, 1), dtype=torch.long, device=self.get_device())
            attention_mask = torch.cat([attention_mask, new_mask], dim=1)

        full_text = self.tokenizer.decode(token_ids[0].tolist(), skip_special_tokens=False)
        reply = full_text[len(prompt_text):].strip() if "Assistant:" in prompt_text else full_text.strip()
        return reply

# -----------------------------
# 4. SAVE & LOAD
# -----------------------------
def save_model(model, optimizer, args, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }
    torch.save(save_info, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath, device):
    saved = torch.load(filepath, map_location=device, weights_only=False)
    args = saved['args']
    model = SonnetGPT(args)
    model.load_state_dict(saved['model'])
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    optimizer.load_state_dict(saved['optim'])
    return model, optimizer, args

# -----------------------------
# 5. TRAINING LOOP
# -----------------------------
def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = InstructionDataset(args.train_path, tokenizer, args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length)
    )

    # Load your pre-trained sonnet model
    print(f"Loading pre-trained model from {args.model_path}...")
    model, optimizer, model_args = load_model(args.model_path, device)
    model.train()

    # Update args if needed
    model_args.lr = args.lr
    model_args.epochs = args.epochs

    for epoch in range(args.epochs):
        epoch_loss = 0
        num_batches = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=TQDM_DISABLE)

        for batch in progress_bar:
            b_ids = batch['input_ids'].to(device)
            b_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            # Shift for causal LM
            shift_logits = rearrange(logits[:, :-1], 'b t d -> (b t) d')
            shift_labels = b_ids[:, 1:].flatten()
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=tokenizer.eos_token_id)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

        # Sample generation
        model.eval()
        with torch.no_grad():
            prompt = "User: Write a sonnet about the sea.\nAssistant:"
            print("\n--- Sample Generation ---")
            print(model.generate(prompt, temperature=args.temperature, top_p=args.top_p))
            print("--- End Sample ---\n")
        model.train()

        # Save checkpoint
        save_model(model, optimizer, model_args, f"{epoch}_{args.filepath}")

# -----------------------------
# 6. GENERATE RESPONSES
# -----------------------------
@torch.no_grad()
def generate_responses(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    saved = torch.load(f"{args.epochs-1}_{args.filepath}", map_location=device, weights_only=False)
    model = SonnetGPT(saved['args'])
    model.load_state_dict(saved['model'])
    model = model.to(device)
    model.eval()

    test_prompts = [
        "User: Generate a poem on tragedy.\nAssistant:",
        "User: Write a romantic verse about the moon.\nAssistant:",
        "User: Compose a speech about ambition and love.\nAssistant:"
    ]

    print("\n🎉 Final Model Responses:")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print(f"Response: {model.generate(prompt, temperature=args.temperature, top_p=args.top_p)}")

# -----------------------------
# 7. ARGS & MAIN
# -----------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Instruction-tune your SonnetGPT model.")
    parser.add_argument("--train_path", type=str, default="/home/nabin/Desktop/Allprojects/GPT2/data/001_first_1000.jsonl", help="Path to instruction dataset")
    parser.add_argument("--model_path", type=str, default="/home/nabin/Desktop/Allprojects/GPT2/trained_model/9_10-1e-05-sonnet.pt", help="Path to your fine-tuned sonnet .pt file")
    parser.add_argument("--filepath", type=str, default="10-1e-05-instruct-sonnet.pt", help="Output filename")
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--model_size", type=str, default="gpt2-medium", choices=['gpt2', 'gpt2-medium', 'gpt2-large'])
    return parser.parse_args()

def add_arguments(args):
    size_map = {
        'gpt2': (768, 12, 12),
        'gpt2-medium': (1024, 24, 16),
        'gpt2-large': (1280, 36, 20)
    }
    args.d, args.l, args.num_heads = size_map[args.model_size]
    return args

if __name__ == "__main__":
    args = get_args()
    args = add_arguments(args)
    seed_everything()

    # Set output filename
    args.filepath = f'{args.epochs}-{args.lr}-instruct-sonnet.pt'

    train(args)
    generate_responses(args)