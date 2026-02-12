from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel  # PEFT = Parameter-Efficient Fine-Tuning library for LoRA
import torch
import json
import argparse
import random

print("Testing LoRA adapter (unmerged)...")

# Parse command-line arguments for flexible script execution
parser = argparse.ArgumentParser(description='Merge LoRA adapter with base model')
parser.add_argument('--base', type=str, required=True, help='Base model name or path')
parser.add_argument('--adapter', type=str, required=True, help='Path to LoRA adapter folder')
parser.add_argument('--testdatapath', type=str, required=True, help='Path to test data file')
#"data/finetuneData/test_deberta.json"
parser.add_argument('--testcount', type=int, required=True, help='Count of dataitems to test')
args = parser.parse_args()

# Store parsed arguments in variables for readability
BASE = args.base
ADAPTER = args.adapter 
TEST_DATA_PATH = args.testdatapath
TESTDATACOUNT = args.testcount

# Load the base pretrained model for sequence classification with 6 output classes
base_model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=6)
# Load the LoRA adapter on top of the base model (adapters are kept separate, not merged)
model = PeftModel.from_pretrained(base_model, ADAPTER)  
# Load the tokenizer to convert text into model inputs
tokenizer = AutoTokenizer.from_pretrained(BASE)
# Set model to evaluation mode (disables dropout, etc.)
model.eval()

# Load test dataset from JSON file
with open(TEST_DATA_PATH) as f:
    test_data = json.load(f)
    random.shuffle(test_data)  # Shuffle test data to ensure random sampling if needed

# Initialize counters for accuracy calculation
correct = 0
# Use the minimum of requested test count and actual dataset size
total = min(TESTDATACOUNT, len(test_data))

# Iterate through test samples
for i, sample in enumerate(test_data[:total]):
    # Print progress on same line (carriage return overwrites previous output)
    print(f"Testing sample {i+1}/{total}...", end="\r")
    # Tokenize input text with truncation to max 256 tokens
    inputs = tokenizer(sample["text"], return_tensors="pt", truncation=True, max_length=256)
    # Run inference without computing gradients (saves memory and speeds up)
    with torch.no_grad():
        outputs = model(**inputs)
    # Get predicted class by finding index of maximum logit value
    pred = torch.argmax(outputs.logits, dim=1).item()
    # Check if prediction matches ground truth label
    if pred == sample["label"]:
        correct += 1

# Print final accuracy percentage with 2 decimal places
print(f"\nAccuracy with unmerged adapter: {correct/total*100:.2f}%")