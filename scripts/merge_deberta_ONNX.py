from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from peft import AutoPeftModelForSequenceClassification  # PEFT library for loading LoRA models
import os
import argparse
import glob

# Parse command-line arguments for flexible script execution
parser = argparse.ArgumentParser(description='Merge LoRA adapter with base model')
parser.add_argument('--base', type=str, required=True, help='Base model name or path')
parser.add_argument('--adapter', type=str, required=True, help='Path to LoRA adapter folder')
parser.add_argument('--output', type=str, required=True, help='Output directory for merged model')
args = parser.parse_args()

# Store arguments in readable variable names
BASE = args.base
ADAPTER = args.adapter 
OUT = args.output

# Create output directory if it doesn't exist
os.makedirs(OUT, exist_ok=True)

# STEP 1: Detect num_labels from the trained model checkpoint
# LoRA adapters don't store num_labels, but checkpoint configs do
import glob
# Find all checkpoint directories (e.g., checkpoint-1000, checkpoint-2000)
checkpoint_dirs = glob.glob(os.path.join(ADAPTER, "checkpoint-*"))
if checkpoint_dirs:
    # Get the checkpoint with the highest step number (most recent)
    latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split("-")[-1]))
    try:
        # Look for trainer state file which contains training metadata
        checkpoint_config_path = os.path.join(latest_checkpoint, "trainer_state.json")
        if os.path.exists(checkpoint_config_path):
            import json
            with open(checkpoint_config_path) as f:
                trainer_state = json.load(f)
            print(f"Loaded trainer state from checkpoint: {latest_checkpoint}")
    except Exception as e:
        print(f"Could not load checkpoint config: {e}")

# Load base model configuration with 6 output labels (email classification task)
cfg = AutoConfig.from_pretrained(BASE, num_labels=6)
print(f"Loaded base model config. Default num_labels: {cfg.num_labels}")

# STEP 2: Load the FINE-TUNED model (base encoder + trained classifier head)
# Important: This loads the full fine-tuned checkpoint, not just the adapter
print("🔄 Loading fine-tuned model from checkpoint...")

# Find the best/latest checkpoint directory
checkpoints = glob.glob(os.path.join(ADAPTER, "checkpoint-*"))
if checkpoints:
    # Select the checkpoint with the highest iteration number
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    print(f"Using checkpoint: {latest_checkpoint}")
    model_path = latest_checkpoint
else:
    # Fallback: use the adapter directory itself if no checkpoints found
    model_path = ADAPTER
    print(f"Using adapter directory: {ADAPTER}")

# Load the fine-tuned model using PEFT's auto loader
# This automatically detects and applies the LoRA adapter weights
print("🔄 Loading model with adapter weights...")

model = AutoPeftModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=cfg.num_labels  # Ensure model has correct number of output classes
)

print("✓ Fine-tuned model loaded.")
print(f"✓ Model has {model.config.num_labels} labels")

# STEP 3: Merge LoRA adapter weights into the base model
# This combines the low-rank matrices with the original weights
# Result: a standard model without separate adapter (faster inference, easier deployment)
print("🔄 Merging LoRA adapter into base weights...")
model = model.merge_and_unload()
print("✅ Adapter merged successfully")

# STEP 4: Add human-readable label mappings to model config
# Maps numeric labels (0-5) to meaningful category names
label_mapping = {
    "id2label": {  # Convert numeric ID to text label
        "0": "forum",
        "1": "promotions",
        "2": "social_media",
        "3": "spam",
        "4": "updates",
        "5": "verify_code"
    },
    "label2id": {  # Convert text label to numeric ID
        "forum": 0,
        "promotions": 1,
        "social_media": 2,
        "spam": 3,
        "updates": 4,
        "verify_code": 5
    }
}

# Apply label mappings to model configuration
model.config.id2label = label_mapping["id2label"]
model.config.label2id = label_mapping["label2id"]

# STEP 5: Save the merged model to disk
print("\n🔄 Saving merged model...")
model.save_pretrained(OUT)  # Saves model weights and config
print("\n" + "="*60)
print(f"✅ Merged model saved to: {OUT}")
print("="*60)

# STEP 6: Save the tokenizer
print("\n🔄 Saving tokenizer...")
try:
    # First, try to load tokenizer from the adapter directory
    tok = AutoTokenizer.from_pretrained(ADAPTER, use_fast=True)
    print("✅ Loaded tokenizer from adapter")
except Exception as e:
    # Fallback: if adapter doesn't have tokenizer, use base model's tokenizer
    print(f"⚠ Could not load tokenizer from adapter: {e}")
    print("🔄 Loading tokenizer from base model instead...")
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    print("✅ Loaded tokenizer from base model")

# Save tokenizer files to output directory
tok.save_pretrained(OUT)
print("\n" + "="*60)
print(f"✅ Tokenizer saved to: {OUT}")
print("="*60)