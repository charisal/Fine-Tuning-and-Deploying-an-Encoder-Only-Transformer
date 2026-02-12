import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import argparse
import os
import numpy as np
import onnxruntime as ort
from collections import defaultdict


# define label mapping
LABEL_NAMES = ["forum", "promotions", "social_media", "spam", "updates", "verify_code"]



# Parse command-line arguments for script configuration
parser = argparse.ArgumentParser(description='Evaluate merged model performance')
parser.add_argument('--model', type=str, required=True, help='Path to model folder')
parser.add_argument('--testdatapath', type=str, required=True, help='Path to test data file')
parser.add_argument('--testcount', type=int, required=True, help='Count of dataitems to test')
args = parser.parse_args()

# Store parsed arguments in variables
MODEL_PATH = args.model
TEST_DATA_PATH = args.testdatapath
TESTDATACOUNT = args.testcount

# Detect model type based on files in the directory
def detect_model_type(model_path):
    """Detect whether model is ONNX or PyTorch based on files in directory"""
    files = os.listdir(model_path)
    has_onnx = any(f.endswith('.onnx') for f in files)
    has_pytorch = any(f in ['pytorch_model.bin', 'model.safetensors', 'config.json'] for f in files)
    
    if has_onnx and not has_pytorch:
        return 'onnx'
    elif has_pytorch and not has_onnx:
        return 'pytorch'
    elif has_onnx and has_pytorch:
        return 'both'  # Will default to PyTorch
    else:
        return 'unknown'
        



model_type = detect_model_type(MODEL_PATH)
print(f"Detected model type: {model_type}")

# Load tokenizer (works for both ONNX and PyTorch)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load model based on type
if model_type == "onnx":
    print("Loading ONNX merged model...")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    
    onnx_model_path = os.path.join(MODEL_PATH, "model.onnx")
    session = ort.InferenceSession(onnx_model_path, providers=providers, sess_options=session_options)
    
elif model_type == "pytorch":
    print("Loading PyTorch merged model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    
elif model_type == "both":
    print("Both model types found. Using ONNX...")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    
    onnx_model_path = os.path.join(MODEL_PATH, "model.onnx")
    session = ort.InferenceSession(onnx_model_path, providers=providers, sess_options=session_options)
    model_type = "onnx"  # Set to onnx for inference logic
    
else:
    raise ValueError(f"No supported model format found in {MODEL_PATH}")
str=""
# Load test dataset from JSON file
with open(TEST_DATA_PATH) as f:
    test_data = json.load(f)

# Initialize overall accuracy counters
correct = 0
total = min(TESTDATACOUNT, len(test_data))

# Initialize per-class accuracy tracking dictionaries for 6 classes
class_correct = {i: 0 for i in range(6)}  # Correct predictions per class
class_total = {i: 0 for i in range(6)}    # Total samples per class

confusion = defaultdict(lambda: defaultdict(int))

total_time = 0
for i, sample in enumerate(test_data[:total]):
    # Display progress indicator (overwrites previous line)
    print(f"Testing sample {i+1}/{total}...", end="\r")
    text = sample["text"]
    true_label = sample["label"]
    start_time = time.time()
    # Handle inference based on model type
    if model_type == 'onnx':
        # ONNX model inference
        inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=256, padding="max_length")
        input_feed = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }
        outputs = session.run(None, input_feed)
        logits = outputs[0][0]  # Get logits from ONNX output
        pred = np.argmax(logits)
        
    else:  # PyTorch model inference
        # Tokenize input text with truncation to 256 tokens max
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        # Run inference without gradient computation (saves memory)
        with torch.no_grad():
            outputs = model(**inputs)
        # Get predicted class by selecting highest logit score
        pred = torch.argmax(outputs.logits, dim=1).item()
    end_time = time.time()
    total_time += (end_time - start_time)
    # Track total samples for this class
    class_total[true_label] += 1
    # Check if prediction matches ground truth
    if pred == true_label:
        correct += 1
        class_correct[true_label] += 1  # Track correct predictions per class
    if pred != true_label:
        confusion[true_label][pred] += 1

# Print overall accuracy across all classes
avg_inference_time = (total_time / total) * 1000  # ms
print(f"\nEvaluation Results ({model_type.upper()} model):")
print(f"\nOverall Accuracy: {correct/total*100:.2f}%")
print(f"Average inference time: {avg_inference_time:.2f} ms")
print(f"Total inference time: {total_time:.2f} s")

# Print detailed per-class accuracy breakdown
print("\nPer-class accuracy:")
for i in range(6):
    # Only show classes that have samples in the test set
    if class_total[i] > 0:
        acc = class_correct[i] / class_total[i] * 100
        # Display: class number, accuracy percentage, and fraction (correct/total)
        print(f"  Class {i} ({LABEL_NAMES[i]}): {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
print("\nMost common misclassifications:")
if confusion:
    for true_label, preds in confusion.items():
        for pred_label, count in sorted(preds.items(), key=lambda x: -x[1])[:3]:
            print(f"  {LABEL_NAMES[true_label]} → {LABEL_NAMES[pred_label]}: {count} times")
else:
    print("  None - perfect classification!")