"""
Comprehensive comparison of ALL models:
- Base DeBERTa-v3 (no fine-tuning)
- Fine-tuned Unquantized ONNX
- Fine-tuned Unquantized ONNX with optimizations
- Fine-tuned Quantized ONNX

Tests on first 1000 samples from test dataset
"""

import json
import time
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnxruntime as ort
import argparse
import random

parser = argparse.ArgumentParser(description='Compare available models (base, fine-tuned, optimized, and quantized)')
parser.add_argument('--base', type=str, help='Base model name or path (optional)')
parser.add_argument('--finetuned', type=str, help='Path to fine-tuned ONNX model directory (optional)')
parser.add_argument('--optimized', type=str, help='Path to optimized ONNX model directory (optional)')
parser.add_argument('--quantized', type=str, help='Path to quantized ONNX model directory (optional)')
parser.add_argument('--testdatapath', type=str, required=True, help='Path to test data file')

parser.add_argument('--testcount', type=int, required=True, help='Count of dataitems to test')
args = parser.parse_args()

# Check that at least one model is provided
if not any([args.base, args.finetuned, args.optimized, args.quantized]):
    parser.error("At least one model must be specified (--base, --finetuned, --optimized, or --quantized)")

# Store arguments in readable variable names
BASE = args.base
FINETUNED = args.finetuned 
OPTIMIZED = args.optimized
QUANTIZED = args.quantized
TEST_DATA_PATH = args.testdatapath
TEST_COUNT = args.testcount

# Track which models are available
available_models = []
models = {}

# ONNX providers configuration
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

print("=" * 90)
print("LOADING AVAILABLE MODELS")
print("=" * 90)

# 1. Load BASE model (no fine-tuning) if provided
if BASE:
    print("\n[1/4] Loading BASE microsoft/deberta-v3-base model (NO fine-tuning)...")
    try:
        base_tokenizer = AutoTokenizer.from_pretrained(BASE)
        base_model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=6)
        base_model.eval()
        models['base'] = {'tokenizer': base_tokenizer, 'model': base_model, 'type': 'pytorch'}
        available_models.append('base')
        print("✓ Base model loaded")
    except Exception as e:
        print(f"✗ Failed to load base model: {e}")
else:
    print("\n[1/4] Skipping BASE model (not provided)")

# 2. Load Fine-tuned ONNX model if provided
if FINETUNED:
    print("\n[2/4] Loading Fine-tuned UNQUANTIZED ONNX model...")
    try:
        finetuned_tokenizer = AutoTokenizer.from_pretrained(BASE if BASE else "microsoft/deberta-v3-base")
        finetuned_model_path = os.path.join(FINETUNED, "model.onnx")
        if not os.path.exists(finetuned_model_path):
            raise FileNotFoundError(f"ONNX model not found at {finetuned_model_path}")
        finetuned_session = ort.InferenceSession(finetuned_model_path, providers=providers)
        models['finetuned'] = {'tokenizer': finetuned_tokenizer, 'session': finetuned_session, 'type': 'onnx'}
        available_models.append('finetuned')
        print(f"✓ Fine-tuned UNQUANTIZED ONNX model loaded (Provider: {finetuned_session.get_providers()[0]})")
    except Exception as e:
        print(f"✗ Failed to load fine-tuned model: {e}")
else:
    print("\n[2/4] Skipping Fine-tuned ONNX model (not provided)")

# 3. Load OPTIMIZED ONNX model if provided
if OPTIMIZED:
    print("\n[3/4] Loading Fine-tuned OPTIMIZED ONNX model...")
    try:
        optimized_tokenizer = AutoTokenizer.from_pretrained(OPTIMIZED if os.path.exists(os.path.join(OPTIMIZED, "tokenizer.json")) else (BASE if BASE else "microsoft/deberta-v3-base"))
        optimized_model_path = os.path.join(OPTIMIZED, "model.onnx")
        if not os.path.exists(optimized_model_path):
            raise FileNotFoundError(f"ONNX model not found at {optimized_model_path}")
        optimized_session = ort.InferenceSession(optimized_model_path, providers=providers)
        models['optimized'] = {'tokenizer': optimized_tokenizer, 'session': optimized_session, 'type': 'onnx'}
        available_models.append('optimized')
        print(f"✓ Optimized ONNX model loaded (Provider: {optimized_session.get_providers()[0]})")
    except Exception as e:
        print(f"✗ Failed to load optimized model: {e}")
else:
    print("\n[3/4] Skipping Optimized ONNX model (not provided)")

# 4. Load QUANTIZED ONNX model if provided
if QUANTIZED:
    print("\n[4/4] Loading Fine-tuned QUANTIZED ONNX model...")
    try:
        quant_tokenizer = AutoTokenizer.from_pretrained(QUANTIZED if os.path.exists(os.path.join(QUANTIZED, "tokenizer.json")) else (BASE if BASE else "microsoft/deberta-v3-base"))
        quant_model_path = os.path.join(QUANTIZED, "model.onnx")
        if not os.path.exists(quant_model_path):
            raise FileNotFoundError(f"ONNX model not found at {quant_model_path}")
        quant_session = ort.InferenceSession(quant_model_path, providers=providers)
        models['quantized'] = {'tokenizer': quant_tokenizer, 'session': quant_session, 'type': 'onnx'}
        available_models.append('quantized')
        print(f"✓ Quantized ONNX model loaded (Provider: {quant_session.get_providers()[0]})")
    except Exception as e:
        print(f"✗ Failed to load quantized model: {e}")
else:
    print("\n[4/4] Skipping Quantized ONNX model (not provided)")

if not available_models:
    print("\n✗ No models were successfully loaded. Exiting.")
    exit(1)

print(f"\n✓ Successfully loaded {len(available_models)} model(s): {', '.join(available_models)}")

# Load test dataset
print("\n" + "=" * 90)
print("LOADING TEST DATASET")
print("=" * 90)
try:
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)
except FileNotFoundError:
    print(f"✗ Test data file not found: {TEST_DATA_PATH}")
    exit(1)
except json.JSONDecodeError:
    print(f"✗ Invalid JSON in test data file: {TEST_DATA_PATH}")
    exit(1)

NUM_SAMPLES = min(TEST_COUNT, len(test_data))
test_subset = test_data[:NUM_SAMPLES]
print(f"Loaded {len(test_data)} total samples, using first {NUM_SAMPLES} for comparison\n")

# Initialize statistics for available models only
stats = {}
for model_name in available_models:
    stats[model_name] = {"correct": 0, "times": [], "class_stats": {i: {"correct": 0, "total": 0} for i in range(6)}}

print("=" * 90)
print(f"TESTING {len(available_models)} MODEL(S) ON {NUM_SAMPLES} SAMPLES")
print("=" * 90)

with torch.no_grad():
    for i, sample in enumerate(test_subset, 1):
        if i % 200 == 0:
            print(f"Progress: {i}/{NUM_SAMPLES} samples processed...")
        
        try:
            text = sample["text"]
            true_label = sample["label"]
        except KeyError as e:
            print(f"⚠ Skipping sample {i}: Missing key {e}")
            continue
        
        # Update class totals for all available models
        for model_name in available_models:
            stats[model_name]["class_stats"][true_label]["total"] += 1
        
        # Test each available model
        for model_name in available_models:
            model_info = models[model_name]
            
            if model_info['type'] == 'pytorch':
                # PyTorch model (base)
                inputs = model_info['tokenizer'](text, return_tensors="pt", truncation=True, max_length=256, padding="max_length")
                start = time.time()
                outputs = model_info['model'](**inputs)
                inference_time = (time.time() - start) * 1000
                pred = torch.argmax(outputs.logits, dim=1).item()
                
            elif model_info['type'] == 'onnx':
                # ONNX model (finetuned, optimized, quantized)
                inputs = model_info['tokenizer'](text, return_tensors="np", truncation=True, max_length=256, padding="max_length")
                input_feed = {
                    "input_ids": inputs["input_ids"].astype(np.int64),
                    "attention_mask": inputs["attention_mask"].astype(np.int64)
                }
                start = time.time()
                outputs = model_info['session'].run(None, input_feed)
                inference_time = (time.time() - start) * 1000
                logits = outputs[0][0]
                pred = np.argmax(logits)
            
            # Record results
            stats[model_name]["times"].append(inference_time)
            if pred == true_label:
                stats[model_name]["correct"] += 1
                stats[model_name]["class_stats"][true_label]["correct"] += 1

print(f"✓ Testing complete!\n")

# Calculate metrics for available models
def calc_metrics(stat_dict, num_samples):
    acc = (stat_dict["correct"] / num_samples) * 100
    mean_time = np.mean(stat_dict["times"])
    median_time = np.median(stat_dict["times"])
    return acc, mean_time, median_time

# Calculate metrics for all available models
model_metrics = {}
for model_name in available_models:
    acc, mean_time, median_time = calc_metrics(stats[model_name], NUM_SAMPLES)
    model_metrics[model_name] = {
        'accuracy': acc,
        'mean_time': mean_time,
        'median_time': median_time,
        'correct': stats[model_name]["correct"]
    }

def get_model_size(model_dir, model_type='onnx'):
    """Get size of model file"""
    if model_type == 'pytorch':
        return "~700 MB (estimated)"  # PyTorch model size estimate
    elif model_type == 'onnx':
        model_path = os.path.join(model_dir, "model.onnx")
        if not os.path.exists(model_path):
            return 0.0
        return os.path.getsize(model_path) / (1024**2)
    return "Unknown"

# Get model sizes
model_sizes = {}
if 'base' in available_models:
    model_sizes['base'] = get_model_size(BASE, 'pytorch')
if 'finetuned' in available_models:
    model_sizes['finetuned'] = get_model_size(FINETUNED, 'onnx')
if 'optimized' in available_models:
    model_sizes['optimized'] = get_model_size(OPTIMIZED, 'onnx')
if 'quantized' in available_models:
    model_sizes['quantized'] = get_model_size(QUANTIZED, 'onnx')



# ===== DISPLAY RESULTS =====
print("=" * 90)
print(f"MODEL COMPARISON ({NUM_SAMPLES} SAMPLES) - {len(available_models)} MODEL(S) TESTED")
print("=" * 90)
print()

# Define model display names
model_display_names = {
    'base': 'Base (No FT)',
    'finetuned': 'Fine-tuned ONNX',
    'optimized': 'Optimized ONNX', 
    'quantized': 'Quantized ONNX'
}

# Overall Comparison Table
print("OVERALL PERFORMANCE")
print("-" * 90)

# Dynamic column headers
header = f"{'Metric':<30}"
for model_name in available_models:
    header += f" {model_display_names[model_name]:<20}"
print(header)
print("-" * 90)

# Accuracy row
accuracy_row = f"{'Accuracy':<30}"
for model_name in available_models:
    accuracy_row += f" {model_metrics[model_name]['accuracy']:>6.2f}%{'':<12}"
print(accuracy_row)

# Correct predictions row
correct_row = f"{'Correct Predictions':<30}"
for model_name in available_models:
    correct_row += f" {model_metrics[model_name]['correct']:>4}/{NUM_SAMPLES:<14}"
print(correct_row)

# Mean inference time row
mean_time_row = f"{'Mean Inference Time':<30}"
for model_name in available_models:
    mean_time_row += f" {model_metrics[model_name]['mean_time']:>7.2f} ms{'':<11}"
print(mean_time_row)

# Median inference time row
median_time_row = f"{'Median Inference Time':<30}"
for model_name in available_models:
    median_time_row += f" {model_metrics[model_name]['median_time']:>7.2f} ms{'':<11}"
print(median_time_row)

# Model size row
size_row = f"{'Model Size':<30}"
for model_name in available_models:
    size = model_sizes.get(model_name, "Unknown")
    if isinstance(size, float):
        size_row += f" {size:>7.2f} MB{'':<11}"
    else:
        size_row += f" {size:<20}"
print(size_row)

print("-" * 90)
print()

# Per-Class Accuracy Table
print("PER-CLASS ACCURACY BREAKDOWN")
print("-" * 90)

class_header = f"{'Class':<10} {'Samples':<12}"
for model_name in available_models:
    class_header += f" {model_display_names[model_name]:<20}"
print(class_header)
print("-" * 90)

for class_id in range(6):
    # Get number of samples for this class (should be same for all models)
    n_samples = stats[available_models[0]]["class_stats"][class_id]["total"] if available_models else 0
    
    class_row = f"{class_id:<10} {n_samples:<12}"
    for model_name in available_models:
        class_acc = (stats[model_name]["class_stats"][class_id]["correct"] / n_samples * 100) if n_samples > 0 else 0
        class_row += f" {class_acc:>6.2f}%{'':<12}"
    print(class_row)

print("-" * 90)
print()

# Impact Analysis - only show when models are available for comparison
if len(available_models) > 1:
    print("IMPACT ANALYSIS")
    print("-" * 90)
    
    # Base to fine-tuned comparison
    if 'base' in available_models and 'finetuned' in available_models:
        acc_diff = model_metrics['finetuned']['accuracy'] - model_metrics['base']['accuracy']
        time_diff = model_metrics['finetuned']['mean_time'] - model_metrics['base']['mean_time']
        print(f"Base → Fine-tuned:        Accuracy: {model_metrics['base']['accuracy']:.2f}% → {model_metrics['finetuned']['accuracy']:.2f}% ({acc_diff:+.2f}%)")
        print(f"                          Speed: {model_metrics['base']['mean_time']:.2f}ms → {model_metrics['finetuned']['mean_time']:.2f}ms ({time_diff:+.2f}ms)")
        print()
    
    # Fine-tuned to optimized comparison
    if 'finetuned' in available_models and 'optimized' in available_models:
        acc_diff = model_metrics['optimized']['accuracy'] - model_metrics['finetuned']['accuracy']
        time_diff = model_metrics['optimized']['mean_time'] - model_metrics['finetuned']['mean_time']
        if isinstance(model_sizes['finetuned'], (int, float)) and isinstance(model_sizes['optimized'], (int, float)):
            size_change = ((model_sizes['optimized'] / model_sizes['finetuned']) - 1) * 100
            size_str = f"Size: {model_sizes['finetuned']:.2f}MB → {model_sizes['optimized']:.2f}MB ({size_change:+.2f}%)"
        else:
            size_str = ""
        print(f"Fine-tuned → Optimized:   Accuracy: {model_metrics['finetuned']['accuracy']:.2f}% → {model_metrics['optimized']['accuracy']:.2f}% ({acc_diff:+.2f}%)")
        print(f"                          Speed: {model_metrics['finetuned']['mean_time']:.2f}ms → {model_metrics['optimized']['mean_time']:.2f}ms ({time_diff:+.2f}ms)")
        if size_str:
            print(f"                          {size_str}")
        print()
    
    # Optimized to quantized comparison
    if 'optimized' in available_models and 'quantized' in available_models:
        acc_diff = model_metrics['quantized']['accuracy'] - model_metrics['optimized']['accuracy']
        time_diff = model_metrics['quantized']['mean_time'] - model_metrics['optimized']['mean_time']
        if isinstance(model_sizes['optimized'], (int, float)) and isinstance(model_sizes['quantized'], (int, float)):
            size_change = ((model_sizes['quantized'] / model_sizes['optimized']) - 1) * 100
            size_str = f"Size: {model_sizes['optimized']:.2f}MB → {model_sizes['quantized']:.2f}MB ({size_change:+.2f}%)"
        else:
            size_str = ""
        print(f"Optimized → Quantized:    Accuracy: {model_metrics['optimized']['accuracy']:.2f}% → {model_metrics['quantized']['accuracy']:.2f}% ({acc_diff:+.2f}%)")
        print(f"                          Speed: {model_metrics['optimized']['mean_time']:.2f}ms → {model_metrics['quantized']['mean_time']:.2f}ms ({time_diff:+.2f}ms)")
        if size_str:
            print(f"                          {size_str}")
        print()
    
    # Fine-tuned to quantized direct comparison (if optimized not available)
    if 'finetuned' in available_models and 'quantized' in available_models and 'optimized' not in available_models:
        acc_diff = model_metrics['quantized']['accuracy'] - model_metrics['finetuned']['accuracy']
        time_diff = model_metrics['quantized']['mean_time'] - model_metrics['finetuned']['mean_time']
        if isinstance(model_sizes['finetuned'], (int, float)) and isinstance(model_sizes['quantized'], (int, float)):
            size_change = ((model_sizes['quantized'] / model_sizes['finetuned']) - 1) * 100
            size_str = f"Size: {model_sizes['finetuned']:.2f}MB → {model_sizes['quantized']:.2f}MB ({size_change:+.2f}%)"
        else:
            size_str = ""
        print(f"Fine-tuned → Quantized:   Accuracy: {model_metrics['finetuned']['accuracy']:.2f}% → {model_metrics['quantized']['accuracy']:.2f}% ({acc_diff:+.2f}%)")
        print(f"                          Speed: {model_metrics['finetuned']['mean_time']:.2f}ms → {model_metrics['quantized']['mean_time']:.2f}ms ({time_diff:+.2f}ms)")
        if size_str:
            print(f"                          {size_str}")
        print()
    
    print("-" * 90)
    print()

# Recommendations
print("DEPLOYMENT RECOMMENDATION")
print("-" * 90)

if len(available_models) == 1:
    model_name = available_models[0]
    print(f"Only one model tested: {model_display_names[model_name]}")
    print(f"Accuracy: {model_metrics[model_name]['accuracy']:.2f}%")
    print(f"Mean inference time: {model_metrics[model_name]['mean_time']:.2f}ms")
    print("RECOMMENDED: Use this model (no alternatives tested)")
else:
    # Find the best model based on different criteria
    best_accuracy_model = max(available_models, key=lambda x: model_metrics[x]['accuracy'])
    best_speed_model = min(available_models, key=lambda x: model_metrics[x]['mean_time'])
    
    # Smart recommendation logic
    if 'quantized' in available_models and 'finetuned' in available_models:
        speedup = ((model_metrics['finetuned']['mean_time'] - model_metrics['quantized']['mean_time']) / model_metrics['finetuned']['mean_time']) * 100
        acc_loss = model_metrics['finetuned']['accuracy'] - model_metrics['quantized']['accuracy']
        
        if isinstance(model_sizes.get('finetuned'), (int, float)) and isinstance(model_sizes.get('quantized'), (int, float)):
            size_reduction = ((model_sizes['finetuned'] - model_sizes['quantized']) / model_sizes['finetuned']) * 100
        else:
            size_reduction = 0
        
        if acc_loss > 5:  # More than 5% accuracy loss
            print(f"⚠ Quantization degradation too high: -{acc_loss:.2f}%")
            print("RECOMMENDED: Fine-tuned Unquantized ONNX")
        elif speedup > 20 or size_reduction > 50:  # Significant gains
            print(f"✓ Quantization worth it: {speedup:.1f}% faster, {size_reduction:.1f}% smaller")
            print("RECOMMENDED: Quantized ONNX")
        else:
            print("RECOMMENDED: Fine-tuned Unquantized ONNX (marginal quantization benefit)")
    else:
        # General recommendation based on available models
        if best_accuracy_model == best_speed_model:
            print(f"RECOMMENDED: {model_display_names[best_accuracy_model]} (best accuracy AND speed)")
        else:
            print(f"Best Accuracy: {model_display_names[best_accuracy_model]} ({model_metrics[best_accuracy_model]['accuracy']:.2f}%)")
            print(f"Best Speed: {model_display_names[best_speed_model]} ({model_metrics[best_speed_model]['mean_time']:.2f}ms)")
            print("RECOMMENDED: Choose based on your priority (accuracy vs speed)")

print("-" * 90)
