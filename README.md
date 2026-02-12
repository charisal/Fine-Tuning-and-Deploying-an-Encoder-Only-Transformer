# Fine-Tuning and Deploying an Encoder-Only Transformer Using ONNX Runtime

This repository contains the complete implementation referenced in the blog post:

**“Fine-Tuning and Deploying an Encoder-Only Transformer Using ONNX Runtime”**
[https://serkanaytekin.com/from-deberta-to-onnx-with-microsoft-olive-and-onnxruntime/](https://serkanaytekin.com/from-deberta-to-onnx-with-microsoft-olive-and-onnxruntime/)

It demonstrates an end-to-end workflow for:

* Fine-tuning an **encoder-only Transformer (DeBERTa-v3-base)** using LoRA
* Merging the adapter into a standalone model
* Converting the model to ONNX
* Optimizing the ONNX graph with Microsoft Olive
* Applying INT8 dynamic quantization
* Evaluating all pipeline stages consistently

The focus is on classification workloads and hardware-aware deployment using ONNX Runtime.

---

## Project Overview

The project follows a structured lifecycle:

1. **Fine-Tuning (LoRA)**
2. **Merge Adapter into Base Model**
3. **Convert to ONNX**
4. **Optimize with Olive**
5. **Quantize to INT8**
6. **Evaluate Across All Stages**

The result is a compact, optimized encoder-only classification model suitable for efficient local execution.

---

## Key Results (From Blog)

* Accuracy after fine-tuning: ~99.5%
* No accuracy degradation after ONNX conversion or optimization
* ~30% latency improvement after quantization
* ~70% model size reduction (INT8)

All evaluations were performed on the full test set (1,351 samples).

---

## Model Architecture

* Base Model: `microsoft/deberta-v3-base`
* Architecture Type: Encoder-only Transformer
* Task: Multi-class text classification
* Adaptation: LoRA (Low-Rank Adaptation)
* Runtime: ONNX Runtime
* Optimization Tooling: Microsoft Olive

---

## Important Note

This project focuses on **encoder-only classification models**.
---

