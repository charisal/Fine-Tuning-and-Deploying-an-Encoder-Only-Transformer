import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

MODEL = "models/4_optimized/model.onnx"
TOKENIZER = "models/4_optimized"
ID2LABEL = {0: "forum", 1: "promotions", 2: "social_media", 3: "spam", 4: "updates", 5: "verify_code"}

session = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)


def classify(text):
    inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=512)
    logits = session.run(None, {"input_ids": inputs["input_ids"].astype(np.int64), "attention_mask": inputs["attention_mask"].astype(np.int64)})[0][0]
    pred_id = int(np.argmax(logits))
    return ID2LABEL[pred_id]
  
print(classify("Your verification code is 123456"))
print(classify("70% off sale today only!"))