#%% Importing libraries
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import json

#%% Loading data
data = pd.read_csv("Product_Normalization_GRI.csv")

# Extract text and labels
texts = data['Room Description']
labels = data['Guest Room Info']

#%%
# Split data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Convert to dataset format
train_dataset = [{"text": txt, "label": lbl} for txt, lbl in zip(train_texts, train_labels)]
test_dataset = [{"text": txt, "label": lbl} for txt, lbl in zip(test_texts, test_labels)]

# Save to JSON
with open("train_data.json", "w") as f:
    json.dump(train_dataset, f, indent=4)

with open("test_data.json", "w") as f:
    json.dump(test_dataset, f, indent=4)

#%% Loading model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Choose your model (change based on what you're fine-tuning)
model_id = "meta-llama/Llama-3.1-8B-Instruct"  # Example: LLaMA 2
num_labels = 30  # Update based on the number of unique RoomType labels

import transformers
import torch

pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
