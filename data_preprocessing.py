#%%
# Importing the necessary libraries
import pandas as pd
import numpy as np
import torch
import transformers
import accelerate
import datasets
import evaluate
import numpy
import pandas
import tqdm
import wandb
import sentencepiece
import google.generativeai as genai
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# %%
# Raw text data
dataset_raw = pd.read_csv('Data/Product_Normalization_GRI.csv')
dataset_raw.head();

# %%
# Description column in Raw Data
raw_text = dataset_raw['Room Description'].to_list()
#raw_text_str = "; ".join(raw_text)
#raw_text_str[:200]
raw_text_str1 = raw_text[0]
#raw_text_str1

sub_raw_text = raw_text[:2] + raw_text[10000:10001]
sub_raw_text

# %%
# Normalized Product Attribute Data
normalized_product_attributes = pd.read_excel('Data/Normalized_product_attribute_name.xlsx', sheet_name = 'Normalized Product Attributes')
normalized_product_attributes.head()

# %%
# Preparing the Normalized Product Attribute data to be used for the prompt
bed_types = normalized_product_attributes['BedType'].dropna().unique()
bed_types_list = bed_types.tolist()
bed_types_str = ", ".join(bed_types_list)
bed_types_str

# %%
room_types = normalized_product_attributes['RoomType'].dropna().unique()
room_types_list = room_types.tolist()
room_types_str = ", ".join(room_types_list)
room_types_str

# %%
rate_plan_inclusives_raw = normalized_product_attributes['Rateplan Incl/Value-adds'].dropna().unique()
rate_plan_inclusives = rate_plan_inclusives_raw.tolist()
rate_plan_inclusives_str = ", ".join(rate_plan_inclusives)
rate_plan_inclusives_str

# %%
room_amenities_raw = normalized_product_attributes['Room Amenities'].dropna().unique()
room_amenities_list = room_amenities_raw.tolist()
room_amenities_str = ", ".join(room_amenities_list)
room_amenities_str

# %%
meal_plan_raw = normalized_product_attributes['Mealplan'].dropna().unique()
meal_plan_list = meal_plan_raw.tolist()
meal_plan_str = ", ".join(meal_plan_list)
meal_plan_str

# %%
room_view_raw = normalized_product_attributes['RoomView'].dropna().unique()
room_view_list = room_view_raw.tolist()
room_view_str = ", ".join(room_view_list)
room_view_str

# %%
taxes_fees_raw = normalized_product_attributes['Taxes & Fees'].dropna().unique()
taxes_fees_list = taxes_fees_raw.tolist()
taxes_fees_str = ", ".join(taxes_fees_list)
taxes_fees_str

# %%
# Fetching the model
project_id = 'fluent-vortex-449308-f6'
model_name = 'gemini-1.5-flash-latest'

# %%
# Initialize the Vertex AI client
vertexai.init(project = project_id, location = 'us-central1')
model = GenerativeModel(model_name = model_name)

# %%
# Function to prompt the model with a single text
def process_single_text(text, model):
    # Preparing the prompt
    prompt = f"""
    Extract the following attributes from the text below:
    - Room Type
    - Bed Type
    - Rate Plan Inclusives
    - Room Amenities
    - Meal Plan
    - Room View
    - Taxes and Fees

    Format your response as a JSON object with these exact keys:
    {
        "room_type": "string",
        "bed_type": "string",
        "rate_plan_inclusives": "list of strings",
        "room_amenities": "list of strings",
        "meal_plan": "string",
        "room_view": "string",
        "taxes_and_fees": "string"
    }

    Here are two examples of correct extractions:

    Input: "IN THE ART OF NEW YORK|MADISON KING RM. 450 SQFT. MADISON AVE VIEWS LRGE MARBLE BATH SEP SHOWER. COMP WIFI"
    Output: {
        "room_type": "King Bedroom",
        "bed_type": "Unknown",
        "rate_plan_inclusives": ["Private bath or shower", "Complimetary wirless internet"],
        "room_amenities": ["Separate tub and shower","Wireless internet connection"],
        "meal_plan": "Unknown",
        "room_view": "Avenue View",
        "taxes_and_fees": "Unknown"
    }

    Input: "AAA MEMBER RATE|ECO-FICIENT QUEEN ROOM RUNWAY VIEW. COMPLEMENTARY WIFI, IN ROOM COFFEE, IRON BOARD"
    Output: {
        "room_type": "Queen Bedroom",
        "bed_type": "Unknown",
        "rate_plan_inclusives": ["Eco Friendly ", "Complimentary in-room coffee or tea", "Ironing board"],
        "room_amenities": ["Wireless internet connection", "Ironing board"],
        "meal_plan": "Unknown",
        "room_view": "Runway View",
        "taxes_and_fees": "Unknown"
    }

    Now extract from this text: {text} 

    Extract room attributes using ONLY the following normalized values:
    Room Types: {room_types_str}
    Bed Types: {bed_types_str}
    Rate Plan Inclusives: {rate_plan_inclusives_str}
    Room Amenities: {room_amenities_str}
    Meal Plans: {meal_plan_str}
    Room Views: {room_view_str}
    Taxes and Fees: {taxes_fees_str}

    If a value doesn't match any of the allowed options, select the closest match or return "Unknown".

    Rules for extraction:
    1. Each room must have exactly one room type
    2. Each room must have at least one bed type
    3. Rate Plan Inclusives should be a list
    4. Meal Plan is optional
    3. View is optional
    4. Amenities should be a list
    5. Do not infer attributes that aren't explicitly mentioned

    Special cases to handle:
    - If multiple bed types are mentioned, list all of them
    - If amenities are described but not in our standard list, choose the closest match
    - If the description is unclear or ambiguous, mark as "Unclear"
    - Ignore pricing and promotional information


    For each extracted attribute, provide a confidence score, like below example:
    {
        "room_type": {"value": "Queen Bedroom", "confidence": 0.9},
        "bed_type": {"value": "King Bed", "confidence": 0.8},
        "rate_plan_inclusives": {
            "values": ["Unknown"],
            "confidence": 0.85
        },
        "room_amenities": {
            "values": ["Wireless internet connection", "Ironing board"],
            "confidence": 0.85
        },
        "meal_plan": {"value": "Unknown", "confidence": 0.8},
        "room_view": {"value": "Ocean view", "confidence": 0.95},
        "taxes_and_fees": {"value": "Unknown", "confidence": 0.95}
    }

    Follow these steps:
    1. First, identify all mentioned attributes in the text
    2. Then, match each attribute to the closest normalized value
    3. Validate against the allowed values list
    4. Format the final output as JSON
    5. Explain your reasoning for any uncertain matches

    Show your work:
    Identified attributes: ...
    Normalized matches: ...
    Final output: ...
    Reasoning: ...

    """
    response = model.generate_content(prompt)
    return response

#%%
result1 = process_single_text(raw_text_str1, model)
result1


# %%
# Process texts with the model in a loop
results = []
for text in sub_raw_text:
    try:
        result = process_single_text(text, model)
        results.append(result)
    except Exception as e:
        print(f"Error processing text: {text[:50]}...")
        print(f"Error: {str(e)}")
        results.append(None)

#results.save('results.json')

# %%
import json
with open('results.json', 'r') as file:
    json.dump(results, file, indent=4)

# %%
