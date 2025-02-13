#%%
# Import required libraries
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
import re
import spacy
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

#%%
# Prep data for attribute extraction
dataset_normalized_attributes = pd.read_excel('Data/Normalized_product_attribute_name.xlsx', sheet_name = 'Normalized Product Attributes')
dataset_normalized_attributes

room_types = dataset_normalized_attributes['RoomType'].dropna().unique()
room_types_list = room_types.tolist()

bed_types = dataset_normalized_attributes['BedType'].dropna().unique()
bed_types_list = bed_types.tolist()

rate_plan_inclusives_raw = dataset_normalized_attributes['Rateplan Incl/Value-adds'].dropna().unique()
rate_plan_inclusives = rate_plan_inclusives_raw.tolist()

room_amenities_raw = dataset_normalized_attributes['Room Amenities'].dropna().unique()
room_amenities_list = room_amenities_raw.tolist()

meal_plan_raw = dataset_normalized_attributes['Mealplan'].dropna().unique()
meal_plan_list = meal_plan_raw.tolist()

room_view_raw = dataset_normalized_attributes['RoomView'].dropna().unique()
room_view_list = room_view_raw.tolist()

taxes_fees_raw = dataset_normalized_attributes['Taxes & Fees'].dropna().unique()
taxes_fees_list = taxes_fees_raw.tolist()


#%%
def preprocess_pattern_list(pattern_list):
    """
    Convert camelCase/PascalCase patterns to lowercase and split into individual words
    Example: 'KingBed' -> ['king', 'bed']
    """
    processed = []
    for pattern in pattern_list:
        # Handle NaN values and empty strings
        if pd.isna(pattern) or str(pattern).strip() == '':
            continue
            
        # Convert to string and strip whitespace
        pattern = str(pattern).strip()
        
        # Split camelCase/PascalCase
        words = re.findall('[A-Z][^A-Z]*', pattern)
        
        # Only add non-empty patterns
        if words:  # If words were found by the regex
            processed_pattern = ' '.join(word.lower() for word in words).strip()
            if processed_pattern:  # If the result isn't empty
                processed.append(processed_pattern)
        else:  # If no camelCase/PascalCase found, just lowercase the whole thing
            if pattern:  # If pattern isn't empty
                processed.append(pattern.lower().strip())
    
    # Remove duplicates and empty strings
    return [p for p in list(set(processed)) if p.strip()]

#%% Extract attributes using spaCy
def analyze_with_spacy(texts):
    # Load transformer-based English model
    try:
        nlp = spacy.load("en_core_web_trf")
    except OSError:
        print("Downloading en_core_web_trf model...")
        spacy.cli.download("en_core_web_trf")
        nlp = spacy.load("en_core_web_trf")
    
    # Preprocess the pattern lists
    processed_room_types = preprocess_pattern_list(room_types_list)
    processed_bed_types = preprocess_pattern_list(bed_types_list)
    processed_room_views = preprocess_pattern_list(room_view_list)
    processed_amenities = preprocess_pattern_list(room_amenities_list)
    
    # Custom labels for hospitality domain
    room_patterns = [
        {"label": "ROOM_TYPE", "pattern": [{"LOWER": {"IN": processed_room_types}}]},
        {"label": "BED_TYPE", "pattern": [{"LOWER": {"IN": processed_bed_types}}]},
        {"label": "VIEW_TYPE", "pattern": [
            {"LOWER": {"IN": processed_room_views}},
            {"LOWER": "view", "OP": "?"}
        ]},
        {"label": "AMENITY", "pattern": [{"LOWER": {"IN": processed_amenities}}]}
    ]
    
    # Add patterns to pipeline
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
    else:
        ruler = nlp.get_pipe("entity_ruler")
    ruler.add_patterns(room_patterns)
    
    results = defaultdict(list)
    
    # Process each text with batch processing for better performance
    batch_size = 32  # Adjusting a random number based on potential available RAM
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing with spaCy"):
        batch_texts = texts[i:i + batch_size]
        batch_texts = [text if pd.notna(text) else "" for text in batch_texts]
        
        # Process batch
        docs = nlp.pipe(batch_texts)
        
        for doc in docs:
            entities = {
                "text": doc.text,
                "room_types": [],
                "bed_types": [],
                "view_types": [],
                "amenities": [],
                "other_entities": []
            }
            
            for ent in doc.ents:
                if ent.label_ == "ROOM_TYPE":
                    entities["room_types"].append(ent.text.lower())
                elif ent.label_ == "BED_TYPE":
                    entities["bed_types"].append(ent.text.lower())
                elif ent.label_ == "VIEW_TYPE":
                    entities["view_types"].append(ent.text.lower())
                elif ent.label_ == "AMENITY":
                    entities["amenities"].append(ent.text.lower())
                else:
                    entities["other_entities"].append((ent.text.lower(), ent.label_))
            
            for key, value in entities.items():
                results[key].append(value)
    
    return pd.DataFrame(results)


#%%
# Analyze the results
def analyze_entity_frequencies(df):
    print("\nMost common room types:")
    room_types = [item for sublist in df['room_types'] for item in sublist]
    print(pd.Series(room_types).value_counts().head(10))
    
    print("\nMost common bed types:")
    bed_types = [item for sublist in df['bed_types'] for item in sublist]
    print(pd.Series(bed_types).value_counts().head(10))
    
    print("\nMost common view types:")
    view_types = [item for sublist in df['view_types'] for item in sublist]
    print(pd.Series(view_types).value_counts().head(10))
    
    print("\nMost common amenities:")
    amenities = [item for sublist in df['amenities'] for item in sublist]
    print(pd.Series(amenities).value_counts().head(10))
    
    print("\nSample of other entities found:")
    other_entities = [item for sublist in df['other_entities'] for item in sublist]
    print(pd.Series([e[0] for e in other_entities]).value_counts().head(10))


#%%
# Print the processed patterns for verification
print("Processed Room Types:", preprocess_pattern_list(room_types_list))
print("Processed Bed Types:", preprocess_pattern_list(bed_types_list))
print("Processed View Types:", preprocess_pattern_list(room_view_list))
print("Processed Amenities:", preprocess_pattern_list(room_amenities_list))

#%%
# Run analysis on a sample first to verify results
sample_size = 100
sample_texts = dataset_raw['Room Description'].sample(n=sample_size, random_state=42)

print("\nRunning spaCy analysis with transformer model...")
spacy_results = analyze_with_spacy(sample_texts)

#%%
analyze_entity_frequencies(spacy_results)

#%%
# If results look good, process the full dataset
print("\nProcessing full dataset...")
full_results = analyze_with_spacy(dataset_raw['Room Description'])
analyze_entity_frequencies(full_results)

#%%
# Add extracted features back to original dataframe
dataset_new = dataset_raw.copy()
dataset_new['extracted_room_types'] = full_results['room_types']
dataset_new['extracted_bed_types'] = full_results['bed_types']
dataset_new['extracted_view_types'] = full_results['view_types']
dataset_new['extracted_amenities'] = full_results['amenities']

dataset_new.to_excel('Data/dataset_spacy_extraction_v1.xlsx', index=False)



#%% Extract attributes using BERT embeddings and semantic similarity
def extract_with_bert(texts, attribute_lists):
    """
    Extract attributes using BERT embeddings and semantic similarity
    """
    # Load the model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Prepare attribute lists
    attributes = {
        'room_types': preprocess_pattern_list(room_types_list),
        'bed_types': preprocess_pattern_list(bed_types_list),
        'views': preprocess_pattern_list(room_view_list),
        'amenities': preprocess_pattern_list(room_amenities_list)
    }
    
    # Create embeddings for all attribute values
    attribute_embeddings = {}
    for attr_type, attr_list in attributes.items():
        attribute_embeddings[attr_type] = model.encode(attr_list, convert_to_tensor=True)
    
    results = defaultdict(list)
    
    # Process texts in batches
    batch_size = 32
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing with BERT"):
        batch_texts = texts[i:i + batch_size]
        batch_texts = [text if pd.notna(text) else "" for text in batch_texts]
        
        # Encode all texts in batch
        text_embeddings = model.encode(batch_texts, convert_to_tensor=True)
        
        # For each text in the batch
        for idx, text_embedding in enumerate(text_embeddings):
            entities = {
                "text": batch_texts[idx],
                "room_types": [],
                "bed_types": [],
                "views": [],
                "amenities": []
            }
            
            # For each attribute type
            for attr_type, attr_embeddings in attribute_embeddings.items():
                # Calculate similarities
                similarities = util.cos_sim(text_embedding, attr_embeddings)
                
                # Get matches above threshold
                threshold = 0.5  # Adjust this threshold as needed
                matches = torch.where(similarities > threshold)[0]
                
                # Add matched attributes
                for match in matches:
                    entities[attr_type].append(attributes[attr_type][match])
            
            # Add results
            for key, value in entities.items():
                results[key].append(value)
    
    return pd.DataFrame(results)

#%%
# Test on a sample first
sample_size = 100
sample_texts = dataset_raw['Room Description'].sample(n=sample_size, random_state=42)

print("Running BERT analysis...")
bert_results = extract_with_bert(sample_texts, {
    'room_types': room_types_list,
    'bed_types': bed_types_list,
    'views': room_view_list,
    'amenities': room_amenities_list
})

#%%
# Analyze the results
def analyze_bert_results(df):
    for col in ['room_types', 'bed_types', 'views', 'amenities']:
        print(f"\nMost common {col}:")
        all_items = [item for sublist in df[col] if sublist for item in sublist]
        if all_items:
            print(pd.Series(all_items).value_counts().head(10))
        else:
            print("No items found")

analyze_bert_results(bert_results)

#%%
# If results look good, process the full dataset
print("\nProcessing full dataset...")
full_bert_results = extract_with_bert(dataset_raw['Room Description'], {
    'room_types': room_types_list,
    'bed_types': bed_types_list,
    'views': room_view_list,
    'amenities': room_amenities_list
})

analyze_bert_results(full_bert_results)

#%%
# Add results to original dataframe
dataset_new = dataset_raw.copy()
dataset_new['bert_room_types'] = full_bert_results['room_types']
dataset_new['bert_bed_types'] = full_bert_results['bed_types']
dataset_new['bert_views'] = full_bert_results['views']
dataset_new['bert_amenities'] = full_bert_results['amenities']

#%%
dataset_new.to_excel('Data/dataset_bert_extraction_v1.xlsx', index=False)
