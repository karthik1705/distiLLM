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
dataset_raw = pd.read_csv('Data/Product_Normalization_GRI.csv')
dataset_raw.head()

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
    processed = []
    for pattern in pattern_list:
        # Handle NaN values and empty strings
        if pd.isna(pattern) or str(pattern).strip() == '':
            continue
        else:  # Just lowercase the whole thing
            if pattern:  # If pattern isn't empty
                processed.append(pattern.lower().strip())
    
    # Remove duplicates and empty strings
    return [p for p in list(set(processed)) if p.strip()]

#%% Extract attributes using spaCy
def analyze_with_spacy(texts):
    """Extract attributes using spaCy NER with confidence scores"""
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
    
    # Custom labels for hospitality domain
    room_patterns = [
        {"label": "ROOM_TYPE", "pattern": [{"LOWER": {"IN": processed_room_types}}]},
        {"label": "BED_TYPE", "pattern": [{"LOWER": {"IN": processed_bed_types}}]},
        {"label": "VIEW_TYPE", "pattern": [{"LOWER": {"IN": processed_room_views}}]}
    ]
    
    # Add patterns to pipeline
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
    else:
        ruler = nlp.get_pipe("entity_ruler")
    ruler.add_patterns(room_patterns)
    
    results = []
    
    # Process each text with batch processing
    batch_size = 32
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
                "view_types": []
            }
            
            # Check for accessibility keywords
            is_accessible = 'access' in doc.text.lower() or 'ada' in doc.text.lower()
            
            for ent in doc.ents:
                # Default confidence score for rule-based matches
                score = 1.0
                
                # Try to get tensor score if available
                try:
                    if hasattr(ent, 'tensor') and ent.tensor is not None:
                        score = float(ent.tensor.max())
                except (ValueError, AttributeError):
                    pass
                
                if ent.label_ == "ROOM_TYPE":
                    entities["room_types"].append((ent.text.lower(), score))
                elif ent.label_ == "BED_TYPE":
                    entities["bed_types"].append((ent.text.lower(), score))
                elif ent.label_ == "VIEW_TYPE":
                    entities["view_types"].append((ent.text.lower(), score))
            
            # Add Accessible Room if keywords found
            if is_accessible:
                entities["room_types"].append(("accessible room", 1.0))
            
            # Sort each list by confidence score
            for key in ["room_types", "bed_types", "view_types"]:
                entities[key] = sorted(entities[key], key=lambda x: x[1], reverse=True)
            
            results.append(entities)
    
    return pd.DataFrame(results)



#%%
'''
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

# Print the processed patterns for verification
print("Processed Room Types:", preprocess_pattern_list(room_types_list))
print("Processed Bed Types:", preprocess_pattern_list(bed_types_list))
print("Processed View Types:", preprocess_pattern_list(room_view_list))
print("Processed Amenities:", preprocess_pattern_list(room_amenities_list))

# Run analysis on a sample first to verify results
sample_size = 100
sample_texts = dataset_raw['Room Description'].sample(n=sample_size, random_state=42)

print("\nRunning spaCy analysis with transformer model...")
spacy_results = analyze_with_spacy(sample_texts)

#%%
analyze_entity_frequencies(spacy_results)
#%%
spacy_results['room_types'].values
#%%
# Add extracted features back to original sample dataframe
dataset_sample_new = pd.DataFrame(sample_texts.copy())
dataset_sample_new['extracted_room_types'] = spacy_results['room_types'].values
dataset_sample_new['extracted_bed_types'] = spacy_results['bed_types'].values
dataset_sample_new['extracted_view_types'] = spacy_results['view_types'].values
dataset_sample_new['extracted_amenities'] = spacy_results['amenities'].values

dataset_sample_new.to_excel('Data/dataset_spacy_extraction_sample_v2.xlsx', index=False)
'''

#%%
# If results look good, process the full dataset
print("\nProcessing full dataset...")
full_results_spacy = analyze_with_spacy(dataset_raw['Room Description'])

#%% Evaluate spaCy accuracy on full dataset
def evaluate_spacy_accuracy(spacy_results, full_data):
    """Evaluate spaCy extraction accuracy with confidence scores"""
    if len(spacy_results) != len(full_data):
        print(f"Warning: Results ({len(spacy_results)}) and samples ({len(full_data)}) length mismatch")
        n_samples = min(len(spacy_results), len(full_data))
        spacy_results = spacy_results.iloc[:n_samples]
        full_data = full_data.iloc[:n_samples]
    
    correct_any = 0
    correct_top = 0
    total = len(full_data)
    
    print("\nSpaCy Extraction Evaluation:")
    print("===========================")
    
    for i, (_, row) in enumerate(full_data.iterrows()):
        true_label = str(row['Guest Room Info']).lower()
        predictions = spacy_results.iloc[i]['room_types']
        
        if not predictions:
            continue
            
        # Check top prediction
        top_pred = predictions[0][0].lower()
        top_match = true_label in top_pred or top_pred in true_label
        
        # Check any prediction
        any_match = any(true_label in pred[0].lower() or pred[0].lower() in true_label 
                       for pred in predictions)
        
        correct_top += top_match
        correct_any += any_match
        
        if i < 5:
            print(f"\nDescription: {row['Room Description']}")
            print(f"True Label: {true_label}")
            print(f"Top Prediction: {top_pred} (score: {predictions[0][1]:.3f})")
            print(f"All Predictions: {[(p[0], f'{p[1]:.3f}') for p in predictions]}")
            print(f"Top Match: {'✓' if top_match else '✗'}")
            print(f"Any Match: {'✓' if any_match else '✗'}")
    
    top_accuracy = correct_top / total
    any_accuracy = correct_any / total
    
    print(f"\nTop Prediction Accuracy: {top_accuracy:.2%}")
    print(f"Any Match Accuracy: {any_accuracy:.2%}")
    
    return {"top_accuracy": top_accuracy, "any_accuracy": any_accuracy}

#%%
# Run spaCy on full dataset and evaluate
#spacy_results = analyze_with_spacy(dataset_raw['Room Description'])
accuracy_spacy = evaluate_spacy_accuracy(full_results_spacy, dataset_raw)

#dataset_new.to_excel('Data/dataset_spacy_extraction_v1.xlsx', index=False)

#%%
'''
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
                threshold = 0.7  # Adjust this threshold as needed
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
dataset_sample_new

#%% Saving sample results
bert_results.to_excel('Data/dataset_bert_extraction_sample_v2.xlsx', index=False)


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
'''



#%%
def extract_with_bert_modified(texts, attribute_lists):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    attributes = {
        'room_types': preprocess_pattern_list(room_types_list),
        'bed_types': preprocess_pattern_list(bed_types_list),
        'views': preprocess_pattern_list(room_view_list)
    }
    
    # Encode attributes once
    attribute_embeddings = {}
    for attr_type, attr_list in attributes.items():
        attribute_embeddings[attr_type] = model.encode(attr_list, convert_to_tensor=True)
    
    results = []
    batch_size = 32
    
    # Convert texts to list if it's a Series
    texts = texts.tolist() if isinstance(texts, pd.Series) else texts
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        batch_texts = [str(text) if pd.notna(text) else "" for text in batch_texts]
        text_embeddings = model.encode(batch_texts, convert_to_tensor=True)
        
        for idx, (text_embedding, text) in enumerate(zip(text_embeddings, batch_texts)):
            matches = {}
            
            # Check for accessibility keywords
            text_upper = text.upper()
            is_accessible = 'ACCESS' in text_upper or 'ADA' in text_upper
            
            for attr_type, attr_embeddings in attribute_embeddings.items():
                if len(text_embedding.shape) == 1:
                    text_embedding = text_embedding.unsqueeze(0)
                
                similarities = util.cos_sim(text_embedding, attr_embeddings)[0]
                
                # Dynamic thresholding
                max_sim = torch.max(similarities).item()
                if max_sim < 0.3:
                    matches[attr_type] = []
                    continue
                
                relative_threshold = max_sim * 0.8
                matched_indices = torch.where(similarities > relative_threshold)[0]
                
                matches[attr_type] = [
                    (attributes[attr_type][i], similarities[i].item())
                    for i in matched_indices.tolist()
                ]
                
                # If it's room_types and the description is accessible
                if attr_type == 'room_types' and is_accessible:
                    # Add or prioritize 'Accessible' room type
                    accessible_score = 1.0  # High confidence score
                    matches[attr_type].append(('Accessible Room', accessible_score))
            
            results.append(matches)
    
    return pd.DataFrame(results)


#%%
'''
# Test on a sample first
sample_size = 100
sample_texts = dataset_raw['Room Description'].sample(n=sample_size, random_state=42)

print("Running (Modified) BERT analysis...")
bert_results = extract_with_bert_modified(sample_texts, {
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
bert_results['text'] = sample_texts.values

#%% Saving sample results

bert_results.to_excel('Data/dataset_bert_modified_extraction_sample_v2.xlsx', index=False)
'''


# %%
def evaluate_bert_accuracy(bert_results, sampled_data):
    """Evaluate BERT extraction accuracy against true labels"""
    # Ensure we're using the same number of samples
    if len(bert_results) != len(sampled_data):
        print(f"Warning: Results ({len(bert_results)}) and samples ({len(sampled_data)}) length mismatch")
        # Use the smaller length
        n_samples = min(len(bert_results), len(sampled_data))
        bert_results = bert_results.iloc[:n_samples]
        sampled_data = sampled_data.iloc[:n_samples]
    
    correct = 0
    total = len(sampled_data)
    
    print("\nBERT Extraction Evaluation:")
    print("===========================")
    
    for i, (_, row) in enumerate(sampled_data.iterrows()):
        true_label = str(row['Guest Room Info']).lower()
        # Extract just the room type strings from the (type, score) tuples
        predicted_types = [item[0].lower() for item in bert_results.iloc[i]['room_types']]
        
        # Check for match
        match = any(true_label in pred or pred in true_label for pred in predicted_types)
        correct += match
        
        # Print first 5 examples
        if i < 5:
            print(f"\nDescription: {row['Room Description']}")
            print(f"True Label: {true_label}")
            print(f"Predicted Types: {predicted_types}")
            print(f"Match: {'✓' if match else '✗'}")
    
    accuracy = correct / total
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    
    return accuracy

#%%
# First run BERT extraction on the sampled data
# sample_size = 100
# Running on full data
full_data = dataset_raw #.sample(n=sample_size, random_state=42)
bert_results = extract_with_bert_modified(full_data['Room Description'], {
    'room_types': room_types_list,
    'bed_types': bed_types_list,
    'views': room_view_list
})

# Then evaluate
accuracy = evaluate_bert_accuracy(bert_results, full_data)
# %%
bert_results['room_types'][0]
# %%
def evaluate_bert_accuracy_2(bert_results, full_data):
    """Evaluate BERT extraction accuracy against true labels"""
    if len(bert_results) != len(full_data):
        print(f"Warning: Results ({len(bert_results)}) and samples ({len(full_data)}) length mismatch")
        n_samples = min(len(bert_results), len(full_data))
        bert_results = bert_results.iloc[:n_samples]
        full_data = full_data.iloc[:n_samples]
    
    correct_any = 0  # Counter for any match
    correct_top = 0  # Counter for top prediction match
    total = len(full_data)
    
    print("\nBERT Extraction Evaluation:")
    print("===========================")
    
    for i, (_, row) in enumerate(full_data.iterrows()):
        true_label = str(row['Guest Room Info']).lower()
        predictions = bert_results.iloc[i]['room_types']
        
        if not predictions:  # Skip if no predictions
            continue
            
        # Sort predictions by confidence score
        sorted_preds = sorted(predictions, key=lambda x: x[1], reverse=True)
        top_pred = sorted_preds[0][0].lower()
        
        # Check if true label matches top prediction
        top_match = true_label in top_pred or top_pred in true_label
        
        # Check if true label matches any prediction
        any_match = any(true_label in pred[0].lower() or pred[0].lower() in true_label 
                       for pred in predictions)
        
        correct_top += top_match
        correct_any += any_match
        
        # Print first 5 examples
        if i < 5:
            print(f"\nDescription: {row['Room Description']}")
            print(f"True Label: {true_label}")
            print(f"Top Prediction: {top_pred} (score: {sorted_preds[0][1]:.3f})")
            print(f"All Predictions: {[(p[0], f'{p[1]:.3f}') for p in sorted_preds]}")
            print(f"Top Match: {'✓' if top_match else '✗'}")
            print(f"Any Match: {'✓' if any_match else '✗'}")
    
    top_accuracy = correct_top / total
    any_accuracy = correct_any / total
    
    print(f"\nTop Prediction Accuracy: {top_accuracy:.2%}")
    print(f"Any Match Accuracy: {any_accuracy:.2%}")
    
    return {"top_accuracy": top_accuracy, "any_accuracy": any_accuracy}

#%%
accuracy = evaluate_bert_accuracy_2(bert_results, full_data)
# %%
