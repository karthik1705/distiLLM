#%%
import pandas as pd
from collections import Counter
import re

#%%
dataset_raw = pd.read_csv('Data/Product_Normalization_GRI.csv')
dataset_raw.head()

#%%
# Accessible Room without ADA or Accessible keywords
# Find rooms marked as Accessible but without ADA/Access/ible keywords in description => 0
access_analysis = dataset_raw[
    (dataset_raw['Guest Room Info'] == 'Accessible Room') & 
    (~dataset_raw['Room Description'].str.contains('ADA|ACC|ACESS', case=False, na=False))
]

# Display count and sample of these rows
print(f"\nNumber of rooms marked Accessible without ADA/Accessible keywords: {len(access_analysis)}")
print("\nSample of these rows:")
print(access_analysis[['Room Description', 'Guest Room Info']].head())


#%%
#dataset_raw['Room Description'].where(dataset_raw['Room Description'].str.contains('ACCESS'))

# Find rows where 'ADA|ACC\ACESS' appears in Room Description but RoomType is not 'Accessible Room'
access_analysis = dataset_raw[
    (dataset_raw['Room Description'].str.contains('ADA|ACC|ACESS', case=False, na=False)) & 
    (dataset_raw['Guest Room Info'] != 'Accessible Room')
]

# Display count and the matching rows
print(f"Number of rows with 'Accessible Room' in description but not marked as Accessible Room: {len(access_analysis)}")
print("\nSample of these rows:")
print(access_analysis[['Room Description', 'Guest Room Info']])

access_analysis['Guest Room Info'].value_counts()



#%%
# Tokenize and get word frequencies from Room Description

# Function to clean and tokenize text
def tokenize_text(text):
    if pd.isna(text):  # Handle NaN values
        return []
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace specific compound terms before tokenization
    compound_terms = {
        'bed room': 'bedroom'
    }
    
    for term, replacement in compound_terms.items():
        text = text.replace(term, replacement)
    
    # Remove special characters and numbers, keep only letters
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    return words

# Get all words from Room Description
all_words = []
for description in dataset_raw['Room Description']:
    all_words.extend(tokenize_text(description))

# Count word frequencies
word_freq = Counter(all_words)

# Convert to DataFrame for better visualization
word_freq_df = pd.DataFrame.from_dict(word_freq, orient='index', columns=['count'])
word_freq_df = word_freq_df.sort_values('count', ascending=False)

#%%

# Display top 20 most common words
print("Top 20 most frequent words in Room Description:")
print(word_freq_df.head(20))

# %%
dataset_raw['Guest Room Info'].value_counts()



# %%
# Text Analysis and Clusterization using Embeddings

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For generating text embeddings
from sentence_transformers import SentenceTransformer

# For clustering
from sklearn.cluster import KMeans

# For visualization (dimensionality reduction)
from sklearn.manifold import TSNE

# %%
# ----------------------------
# 1. Sample Data Preparation
# ----------------------------
# Replace this sample data with your actual dataset
data = {
    "description": dataset_raw['Room Description'],
    "current_label": dataset_raw['Guest Room Info']
}

# Create a DataFrame
df = pd.DataFrame(data)

# ----------------------------------
# 2. Generate Embeddings for the Text
# ----------------------------------
# Load a pre-trained sentence transformer model
# You can choose another model if needed (e.g., 'all-MiniLM-L6-v2', 'paraphrase-MiniLM-L6-v2', etc.)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings for each room description
embeddings = model.encode(df['description'].tolist(), convert_to_numpy=True)

# ----------------------------
# 3. Clustering the Embeddings
# ----------------------------
# Define the number of clusters (adjust as needed)
num_clusters = 4

# Use KMeans clustering on the embeddings
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Add the cluster labels to the DataFrame
df['cluster'] = clusters

# ----------------------------
# 4. Visualize the Clusters using t-SNE
# ----------------------------
# Reduce embeddings to 2 dimensions using t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Add t-SNE coordinates to the DataFrame
df['tsne_x'] = embeddings_2d[:, 0]
df['tsne_y'] = embeddings_2d[:, 1]

#%%
# Plot the clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='tsne_x', y='tsne_y', hue='cluster', data=df, palette='tab10', s=100, alpha=0.8)
# Optionally, annotate the points with the room description (or part of it)
#for i, row in df.iterrows():
#    plt.text(row['tsne_x'] + 0.2, row['tsne_y'] + 0.2, row['description'], fontsize=9)

plt.title("t-SNE Visualization of Room Description Clusters")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(title="Cluster")
plt.show()

# ----------------------------
# 5. Review the Cluster Assignments
# ----------------------------
# Print the DataFrame with room descriptions, current labels, and assigned cluster
print(df[['description', 'current_label', 'cluster']])

# %%
#df.to_csv('Data/Product_Normalization_GRI_clustered.csv', index=False)

#%%
# Topic Modeling Prep
dataset_normalized_attributes = pd.read_excel('Data/Normalized_product_attribute_name.xlsx', sheet_name = 'Normalized Product Attributes')
dataset_normalized_attributes

room_types = dataset_normalized_attributes['RoomType'].dropna().unique()
room_types_list = room_types.tolist()

# %%
'''
# Collating the Normalized Product Attribute data to be excluded through custom stop words
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
'''


#%%
#Topic modeling
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from bertopic import BERTopic
import plotly.express as px

#%%
# Clean and preprocess text (reusing your existing tokenize_text function)
def preprocess_for_topic_modeling(text):
    if pd.isna(text):
        return ""
    words = tokenize_text(text)
    # Remove very common words that might not be in standard stop words
    custom_stops = {'a', 'an', 'the', 'in', 'on', 'at', 'with', 'and', 'or', 'to', 'for'}
    words = [w for w in words if w not in custom_stops]
    return ' '.join(words)

# Preprocess all descriptions
processed_texts = dataset_raw['Room Description'].apply(preprocess_for_topic_modeling)

# 1. LDA Approach
# --------------
def run_lda_analysis(texts, n_topics=10):
    # Create document-term matrix
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(texts)
    
    # Train LDA model
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=20
    )
    lda_output = lda.fit_transform(doc_term_matrix)
    
    # Print top words for each topic
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-10:-1]]
        print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
    
    return lda_output, lda, vectorizer

# 2. BERTopic Approach
# -------------------
def run_bertopic_analysis(texts, n_topics=10):
    topic_model = BERTopic(nr_topics=n_topics)
    topics, probs = topic_model.fit_transform(texts)
    
    # Get topic info
    topic_info = topic_model.get_topic_info()
    
    # Print topics
    for topic in topic_model.get_topics():
        if topic != -1:  # -1 is reserved for outliers
            print(f"Topic {topic}: {topic_model.get_topic(topic)}")
    
    return topics, probs, topic_model

#%%
# Run both analyses
print("Running LDA Analysis...")
lda_output, lda_model, vectorizer = run_lda_analysis(processed_texts)

print("\nRunning BERTopic Analysis...")
topics, probs, bertopic_model = run_bertopic_analysis(processed_texts)

#%%
# Visualization for LDA
# --------------------
def visualize_topics_distribution(lda_output):
    # Get the dominant topic for each document
    dominant_topics = np.argmax(lda_output, axis=1)
    topic_counts = pd.Series(dominant_topics).value_counts().sort_index()
    
    # Create visualization
    fig = px.bar(
        x=topic_counts.index + 1,  # Add 1 to make topics 1-based
        y=topic_counts.values,
        labels={'x': 'Topic Number', 'y': 'Number of Documents'},
        title='Distribution of Dominant Topics Across Documents'
    )
    fig.show()

# Visualization for BERTopic
# -------------------------
def visualize_bertopic_results(topic_model):
    # Topic visualization
    topic_model.visualize_topics()
    
    # Topic hierarchy
    topic_model.visualize_hierarchy()
    
    # Topic similarity
    topic_model.visualize_heatmap()

#%%
# Run visualizations
print("\nVisualizing LDA Results...")
visualize_topics_distribution(lda_output)

print("\nVisualizing BERTopic Results...")
visualize_bertopic_results(bertopic_model)

# Add topics to original dataframe
dataset_raw['LDA_Topic'] = np.argmax(lda_output, axis=1)
dataset_raw['BERTopic'] = topics
# %%
