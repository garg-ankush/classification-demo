import pandas as pd
import numpy as np
from datasets import Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoTokenizer, AutoModel
from helpers import get_bert_embeddings
import pickle

tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModel.from_pretrained('roberta-base')

import warnings
warnings.filterwarnings('ignore')

################################
# Preprocessing
################################

# Import dataset
dataset = Dataset.from_file("data/train_set_dataset.arrow")
dataframe = dataset.to_pandas()
df = dataframe[['text', 'interaction']]

extending_columns = ['seeking_advice', 'providing_help', 'seeking_validation', 'reaching_out', 'non_directed_interaction']
# Split the interaction column into 5 separate columns
df[extending_columns] = pd.DataFrame(df['interaction'].tolist(), index=df.index)
df = df.drop('interaction', axis=1)
df['category'] =np.where(df[extending_columns].nunique(axis=1) == 1, 
                  'other', 
                  df[extending_columns].idxmax(axis=1))

df.drop(columns=extending_columns, inplace=True)

################################
# Filter dataset
################################

trimmed = []
for category in df['category'].unique():
    selected_df = df[df['category'] == category].sample(n=100, replace=True)
    trimmed.append(selected_df)

trimmed_df = pd.concat(trimmed)
trimmed_df = trimmed_df.sample(len(trimmed_df)) # Shuffle dataset
trimmed_df['category'].value_counts()

################################
# Train model
################################

X_train, X_test, y_train, y_test = train_test_split(trimmed_df['text'], trimmed_df['category'], test_size=0.2, random_state=42, stratify=trimmed_df['category'])

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Get BERT embeddings
X_train_embeddings = get_bert_embeddings(texts=X_train.tolist(), tokenizer=tokenizer, model=model)
X_test_embeddings = get_bert_embeddings(texts=X_test.tolist(), tokenizer=tokenizer, model=model)

# Train model
embeddings_pipeline = make_pipeline(
    RandomForestClassifier(n_estimators=100, random_state=42)
)
embeddings_pipeline.fit(X_train_embeddings, y_train)

predictions = embeddings_pipeline.predict(X_test_embeddings)

# Evaluate model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump([label_encoder, embeddings_pipeline], f)