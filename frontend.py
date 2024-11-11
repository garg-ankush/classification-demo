import streamlit as st
import pickle
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


# Page configuration
st.set_page_config(
    page_title="Loneliness Classification App",
    page_icon="📝",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        height: 250px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title('📝 Loneliness Classification App')
st.markdown("""
    This is a demo app to classify text messages into pre-defined categories that can help
    specialists focus on the type of response they should give to the user.
""")

# Create two columns
col1, col2 = st.columns([2, 1])


with col1:
    # Add some helpful information
    st.markdown('#### How to use')
    st.markdown("""
    1. Enter or paste your text in the text area
    2. Click the 'Classify' button
    3. View the categories
    
    ### About
    This classifier can categorize text into the following classes:
    - Seeking Advice
    - Providing Help
    - Seeking Validation
    - Reaching Out
    - Non-Directed Interaction
    - Other
    """)

with col2:
    # Text input
    user_input = st.text_area(
        'Enter sample text here:', 
        placeholder='Type or paste your text here...'
    )

    @st.cache_resource
    def generate_embeddings(user_input):
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        model = AutoModel.from_pretrained('roberta-base')
        encoded = tokenizer(
            user_input,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = model(**encoded)
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        
        return embeddings


    @st.cache_resource
    def load_model():
        try:
            with open('model.pkl', 'rb') as file:
                label_encoder, model = pickle.load(file)
            return label_encoder, model
        except:
            return None

    label_encoder, model = load_model()

    # Classification button
    if st.button('Predict', type='primary'):
        if user_input:
            try:
                # Generate embeddings
                embeddings = generate_embeddings([user_input])
                
                # Make predictions
                prediction = model.predict(embeddings)

                # Inverse transform predictions to get category names
                prediction = label_encoder.inverse_transform(prediction)[0]

                # Map category names to human readable names    
                categories = {
                    'seeking_advice': 'Seeking Advice',
                    'providing_help': 'Providing Help',
                    'seeking_validation': 'Seeking Validation',
                    'reaching_out': 'Reaching Out',
                    'non_directed_interaction': 'Non-Directed Interaction',
                    'other': 'Other Type. Please check the message for more details.'
                }
                
                prediction = categories.get(prediction)
            
                # Display results
                st.success('Classification Complete!')
                st.write('### Results')
                st.write(f'Classification: **{prediction}**')
                
            except Exception as e:
                st.error(f'An error occurred: {str(e)}')
        else:
            st.warning('Please enter some text to classify.')
