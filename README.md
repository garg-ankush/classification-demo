# Loneliness Classification Demo

## Overview

This project demonstrates a machine learning application that classifies text into different categories of loneliness. It uses the FIG-Loneliness dataset to train models that can identify various aspects of loneliness expressions in text.

Demo: [Loneliness Classification App](https://classification-demo-0.streamlit.app/)

## Features

- Text classification for loneliness-related categories
- Simple and intuitive user interface
- Real-time predictions

## How It Works

1. **Data Source**: The project uses the [FIG-Loneliness dataset](https://huggingface.co/datasets/FIG-Loneliness/FIG-Loneliness) from Hugging Face, which contains annotated Reddit posts related to loneliness.

2. **Data Preprocessing**: The dataset was preprocessed to prepare it for model training.

3. **Model Training**: Two types of models were trained:
   - A simple TF-IDF (Term Frequency-Inverse Document Frequency) based model
   - An embeddings-based model for more nuanced text understanding

4. **Frontend Development**: A Streamlit app was created to provide an easy-to-use interface for text classification.

5. **Deployment**: The app is deployed and accessible online for public use.

## How to Use

1. Visit the [demo app](https://classification-demo-0.streamlit.app/).
2. Enter a text in the provided input field.
3. Click the "Predict" button.
4. View the classification results, which will show the predicted loneliness categories for the input text.

## Technical Details

- **Dataset**: FIG-Loneliness (2,633 lonely and 3,000 non-lonely Reddit posts)
- **Model**: Embeddings-based classifiers
- **Frontend**: Streamlit
- **Deployment**: Streamlit Cloud

## Future Improvements

- Implement more advanced NLP models for better accuracy
- Add explanations for predictions
- Expand the dataset for more diverse loneliness expressions