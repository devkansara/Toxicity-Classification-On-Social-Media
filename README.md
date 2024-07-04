# Classification of Toxic Comments on Social Media

## Overview

This project aims to classify text comments into multiple categories of toxicity using various machine learning and deep learning models. The dataset used is provided by Kaggle and consists of comments labeled with six different categories of toxicity.

## Data

The dataset consists of 159,571 data points with the following features:
- `id`: Unique identifier for each comment.
- `comment_text`: The text of the comment.
- Six binary features representing different categories of toxicity:
  - `toxic`
  - `severe_toxic`
  - `obscene`
  - `threat`
  - `insult`
  - `identity_hate`

## Models Used

### Logistic Regression
- Used for multi-label classification.
- Combined word and character tokenization.
- Fine-tuned using cross-validation.

### Long Short-Term Memory (LSTM)
- Used for handling sequential data.
- Model architecture included an embedding layer, LSTM layer, maxpool layer, dropout layer, and a sigmoid activation layer.
- Applied binary cross-entropy loss and Adam optimizer.

### Ensemble Learning
- Combined Naive-Bayes classifier with logistic regression.
- Initial prediction made with Naive-Bayes, refined with logistic regression.
- Evaluated using ROC and precision-recall graphs.

### BERT
- Pre-processed data to remove unnecessary characters and handled missing values.
- Tokenized comments using BERT's tokenizer.
- Trained using Hugging Face's Trainer with AdamW optimizer and a learning rate scheduler.

## Quantitative Validation Methods
- Used cross-validation scores to evaluate model performance.
- Compared models based on prediction accuracy, ROC curves, and precision-recall curves.

## Results
- **Logistic Regression:** Cross-validation score of 0.978, with overall model accuracy of 91.44%.
- **LSTM:** Achieved 92.22% accuracy.
- **Ensemble Learning:** Highest accuracy at 92.39%.
- **BERT:** Demonstrated significant improvement in precision-recall curves and ROC curves.

