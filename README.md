Explainable BERT for Sentiment Analysis
This project implements a sentiment analysis model using BERT (Bidirectional Encoder Representations from Transformers). The goal is to classify movie reviews as positive or negative while making the predictions interpretable. This repository includes data preprocessing, model training, evaluation, and a method for explaining the model's decisions.

Table of Contents
Project Overview
Dataset
Model Architecture
Installation
Usage
Training the Model
Evaluation
Explainability
Results
Contributing
License
Project Overview
This project leverages BERT, a pre-trained transformer model, to perform sentiment analysis on movie reviews. The model is fine-tuned to classify text reviews as either positive or negative, and methods are implemented to make the model's predictions more explainable using techniques such as attention visualization.

Dataset
The dataset used in this project is the IMDB Dataset of 50K Movie Reviews. It consists of 50,000 reviews labeled as either positive or negative. The dataset is split into training and testing sets using train_test_split.

Model Architecture
The model is based on the BertForSequenceClassification architecture from the transformers library. It includes:

Preprocessing with the BERT tokenizer.
Fine-tuning of the BERT model for binary sentiment classification.
Use of a linear learning rate scheduler for optimizing training.
Installation
To run this project, ensure you have Python 3.7 or later installed. Install the required libraries using:

bash
Copy code
pip install torch pandas scikit-learn transformers nltk
Usage
Clone the repository:

bash
Copy code
git clone https://github.com/NOOB-del-ai/explainable-bert-fro-sentiment-analysis.git
cd explainable-bert-fro-sentiment-analysis
Run the Jupyter notebook: Open the xai-bert.ipynb file using Jupyter Notebook or JupyterLab and run the cells step by step.

Training the Model
The training process involves:

Loading and preprocessing the dataset.
Tokenizing text using the BERT tokenizer.
Creating PyTorch DataLoader instances for training and validation.
Fine-tuning the BERT model using AdamW optimizer.
Evaluation
After training, the model is evaluated using standard metrics such as accuracy, precision, recall, and F1 score to assess its performance on the test set.

Explainability
To make the model's predictions more interpretable, attention weights from the BERT model are visualized, providing insights into which parts of the input text contributed most to the model's decision.

Results
The model achieves good performance on the sentiment classification task. The use of BERT allows it to capture complex patterns in the text data, leading to high accuracy in identifying the sentiment of movie reviews.


MODEL_PARAMS FOLDER: fine tuned model is there.
