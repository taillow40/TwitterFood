## Twitter Food Popularity Sentiment Analysis

### Overview
With the rise of social media, platforms like Twitter provide an endless stream of public opinions, including thoughts on food. This project aims to automate the process of analyzing sentiment in food-related tweets. Using the Sentiment140 dataset, we’ll train two models, a Convolutional Neural Network and a Naive Bayes classifier to classify the positivity of tweets scored 0 for negative and 1 for positive, with a goal of 80% accuracy. We’ll preprocess data using techniques like word embeddings and term frequency matrices and apply these models to tweets mentioning food. Our system will identify popular dishes, measure public sentiment, and present trends using visualizations over time. This project demonstrates how natural language processing can turn social media data into actionable insights.

### Features

- Data Preprocessing
    - Embeddings
        - 
    - Term Frequency Matrices
        - 

- Sentiment Analysis
    - CNN
        - Trains and evaluates a CNN to analyze the sentiment of the input text represented as a mapping to word embeddings (to be recognized by the embedding layer in the CNN)
        - Categorizes positive and negative tweets
        - Returns the test loss and acuuracy to the user
        - Saves trained model weights and biases
        - Saves a train and validation loss plot
    - Naive Bayes
        - 

### Data Sources

Dropbox Link: https://www.dropbox.com/scl/fo/ummxiaqz7tnd66vlb6ntn/ALsbMreFh9J_T7CNLMblAR0?rlkey=20e7h4ym5l125uvhvwib82atd&st=k7yg5j2s&dl=0

Data
- 

Word2Vec Models for Embedding Layer
-

### Repo Overview/Structure

### Getting Started

#### Prerequisites

Ensure you have all necessary data files downloaded as listed in the Data Sources section of the ReadMe

Ensure you have the following installed to run our models and preprocessing steps
- python
- torch
- numpy
- matplotlib
- pickle
- nltk
- gensim
- pandas
- sklearn

#### Config Files

Train Params:
- run_mode: train or infer
- preprocessed_data_paths: List of paths to preprocessed data needed to run the model. (Expected structure per model in example configs)
- save_dir: path to directory to save model results (trained model weights and biases as well as loss plot)
- batch_size
- learning rate
- epochs: train epochs
- dropout: dropout rate
- word2vec_path: path to trained word2vec model for text word embedding layer cnn (null for audio cnn and document embedding cnn)

Infer Params:
- run_mode: train or infer
- net_path: path to the model weights and biases saved from model training (pth file)
- input: input to infer on. A string text. 
- dropout: dropout rate
- word2vec_path: path to trained word2vec model for text cnns (null for audio cnn)

#### Running model training and evaluation or inference

1. clone the repo `git clone https://github.com/taillow40/TwitterFood.git`
2. Download all necessary data
3. Update the config files in `/config_files` to point to the correct data paths
4. navigate to the `/model_code` folder. `cd model_code`
5. Run the following command: `python run_models.py --config <path to config file>` (either inference or train config depending on goal)

Examples:

`python run_models.py --config "C:/Users/Jennie/Documents/NLP/Project/TwitterFood/configs/CNN_config.json"`

`python run_models.py --config "configs/CNN_config.json"`

### Resources/ References

Sentiment 140 Dataset: https://www.kaggle.com/datasets/kazanova/sentiment140
