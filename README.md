## Twitter Food Popularity Sentiment Analysis

### Overview
With the rise of social media, platforms like Twitter provide an endless stream of public opinions, including thoughts on food. This project aims to automate the process of analyzing sentiment in food-related tweets. Using the Sentiment140 dataset, we’ll train two models, a Convolutional Neural Network and a Naive Bayes classifier to classify the positivity of tweets scored 0 for negative and 1 for positive, with a goal of 80% accuracy. We’ll preprocess data using techniques like word embeddings and term frequency matrices and apply these models to tweets mentioning food. Our system will identify popular dishes, measure public sentiment, and present trends using visualizations over time. This project demonstrates how natural language processing can turn social media data into actionable insights.

### Features

- Data Preprocessing
    - Embeddings
        - Generate CBOW and Skip Gram embeddings of dimension 100 and 200
    - Term Frequency Matrices
        - Generate TF and TF-IDF Matrices

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

Dataset Link: https://www.kaggle.com/datasets/kazanova/sentiment140

Data
- `data_by_words.pkl`: tokenized data
- `data_by_words_preprocessed.pkl`: tokenized data that has been cleaned
- `labels.pkl`: sentiment labels for our data
- `twitter.csv`: raw Sentiment 140 Data

Word2Vec Models for Embedding Layer
- `word2vec_100_cbow.model`: word2vec embeddings generated using CBOW dim size 100
- `word2vec_200_cbow.model`: word2vec embeddings generated using CBOW dim size 200
- `word2vec_100_sg.model`: word2vec embeddings generated using Skip Gram dim size 100
- `word2vec_200_sg.model`: word2vec embeddings generated using Skip Gram dim size 200

### Repo Overview/Structure

- `/FoodIntegration`: Food related code
    - `GraphPopularFood.ipynb`: graphs popular foods based on sentiment classification
    - `TweetsByFood.ipynb`: pulls out food related tweets
- `/ModelCode`: Model train and eval code
    - `CNN.py`: CNN train, eval, infer
    - `naive_bayes.py`: Naive Bayes preprocess, train, eval
- `/ModelTraining`: Code and configuration files to train the model
    - `/Configs`: configuration files to easily train our models
        - `CNN_config1.json`: CBOW 100 dim 
        - `CNN_config2.json`: CBOW 200 dim
        - `CNN_config3.json`: SG 100 dim
        - `CNN_config4.json`: SG 200 dim
    - `/DisabledConfigs`: unused configuration files
        - `bayes_config.json`: Naive bayes k fold ablation
        - `SAM_CNN_config.json`: Best model config with different paths
    - `/ModelOutputs`: model outputs/ performance metrics for various experiments run
        - `/CBOW 100 embedding dim`: Accuracy report and loss plot
        - `/CBOW 200 embedding dim`: Accuracy report and loss plot
        - `/Skip Gram 100 embedding dim`: Accuracy report and loss plot
        - `/Skip Gram 200 embedding dim`: Accuracy report and loss plot
        - `kfold_ablation_results.json`: Accuracy report and loss plot
    - `run_all_in_dir.py`: code to run all the configuration experiments in a folder
    - `run_models.py`: code to run a single configuration file from the command line
- `/preprocess`
    - `text_preprocessing.ipynb`: notebook to preprocess the text data into word embedding representations

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
- preprocessed_data_paths: Path to preprocessed data needed to run the model (data by words preprocessed pickle file)
- save_dir: path to directory to save model results (trained model weights and biases as well as loss plot)
- batch_size
- learning rate
- epochs: train epochs
- dropout: dropout rate
- word2vec_path: path to trained word2vec model for text word embedding layer cnn (null for audio cnn and document embedding cnn)

#### Running model training and evaluation or inference

1. clone the repo `git clone https://github.com/taillow40/TwitterFood.git`
2. Download all necessary data
3. Update the config files in `/ModelTraining/Configs` to point to the correct data paths
4. navigate to the `/ModelTraining` folder. `cd ModelTraining`
5. Run the following command: `python run_models.py --config <path to config file>`

Examples:

`python run_models.py --config "TwitterFood/ModelTraining/Configs/CNN_config2.json"`

### Resources/ References

Sentiment 140 Dataset: https://www.kaggle.com/datasets/kazanova/sentiment140
