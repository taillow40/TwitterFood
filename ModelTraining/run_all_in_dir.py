from model_code.CNN import run_cnn
import os
import json
import sys

if __name__ == "__main__":
    directory = 'configs'
    all_paths = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    for config_path in all_paths:
        with open(config_path, 'r') as file:
            config = json.load(file)
        try:
            preprocessed_data_path = config["preprocessed_data_path"]
            label_path = config["label_path"]
            save_dir = config["save_dir"]
            batch_size = config["batch_size"]
            learning_rate = config["learning_rate"]
            epochs = config["epochs"]
            dropout = config["dropout"]
            word2vec_path = config["word2vec_path"]
            embedding_size = config["embedding_size"]
        except KeyError as e:
            print(f"Missing key in configuration: {e}")
        run_cnn(
                preprocessed_data_path,
                label_path,
                word2vec_path,
                batch_size,
                save_dir,
                embedding_size,
                dropout,
                learning_rate,
                epochs
            )
        try:
            print("Configuration: ", config_path)
        except:
            print(f"Error when running configuration: {config_path}")
            #print the error
            print("Error: ", sys.exc_info()[0])
            continue