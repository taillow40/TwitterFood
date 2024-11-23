import argparse
import json
import os
from ModelCode.CNN import run_cnn

def load_config(config_path):
    """Load JSON configuration file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config
    else:
        raise FileNotFoundError(f"Config file not found at: {config_path}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run script with a JSON configuration file.")
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help='Path to the JSON configuration file'
    )
    args = parser.parse_args()

    # Load the configuration file
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(e)
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON config file: {e}")
        return

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
        return
    
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

if __name__ == '__main__':
    main()