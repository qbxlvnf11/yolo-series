import logging
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        
    return config

def get_logger(logging_file_path):
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    head = "%(asctime)s - %(filename)s - %(levelname)s - %(message)s"
    # logging.basicConfig(filename=str(logging_file_path),
    #                         format=head)
    formatter = logging.Formatter(head)

    ## Consol Log
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    ## File Log
    file_handler = logging.FileHandler(logging_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger