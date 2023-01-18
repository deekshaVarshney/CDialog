# CDialog: A Multi-turn Covid-19 Conversation Dataset for Entity-Aware Dialog Generation
Please find the link to the preprint version of the paper.
Link: https://drive.google.com/file/d/1peZVaapiDGF5qd54JPggMkyNy7mVF1zh/view?usp=sharing

This is an implementation of our paper: CDialog: A Multi-turn Covid-19 Conversation Dataset for Entity-Aware Dialog Generation (https://preview.aclanthology.org/emnlp-22-ingestion/2022.emnlp-main.782.pdf)

(EMNLP 2022)
Deeksha Varshney, Aizan Zafar, Niranshu Kumar Behera, Asif Ekbal

For requesting data, kindly fill this form :https://forms.gle/YQfPrYXKgAM5PZnw9

# Overview

The code and sample data is organized as follows:
* `data\` : contains the sample data of CDialog
* `models\` : has the model scripts for BART, Bert and BioBert
* `preprocess\` : contains the preprocess scripts for each of the model mentioned above

# Environment

To create the cdialog environment, run:

`$ conda create --name cdialog --file cdialog_env.txt`

# Preprocess

    Preprocess.py - preprocess the data using bert tokenizer and split the data in 80:10:10 ratio.

    Preprocess_biobert.py – preprocess the data using biobert tokenizer and split the data in 80:10:10 ratio.

    Preprocess_rand.py - preprocess the data first by randomizing the data then using bert tokenizer on it and split the data in 80:10:10 ratio.

    Preprocess_biobert_rand.py – preprocess the data first by randomizing the data then using biobert tokenizer on it and split the data in 80:10:10 ratio.

    Preprocess_entity.py – preprocess the data with entity using biobert tokenizer and split the data with entity in 80:10:10 ratio.

    Preprocess_entity_rand.py - preprocess the data with entity first by randomizing the data and entities simulatenously then using bert tokenizer on it and split the data with entity in 80:10:10 ratio.

    Preprocess_biobert_entity_rand.py - preprocess the data with entity first by randomizing the data and entities simulatenously then using biobert tokenizer on it and split the data with entity in 80:10:10 ratio.

# Arguments

### To train a model use the train.py script for that respective model.
    
   Arugments that can be used while running the train.py script (same for all models however default value may vary model to model):
   
   | Argument | Purpose |
   | --- | --- |
   | model_config | path to the config.json file |
   | gpu | gpu id on which you want the program to run |
   | epochs | number of epochs (default = 30) |
   | num_gradients | (default = 4) |
   | batch_size | (default = 16) |
   | lr | learning rate (default = 1e-5) |
   | load_dir | path to the directory where the models are to be saved/loaded |
   | validate_load_dir | path to the validate.pkl file |
   | train_load_dir | path to the .txt file to save the model training epochs and loss |
   | log_dir |  path to save the perplexity scores in a .txt file |
   | val_epoch_interval | interval at which you want to calculate the valid perplexity (default = 1) |
   | last_epoch_path | path to directory where the models are saved that can be use to finetune on another data |
   | hidden_size | (default = 512) |
   | vocab_size | (default = 50000) |
   | finetune | set 'true' to initiate finetuning (default = 'false') |

### To calculate the perplexity for the trained model use the perplexity.py script for that respective model.

   Arugments that can be used while running the perplexity.py script (same for all models however default value may vary model to model):

   | Argument | Purpose |
   | --- | --- |
   | model_config | path to the config.json file |
   | gpu | gpu id on which you want the program to run |
   | batch_size | (default = 1) |
   | load_dir | path to the directory where the models are to located to be loaded |
   | validate_load_dir | path to the validate.pkl file |
   | train_load_dir | path to the train.pkl file |
   | save_dir  |  path to save the perplexity scores in a .txt file |
   | hidden_size | (default = 512) |
   | vocab_size | (default = 50000) |

    
### To generate the evaluation metrics of the trained model use the generate.py script for that respective model.
    
   Arugments that can be used while running the generate.py script (same for all models however default value may vary model to model):
   
   | Argument | Purpose |
   | --- | --- |
   | gpu | gpu id on which you want the program to run |
   | top_k | (default = 50) |
   | temp | (default = 1.0) |
   | decoder_dir | path to the directory where the models are located to be loaded |
   | test_load_dir | path to the test.pkl file |
   | pred_save_dir | path to where you want to create a .txt file containing the predicted sentences |
   | reference_save_dir | path to where you want to create a .txt file containing the reference/actual sentences |
   | ouput_save_dir | path to where you want to create a .csv file containing the print statment of the .py file |
   | metric_save_dir | path to where you want to create a .txt file containing the evaluated metric scores |
   | hidden_size | (default = 512) |
   | vocab_size | (default = 50000) |

