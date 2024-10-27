# Language Detective
This repository contains a multilingual language predictor model trained on a subset of multilingual datasets. The model is designed to identify the language of a given text input among five target languages: English, Spanish, French, German, and Italian. This project uses xlm-roberta-base as the base model, a multilingual transformer model from Hugging Face’s Transformers library. The model is fine-tuned on multilingual datasets to predict the language of a given input text.

## Project Structure
- sentences.csv: Contains sample sentences labeled with language codes.
- language_predictor_model/: Folder where the trained model and tokenizer are saved for reuse.
- train_predict.py: Script to train and fine-tune the language predictor model.
- predict.py: Script for loading the model and running predictions on new sentences.
- requirements.txt: List of dependencies required to run the project.


## Setup
1. Clone the repository:
```bash
git clone https://github.com/Jide-Muritala/language-detective.git
```
2. Navigate to the repository directory:
```bash
cd language-detective-tensorflow
```
3. Install dependencies: Install the required packages:
```bash
pip install transformers torch accelerate scikit-learn datasets sentencepiece pandas
```
 
4. Download and Prepare Dataset: Place the dataset file sentences.csv in the root directory. This CSV file should contain three columns without headers: id, lang (language code), and text (sentence).
```bash
wget https://downloads.tatoeba.org/exports/sentences.tar.bz2
tar -xvjf sentences.tar.bz2
```



### Others
To run the other notebooks, you need to install these packages using pip:
- classifier_language_recognition-tensor-flow.ipynb
```bash
pip install tensorflow numpy pandas scikit-learn wikipedia-api
```
- classifier-transformers-bert.ipynb
```bash
pip install transformers torch scikit-learn accelerate pandas
```
