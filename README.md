# Language Detective

The **Language Detective** provides several models to identify the language of a given text input. It includes FastText, TensorFlow, and Transformers-based classifiers. 

## Models

1. **FastText Classifier**:
   - **File**: `classifier-fasttext.ipynb`
   - A Jupyter Notebook implementing language classification using the FastText model. Utilizes the pretrained `lid.176.bin` model for language detection.

2. **TensorFlow Classifier**:
   - **File**: `classifier-tensorflow.ipynb`
   - A Jupyter Notebook that demonstrates language classification using TensorFlow.

3. **Transformers (BERT) Classifier**:
   - **File**: `classifier-transformers-bert.ipynb`
   - A Jupyter Notebook that utilizes BERT transformer model for language classification.

4. **Transformers (XLM) Classifier**:
   - **File**: `classifier-transformers-xlm.ipynb`
   - A Jupyter Notebook implementing language classification using the xlm-roberta-base transformer model.



## Getting Started
1. Clone the repository:
```bash
git clone https://github.com/Jide-Muritala/language-detective.git
```

2. Navigate to the repository directory:
```bash
cd language-detective
```

3. Install dependencies for the notebook you want to run:

- FastText
```bash
pip install fasttext numpy
```
For the FastText classifier, you need to download lid.176.bin and place it in the root directory.
```bash
wget https://dl.fbaipublicfiles.com/fasttext/supervised-languages/lid.176.bin
```

- TensorFlow
```bash
pip install tensorflow numpy pandas scikit-learn wikipedia-api
```

- Bert
```bash
pip install transformers torch scikit-learn accelerate pandas
```

- XLM
```bash
pip install transformers torch accelerate scikit-learn datasets sentencepiece pandas
```
For the XLM classifier, download and place the dataset file sentences.csv in the root directory.
```bash
wget https://downloads.tatoeba.org/exports/sentences.tar.bz2
tar -xvjf sentences.tar.bz2
```
