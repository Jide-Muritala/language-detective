# Language Classification using TensorFlow
This repository contains a Jupyter Notebook for classifying languages using TensorFlow and Transformers. The notebook demonstrates the process of building a language classification model that can identify the language of a given text.

## Description
The notebook walks through the following steps:

**Loading Packages**: Importing necessary libraries such as TensorFlow, NumPy, Pandas, and Scikit-learn.

**Data Preparation**: Loading and preprocessing the dataset, including label encoding and text cleaning.

**Model Building**: Creating a neural network model using TensorFlow’s Keras API.

**Training**: Training the model on the dataset.

**Evaluation**: Evaluating the model’s performance using accuracy and confusion matrix.

**Prediction**: Making predictions on new text data.

## Installation
To run the notebook, you need to install these packages using pip:

- classifier_language_recognition-tensor-flow.ipynb
```bash
pip install tensorflow numpy pandas scikit-learn wikipedia-api
```
- classifier-transformers.ipynb
```bash
pip install transformers torch scikit-learn accelerate pandas
```
Dataset
```bash
wget https://downloads.tatoeba.org/exports/sentences.tar.bz2
tar -xvjf sentences.tar.bz2
```
## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Jide-Muritala/language-detective-tensorflow.git
   ```
2. Navigate to the repository directory:
   ```bash
   cd language-detective-tensorflow
   ```
3. Open the Jupyter Notebook:
 
4. Follow the steps in the notebook to train and evaluate the model.

## Example
Here is an example of how to use the model to predict the language of a new text:
```bash
new_text = ["tensorflow es una gran herramienta puedes encontrar muchos tutoriales de packt"]
test_text = tok.texts_to_sequences(new_text)
test_text = tf.keras.preprocessing.sequence.pad_sequences(test_text, maxlen=maxlen)
predictions = model.predict(test_text)
print(predictions.argmax())  # Output: 3 (Spanish)
```


