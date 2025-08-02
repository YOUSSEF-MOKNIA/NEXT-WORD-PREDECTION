# Next Word Prediction using LSTMs
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/YOUSSEF-MOKNIA/NEXT-WORD-PREDECTION)

This repository contains a deep learning project that builds and trains a model to predict the next word in a sequence of text. The model uses a Long Short-Term Memory (LSTM) network, a type of Recurrent Neural Network (RNN), implemented with TensorFlow and Keras.

The project is trained on the text of "The Adventures of Sherlock Holmes" by Arthur Conan Doyle, learning the author's linguistic style to generate text.

## How It Works

The core of this project is the `Next_Word_Prediction.ipynb` notebook, which walks through the entire process:

1.  **Data Loading**: The text from `sherlock-holm.es_stories_plain-text_advs.txt` is loaded into memory.

2.  **Text Preprocessing**:
    *   **Tokenization**: The text is tokenized using `tf.keras.preprocessing.text.Tokenizer`, creating a vocabulary of all unique words and mapping them to integer indices.
    *   **N-gram Creation**: The corpus is converted into sequences of n-grams. For each line of text, subsequences are generated. For example, the phrase "he is always the woman" would produce sequences like `[he, is]`, `[he, is, always]`, `[he, is, always, the]`, and so on.
    *   **Padding**: Since the network requires inputs of a consistent length, the sequences are pre-padded to match the length of the longest sequence.
    *   **Input/Output Split**: The padded sequences are split into input features (`X`) and a target label (`y`). The last word of each sequence becomes the label, and the preceding words form the input.

3.  **Model Architecture**: A Keras Sequential model is constructed with the following layers:
    *   An `Embedding` layer to learn a dense vector representation for each word in the vocabulary.
    *   An `LSTM` layer with 150 units to process the sequential data and learn long-range dependencies.
    *   A `Dense` output layer with a `softmax` activation function, which outputs a probability distribution over the entire vocabulary for the next word.

4.  **Training**: The model is compiled with the `adam` optimizer and `categorical_crossentropy` loss function. It's trained on the prepared data for 100 epochs.

5.  **Prediction**: A function takes a seed text, predicts the most likely next word, appends it to the input text, and repeats the process to generate a sequence of new words.

## How to Run

To run this project on your local machine, follow these steps:

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/youssef-moknia/next-word-predection.git
    cd next-word-predection
    ```

2.  **Install Dependencies**
    You will need Python with TensorFlow and NumPy installed.
    ```bash
    pip install tensorflow numpy
    ```

3.  **Run the Jupyter Notebook**
    Launch the Jupyter Notebook to see the code, visualizations, and outputs.
    ```bash
    jupyter notebook Next_Word_Prediction.ipynb
    ```
    You can execute the cells sequentially to train the model and generate text.

## Example

An example of the model's prediction capabilities from the notebook:

*   **Seed Text**: `"i am"`
*   **Generated Text**: `"i am glad to hear"`

## Further Reading

For a more detailed walkthrough and explanation of the concepts, check out the accompanying Medium article:

[**Next Word Prediction Model with Python and Deep Learning**](https://medium.com/@CrazyForCode/next-word-prediction-model-with-python-and-deep-learning-de04daf31950)

## Want More?
Check out the full article on Medium [here](https://medium.com/@CrazyForCode/next-word-prediction-model-with-python-and-deep-learning-de04daf31950) for a detailed explanation!
