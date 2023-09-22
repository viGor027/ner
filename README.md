# Overview

Idea of the project was to practice creating of a seq2seq model,
and learning on how to preprocess the dataset as well as searching for hyperparameters.

## Tech stack
- tensorflow, keras
- pandas
- sklearn

## Process
After preprocessing a dataset and making it sutiable for the model my approach was to run BayesianOptimization with not many epochs(due to lack of computational power) to pre-select hyperparameters of the model, and then continuing its training with best hiperparameters found for some more epochs using earlystopping.

# Results

Sparse Categorical Crossentropy on the test set was: 1.4120

# Conclusion
Due to lack of resources(low RAM capacity), number of words that was stored in tokenizer had to be decreaed to 500 words, what is the most likely cause of poor model performance. For further model tweaking number of words stored in tokeninzer's dictionary should be increased.
