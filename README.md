# BI-LSTM-CRF-for-NER
This work aims to provide a machine learning model for the Name Entity Recognition of Biochemical-oriented text documents. 

It uses a word-level embedding based on the vocabulary met on the training data and the Embedding layer provided by the Pytorch framework. However, some task-specific features are added. As an example, the following information are encoded : 
- if a word contains only numeric characters
- if a word contains uppercase characters not in the first position.

The neural network contains a Bi-LSTM module and a Conditional Random Field Module (CRF). The Bi-LSTM provides the features to the CRF module which look at two types of potential before attributing a score to a pair of words : the emission and the transition. Then the Vertibi algorithm is used to find the best probability for the tag sequence. 

This work is inspired by the paper “Empower Sequence Labeling with Task-Aware Neural Language Model » (Liyuan Liu et al., 2017) which presents a model for NER based on Bi-LSTM and CRF but used both a word-level and a character-level embedding. 

To run this model : 

1) create the conda environment described by the “environment.yml” file, by running the “install_requirements.sh”.
2) For the training phase : run the “run_model.sh” file with the following arguments : train <train_file_path> <val_file_path>. The <train_file_path> and the <val_file_path> are respectively the relative path to the .txt files containing the training and validation data sets.
The final command must look like “bash run_model.sh train <train_file_path> <val_file_path>”

3) Some files will be generated during the training process :
- “vst229547.pt” file containing the checkpoints of the model during the training
- “vst229547_model” containing the final state dictionnary of the model (only if the training is ended) 
- “word_to_ix.pkl” :  a dictionary containing all the vocabulary words and the respective attributed index
- “tag_to_idx.pkl” : a dictionary containing all the tags and their attributed index
-  “Plot_training_data.png” : a chart representing the evolution of the loss on the validation data set and the training data set across the epochs.

4) For the inference phase : based on the model file or the checkpoints (if the training didn’t get finished), the model is loaded and based on the 2 dictionaries containing the words and the tags associated with their index, an output file will be generated containing the predicted tags.
The command is : ‘bash run_model.sh test <test_file_path> outputfile.txt’ where <test_file_path> is the relative path to the test data set. 
