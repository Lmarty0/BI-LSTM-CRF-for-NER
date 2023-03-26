from cProfile import label
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from network import *
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()
parser.add_argument('--training_mode', default= "train")
parser.add_argument('--training_file', default='train.txt')
parser.add_argument('--validation_file', default='dev.txt')
parser.add_argument('--testing_file')
parser.add_argument('--output_file')

args = parser.parse_args()

training_mode = True

if args.training_mode == "train" : 
    training_mode = True
elif args.training_mode == "test" : 
    training_mode = False
else : 
    print('Argument error')


##### Class dataloader ####

class SentenceGetter(object) : 
    #A class which permits to take a txt file as describe in the requierement and return the list of the sentences
    #Each sentence is a list of tuple : (token, tag)
    #The lenght of the sentence is the number of words composing this sentence

    def __init__(self,data) :
        self.path = data
        with open(data, 'r') as f : 
            self.lines = f.readlines()
        self.tokens=[l.split('\t')[0] for l in self.lines]
        self.ner_label = []
        for l in self.lines : 
            ner = l.split('\t')
            if len(ner)== 2 :
                self.ner_label.append(ner[1][:-1])
            else :
                self.ner_label.append('Nan')

        self.df = pd.DataFrame(list(zip(self.tokens, self.ner_label)), columns=['Tokens', 'NER'])

        self.words = list(set(self.df["Tokens"].values))
        self.tags=list(set(self.df["NER"].values))
        self.tags.remove('Nan')

        #List of the sentences in the text
        self.sentences=[]
        s=[]
        for index, row in self.df.iterrows():
            if row['Tokens']=='\n' :
                self.sentences.append(s)
                s=[]
            else : 
                s.append((row['Tokens'], row['NER']))

        #Mapping of the words and the tags
        self.words2idx = {w : i for i,w in enumerate(self.words)}
        self.tags2idx = {t : i for i,t in enumerate(self.tags)}

        #Max lenght of the sentence for padding

        self.max_len=max([len(s)for s in self.sentences])



#### Utils functions #####

def is_UN(word) : 
    # A function that return True if the word has onlu numeric characters
    asw = True
    for c in word : 
        if not c.isnumeric() : 
            asw = False
    return asw

def has_UC(word) : 
    # Return True if a word has an uppercase character not in the first position
    asw = False
    for c in word[1:]:
        if c.isupper() : 
            asw = True
    return asw

def prepare_sequence(seq, to_ix):
    idxs=[]
    for w in seq : 
        if w in to_ix : 
            idxs.append(to_ix[w])
        else : 
            if is_UN(w) : #Only numerics 
                idxs.append(to_ix['U_N'])
            elif has_UC(w) : #has a capital letter not in the first position
                idxs.append(to_ix['UP_C'])
            else : #is not in the existing vocabulary
                idxs.append(to_ix['OOV'])

    return torch.tensor(idxs, dtype=torch.long).to(device)





#### Model #####

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH_CHECKPOINT = 'vst229547.pt'

if training_mode == True : 

    #### Import Data #####

    S=SentenceGetter(args.training_file)
    V=SentenceGetter(args.validation_file)

    ##### Training #####
    word_to_ix , tag_to_ix = S.words2idx, S.tags2idx
    tag_to_ix[START_TAG] = len(tag_to_ix)
    tag_to_ix[STOP_TAG] = len(tag_to_ix)
        #Add new categories : OOV, U_N, UP_C (out of vocabulary, only numeric characters, uppercase not in the first position)
    word_to_ix['OOV'] = len(word_to_ix)
    word_to_ix['U_N']=len(word_to_ix)
    word_to_ix['UP_C']=len(word_to_ix)

    with open('word_to_ix.pkl', 'wb') as f:
        pickle.dump(word_to_ix, f)
    
    with open('tag_to_ix.pkl', 'wb') as g:
        pickle.dump(tag_to_ix, g)

    training_data=[]
    for sentence in S.sentences: 
        list_words = [tup[0] for tup in sentence]
        list_tags = [tup[1] for tup in sentence]
        training_data.append((list_words, list_tags))

    validation_data=[]
    for sentence in V.sentences: 
        list_words = [tup[0] for tup in sentence]
        list_tags = [tup[1] for tup in sentence]
        validation_data.append((list_words, list_tags))


    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    model.to(device)

    NUM_EPOCHS=30
    

    # Early stopping parameters 
    last_loss= float('inf')
    trigger_times = 0
    patience = 10

    print('Start of the training')

    Train_Loss=[]
    Val_Loss=[]
    Epochs=[]
    for epoch in range(NUM_EPOCHS): 
        #Training Set 
        model.train()
        global_loss=[]
        for sentence, tags in training_data:

            model.zero_grad()
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long).to(device)
            loss = model.neg_log_likelihood(sentence_in, targets)

            loss.backward()
            optimizer.step()
            global_loss.append(loss.item())
        mean_loss = np.mean(np.array(global_loss))
        print("Epoch: %d, loss: %f" % (epoch, mean_loss))


        #Validation Set 
        model.eval()
        validation_loss=[]
        for sentence, tags in validation_data:
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long).to(device)
            loss = model.neg_log_likelihood(sentence_in, targets)
            validation_loss.append(loss.item())
        mean_val_loss = np.mean(np.array(validation_loss))
        print('Validation Set : loss = %f'%(mean_val_loss))

        #Early stopping 
        current_loss = mean_val_loss
        if current_loss > last_loss:
            trigger_times+=1
            if trigger_times >= patience : 
                print ('Early Stopping at Epoch n°', epoch)
                break
        else : 
            trigger_times = 0
        
        last_loss = current_loss

        # Save the checkpoint 

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': mean_loss,
            }, PATH_CHECKPOINT)
        
        print("-- Checkpoint saved --")

        Train_Loss.append(mean_loss)
        Val_Loss.append(mean_val_loss)
        Epochs.append(epoch+1)

    print('End of the training')

    ### Plot and save the training data ####

    plt.plot(Epochs, Train_Loss, label='Training Loss')
    plt.plot(Epochs, Val_Loss, label='Validation Loss')
    plt.legend()
    plt.savefig('Plot_training_data.png')


    #### Save the model ####

    torch.save(model.state_dict(), 'vst229547_model')

else : 

    #### Inference mode ####

    #Import files and model

    with open('word_to_ix.pkl', 'rb') as f:
        word_to_ix = pickle.load(f)

    with open('tag_to_ix.pkl', 'rb') as g:
        tag_to_ix = pickle.load(g)
    
    ix_to_tag={}
    for key, value in tag_to_ix.items():
        ix_to_tag[value] = key


    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    if os.path.isfile('vst229547_model') : 
        model.load_state_dict(torch.load('vst229547_model'))
    elif os.path.isfile(PATH_CHECKPOINT) : 
        print('Use the last check point')
        checkpoint = torch.load('checkpoints.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()


    #Open and read the data 

    with open(args.testing_file, 'r') as f : 
        lines = f.readlines()

    sentences = []
    s=[]
    for line in lines  :
        if line=='\n' : 
            sentences.append(s)
            s=[]
        else : 
            s.append(line.split('\n')[0])

    #Inference 
    Tags = []
    for sentence in sentences : 
        sentence_in = prepare_sequence(sentence, word_to_ix)
        score, tag = model(sentence_in)
        Tags.append(tag)

    Output=[]
    for sentence in Tags : 
        outs=[]
        for id in sentence : 
            outs.append(ix_to_tag[id])
        Output.append(outs)
    with open(args.output_file, 'w') as f:
        for sentence in Output :
            for tag in sentence : 
                f.write(tag+'\n')
            f.write('\n')
    



        



