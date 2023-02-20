#!/usr/bin/env python
# coding: utf-8

# In[80]:


import os
import transformers
from pathlib import Path
import torch
import numpy as np
from math import ceil
from random import shuffle
from itertools import chain
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.models import Model
import tensorflow.keras.optimizers as opt
import pandas as pd
from transformers import RobertaTokenizer

 

# In[81]:


#basePath = '/home/himesh/TagCoder/pythonNotebook'
basePath = r'C:\Users\Himesh\Documents\thesis\pythonNotebook'
positivePathSuffix = '/Positive'
negativePathSuffix = '/Negative'
tokenizerInPath = basePath + '\\tokenizerIn'
tokenizerOutPath = basePath + '\\tokenizerOut'
train_ratio = 0.7

# In[83]:


#tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')

# In[85]:


smellList = ['ComplexMethod']
final_text = ""
print(tokenizerInPath)
for smell in smellList:
    smellPath = os.path.join(tokenizerInPath, smell,'Positive',"")
    #print(smellPath)
    
    for file in os.listdir(smellPath):
        #print(os.path.basename(file))
        with open(os.path.join(smellPath, file),"r") as read_file:
            try:
                text = read_file.read()
                tokenized_text = tokenizer.tokenize(text)#,padding = "max_length")
                input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
                modint = (len(input_ids)) % 512
                #print(modint)
                length = len(input_ids) - modint
            
                input_ids = input_ids[0:length]
                final_text += ' '.join(map(str, input_ids))+' '
            except Exception as e:
                print(e)
                pass
    #Path(os.path.join(tokenizerOutPath,smell,positivePathSuffix, 'tokenizer.tok')).touch(exist_ok=True)        
    with open(os.path.abspath(os.path.join(tokenizerOutPath,smell,'Positive', 'tokenizer.tok')),'w',errors='ignore') as out_file:
        #out_file.touch(exist_ok=True)
        #print(final_text)
        out_file.write(final_text)
    
smellPath = os.path.join(tokenizerInPath, smell,'Negative',"")
print(smellPath)

for file in os.listdir(smellPath):
    #print(os.path.basename(file))
    with open(os.path.join(smellPath, file),"r") as read_file:
        try:
            text = read_file.read()
            tokenized_text = tokenizer.tokenize(text)#,padding = "max_length")
            input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
            modint = (len(input_ids)) % 512
            #print(modint)
            length = len(input_ids) - modint
           
            input_ids = input_ids[0:length]
           
            final_text += ' '.join(map(str, input_ids))+' '
        except Exception as e:
            print(e)
            
            pass
#Path(os.path.join(tokenizerOutPath,smell,positivePathSuffix, 'tokenizer.tok')).touch(exist_ok=True)        
with open(os.path.abspath(os.path.join(tokenizerOutPath,smell,'Negative', 'tokenizer.tok')),'w',errors='ignore') as out_file:
    #out_file.touch(exist_ok=True)
    #print(final_text)
    out_file.write(final_text)


# In[72]:


posInput = []
num_lines_pos = sum(1 for line in open(os.path.join(tokenizerOutPath,smell,'Positive', 'tokenizer.tok'),"r"))
with open(os.path.join(tokenizerOutPath,smell,'Positive', 'tokenizer.tok'),"r") as read_file:
    text = read_file.read()
  
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    #print(text)
    posInput = np.fromstring(text, sep=" ").tolist()
    print(len(posInput))
    # for line in read_file:
    #     if line == '\n':
    #         continue
    #     arr = np.fromstring(line, dtype=np.int32, sep=" ").tolist()
    #     posInput.append(arr)

negInput = []
num_lines_neg = sum(1 for line in open(os.path.join(tokenizerOutPath,smell,'Negative', 'tokenizer.tok'),"r"))

with open(os.path.join(tokenizerOutPath,smell,'Negative', 'tokenizer.tok'),"r") as read_file:
    text = read_file.read()
  
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    #print(text)
    negInput = np.fromstring(text, dtype=np.int32, sep=" ").tolist()

    # for line in read_file:
    #     if line == '\n':
    #         continue
    #     arr = np.fromstring(line, dtype=np.int32, sep=" ").tolist()
    #     negInput.append(arr)

num_lines_all =  num_lines_pos if num_lines_pos < num_lines_neg else num_lines_pos
 

# In[73]:


posInputLen = len(posInput)
negInputLen = len(negInput)
print(str(posInputLen)+"  "+str(negInputLen))
train_data = []
test_data = []
posSize = ceil(posInputLen*train_ratio) - ceil(posInputLen*train_ratio) % 512
print(posSize)
negSize = ceil(negInputLen*train_ratio) - ceil(negInputLen*train_ratio) % 512

test_data.append(posInput[posSize+1:])
test_data[0] = test_data[0][0:len(test_data[0]) - (len(test_data[0])%512)]
test_data.append(negInput[negSize+1:])
test_data[1] = test_data[1][0:len(test_data[1]) - (len(test_data[1])%512)]

test_data_flattened = list(chain.from_iterable(test_data))
test_data_np = np.array(test_data_flattened)

test_label = np.empty(shape=[len(test_data_np)], dtype=np.float32)
print(len(test_label))
test_label[0:posSize] = 1.0
test_label[posSize+1:] = 0.0

total_train_data = 0
train_data.append(posInput[0:posSize])
train_data[0] = train_data[0][0:len(train_data[0]) - (len(train_data[0])%512)]
total_train_data += len(train_data[0])
train_data.append(negInput[0:negSize])
train_data[1] = train_data[1][0:len(train_data[1]) - (len(train_data[1])%512)]
total_train_data += len(train_data[1])
print('total_train_data'+str(total_train_data))
train_data_flattened = list(chain.from_iterable(train_data))
#shuffle(train_data_flattened)
#print(train_data[1])
train_data_np = np.array(train_data_flattened)
train_data_np = train_data_np.reshape(-1,512,1)
shuffle(train_data_np)
print('arr shape')
print(train_data_np.shape)
test_data_np = test_data_np.reshape(len(test_label),1)


# In[ ]:




# In[78]:


def autoencoder_lstm(train_data, test_data_np, smell, layers=1, encoding_dimension=8, no_of_epochs=10, with_bottleneck=True, is_final=False):
    
    encoding_dim = encoding_dimension
    input_layer = Input(shape=(512, 1))
    # input_layer = BatchNormalization()(input_layer)
    no_of_layers = layers
    prev_layer = input_layer
    
    for i in range(no_of_layers):
        encoder = LSTM(int(encoding_dim / pow(2, i)),
                        #activation="relu",
                       return_sequences=True,
                       recurrent_dropout=0.1,
                       dropout=0.1)(prev_layer)
        prev_layer = encoder 
    
    if with_bottleneck:
        prev_layer = LSTM(int(encoding_dim / pow(2, no_of_layers + 1)),
                         #activation="relu",
                          return_sequences=True,
                          recurrent_dropout=0.1,
                          dropout=0.1)(prev_layer)
    for j in range(no_of_layers - 1, -1, -1):
        decoder = LSTM(int(encoding_dim / pow(2, j)),
                        #activation='relu',
                       return_sequences=True,
                       recurrent_dropout=0.1,
                       dropout=0.1)(prev_layer)
        prev_layer = decoder
    prev_layer = TimeDistributed(Dense(1))(prev_layer)
    autoencoder = Model(inputs=input_layer, outputs=prev_layer)

    autoencoder.compile(optimizer='adam',
                        loss='mean_squared_error',
                        metrics=['accuracy'])
    autoencoder.summary()   


    batch_sizes = [32, 64]
    b_size = int(len(train_data) / 512)
    if b_size > len(batch_sizes) - 1:
        b_size = len(batch_sizes) - 1
    history = autoencoder.fit(train_data,
                              train_data,
                              epochs=no_of_epochs,
                              batch_size=batch_sizes[b_size],
                              verbose=1,
                              validation_split=0.2,
                              shuffle=True).history
    
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    
    predictions = autoencoder.predict(test_data_np)
    predictions = predictions.reshape(predictions.shape[0], predictions.shape[1])
    test_data_np = test_data_np.reshape(test_data_np.shape[0], test_data_np.shape[1])
    mse = np.mean(np.power(test_data_np - predictions, 2), axis=1)
    error_df = pd.DataFrame({'Reconstruction_error': mse,
                             'True_class': test_label})
    print(error_df.describe())
    
    


# In[79]:


layers = [1,2]
encoding_dim = [8, 16, 32]
epochs = 100
cur_iter = 1
skip_iter = 2
for layer in layers:
        for bottleneck in [True]:
            for encoding in encoding_dim:
                if cur_iter <= skip_iter:
                    cur_iter += 1
                    continue
                cur_iter += 1
                autoencoder_lstm(train_data_np,test_data_np, smell, layers=layer,encoding_dimension=encoding,no_of_epochs=epochs, with_bottleneck=bottleneck)

# In[ ]:



