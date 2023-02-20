#!/usr/bin/env python
# coding: utf-8

# In[29]:


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
from transformers import RobertaTokenizer, BertTokenizerFast
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split

 

# In[30]:

basePath = '/home/himesh/TagCoder/pythonNotebook'
tokenizerInPath = basePath + '/tokenizerIn'
tokenizerOutPath = basePath + '/tokenizerOut'
#basePath = r'C:\Users\Himesh\Documents\thesis\pythonNotebook'
positivePathSuffix = '/Positive'
negativePathSuffix = '/Negative'
#tokenizerInPath = basePath + '\\tokenizerIn'
#tokenizerOutPath = basePath + '\\tokenizerOut'
train_ratio = 0.7

# In[31]:


#tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertModel.from_pretrained("bert-base-cased")


# In[33]:


tokenized_methods = []
y_pos = []
tokenized_methods_neg = []
smellList = ['ComplexMethod']
final_text = ""
print(tokenizerInPath)
for smell in smellList:
    smellPath = os.path.join(tokenizerInPath, smell,'Positive',"")
    #print(smellPath)
    
    for file in os.listdir(smellPath):
        #print(os.path.basename(file))
        with open(os.path.join(smellPath, file),"r",encoding="utf8") as read_file:
           
            text = read_file.read()
            tokenized_method = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt")
            tokenized_methods.append(tokenized_method)
            y_pos.append(1)
    #Path(os.path.join(tokenizerOutPath,smell,positivePathSuffix, 'tokenizer.tok')).touch(exist_ok=True)        
    # with open(os.path.abspath(os.path.join(tokenizerOutPath,smell,'Positive', 'tokenizer.tok')),'w',errors='ignore') as out_file:
    #     #out_file.touch(exist_ok=True)
    #     #print(final_text)
    #     out_file.write(final_text)
    
smellPath = os.path.join(tokenizerInPath, smell,'Negative',"")
print(smellPath)

for file in os.listdir(smellPath):
    #print(os.path.basename(file))
    with open(os.path.join(smellPath, file),"r",encoding="utf8") as read_file:
        text = read_file.read()
        tokenized_method = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt")
        tokenized_methods_neg.append(tokenized_method)
        y_pos.append(0)
#Path(os.path.join(tokenizerOutPath,smell,positivePathSuffix, 'tokenizer.tok')).touch(exist_ok=True)        
# with open(os.path.abspath(os.path.join(tokenizerOutPath,smell,'Negative', 'tokenizer.tok')),'w',errors='ignore') as out_file:
#     #out_file.touch(exist_ok=True)
#     #print(final_text)
#     out_file.write(final_text)


# In[28]:


tokenized_methods = tokenized_methods[:250]
tokenized_methods_neg = tokenized_methods_neg[:250]
encoded_methods = []

for tokenized_method in tokenized_methods:
    encoded_method = model(**tokenized_method).last_hidden_state
    encoded_methods.append(encoded_method)

for tokenized_method in tokenized_methods_neg:
    encoded_method = model(**tokenized_method).last_hidden_state
    encoded_methods.append(encoded_method)


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(tokenized_methods, y_pos, test_size=0.2, random_state=42)

# In[25]:


input_shape = encoded_methods[0].shape[1:]

# Define the input layer
inputs = Input(shape=input_shape)

# Define the LSTM layers
lstm1 = LSTM(64, return_sequences=True)(inputs)
lstm2 = LSTM(64, return_sequences=True)(lstm1)
lstm3 = LSTM(64, return_sequences=True)(lstm2)

# Define the output layer
output = TimeDistributed(Dense(1, activation='sigmoid'))(lstm3)

# Define the model
model = Model(inputs=inputs, outputs=output)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32)

# In[ ]:


loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# In[ ]:




# In[ ]:


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
    
    


# In[ ]:


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



