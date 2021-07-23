# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import csv
import torch
import numpy as np
import pandas as pd
from Bio import SeqIO
import tensorflow as tf
from traceback import print_tb
from tensorflow.keras import activations
from tensorflow.keras import models, layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.engine.base_layer import Layer


# %%
fasta_train = 'H_train.fasta'
csv_train = 'H_train.csv'
fasta_test = 'H_test.fasta'
csv_test = 'H_test.csv'


## Add label to the train dataset and generate X_train=record.seq and Y_train=label
size_train = 0
train_lst = []
train_samples = []
train_labels = []


letters = 'ACGT'
emb_dict = {letter: number + 1 for number, letter in
            enumerate(letters)}  # number+1 for emb bec

        
with open(csv_train, 'w') as f:
    writer = csv.writer(f)
    for record in SeqIO.parse("../input/lnrna-ncrna-test-and-train-dataset/H_train.fasta", "fasta"):
        label_train = 0 if 'CDS' in record.id else 1
        #print(label_train)
        tarin_sample = []
        writer.writerow([record.id, record.seq, len(record), label_train])
        size_train = size_train+1
        lst=[record.id, str(record.seq), len(record), label_train]  
        #print(lst)
        for index, letter, in enumerate(record.seq):
            tarin_sample.append(emb_dict[letter])

        train_lst.append(lst)
        train_labels.append(label_train)
        train_samples.append(tarin_sample)

# padding
padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(
                train_samples, padding="post"
)     


        
## Convert list to Dataframe        
df_train = pd.DataFrame(train_lst,range(0,size_train),['ID','seq','length','Label'])
# df_train.shape
# df_train.dtypes
# df_train.head

### what are the max lengths of the train dataset?
# max_train = df_train['length'].max()
max_train = 3000



num_embeddings = 5
embedding_dim = 6

# em = torch.nn.Embedding(num_embeddings, embedding_dim).to(device)


padded_inputs, train_labels = np.array(padded_inputs), np.array(train_labels)
print(padded_inputs.shape)
print(train_labels.shape)

length_of_one_rna = max_train

# %% [markdown]
# 
# %% [markdown]
# test
# 

# %%

## Add label to the test dataset and generate X_test=record.seq and Y_test=label
size_test = 0 
test_lst = []
test_samples = []
test_labels = []


with open(csv_test, 'w') as t:
    writer = csv.writer(t)
    for record in SeqIO.parse('../input/lnrna-ncrna-test-and-train-dataset/H_test.fasta', 'fasta'):
        label_test = 0 if 'CDS' in record.id else 1
        writer.writerow([record.id, record.seq, len(record), label_test])
        size_test = size_test+1
        lst1=[record.id, str(record.seq), len(record), label_test]  
        test_lst.append(lst1)
        test_samples.append(str(record.seq))
        test_labels.append(label_test)


## Convert list to Dataframe       
df_test = pd.DataFrame(test_lst,range(0,7000),['ID','seq','length','Label'])
# print(df_test.shape, df_test.dtypes, df_test.head)


### what are the max lengths of the test dataset?
max_test = df_test['length'].max()
# print('the max lengths of the test dataset is {}'.format(max_test))


# %%
input_layer = layers.Input(shape=(max_train,))


# embedded_input = layers.Embedding(input_dim=1, output_dim=6, mask_zero=True)(input_layer)
## input_dim: Integer. Size of the vocabulary,
## input shape: (batch_size, input_length)
## output shape: 3D tensor with shape: (batch_size, input_length, output_dim)

key0 = layers.Dense(length_of_one_rna, name='key_layer')(input_layer)
query0 = layers.Dense(length_of_one_rna, name='query_layer')(input_layer)
values0 = layers.Dense(length_of_one_rna, name='values_layer')(input_layer)

attention_kernel0 = layers.Attention()([key0, query0])
attention_kernel_normilized0 = layers.Softmax()(attention_kernel0)

attention_output0 = layers.Attention()([attention_kernel_normilized0, values0])
attention_output0 = layers.Dense(length_of_one_rna, name='normilize_dims_layer')(attention_output0)

key1 = layers.Dense(length_of_one_rna)(input_layer)
query1 = layers.Dense(length_of_one_rna)(input_layer)
values1 = layers.Dense(length_of_one_rna)(input_layer)

attention_kernel1 = layers.Attention()([key1, query1])
attention_kernel_normilized1 = layers.Softmax()(attention_kernel1)

attention_output1 = layers.Attention()([attention_kernel_normilized1, values1])
attention_output1 = layers.Dense(length_of_one_rna)(attention_output1)

key2 = layers.Dense(length_of_one_rna)(input_layer)
query2 = layers.Dense(length_of_one_rna)(input_layer)
values2 = layers.Dense(length_of_one_rna)(input_layer)

attention_kernel2 = layers.Attention()([key2, query2])
attention_kernel_normilized2 = layers.Softmax()(attention_kernel1)

attention_output2 = layers.Attention()([attention_kernel_normilized2, values2])
attention_output2 = layers.Dense(length_of_one_rna)(attention_output2)

concated_heads = layers.Concatenate()([attention_output0, attention_output1, attention_output2])

attention_output = layers.Dense(length_of_one_rna)(concated_heads)

RNNS_input = layers.Reshape((3000, -1))(attention_output)

# Conv1d = layers.Conv1D(32, 3)(RNNS_input)
# Conv1d = layers.Conv1D(64, 3)(Conv1d)
# Conv1d = layers.Conv1D(128, 3)(Conv1d)

GRU_layer = layers.GRU(100,return_sequences=True, recurrent_dropout=0.5)(RNNS_input)
hidden = layers.Dense(100, activation='tanh')(GRU_layer)
cf = layers.Dense(1, activation='sigmoid')(hidden)


# %%
adam = tf.keras.optimizers.Adam(
    learning_rate=0.0004,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam")
sgd = tf.keras.optimizers.SGD(lr=0.1, momentum=0.8, nesterov=True)

classifier = models.Model(input_layer, cf)

classifier.compile(loss='binary_crossentropy',
                   optimizer= adam,       
                   metrics=['accuracy'])


# %%
classifier.summary()


# %%
plot_model(classifier)


# %%
classifier.fit(padded_inputs, train_labels, batch_size=1, epochs=10)


# %%
classifier.save('./epoch_10_attention.h5')


# %%



