# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

import numpy as np
from tensorflow.python.keras.engine.base_layer import Layer
import torch
from Bio import SeqIO
# import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from tensorflow.keras import activations
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import torch
import csv
from Bio import SeqIO
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

tensorboard = TensorBoard(
  log_dir='./logs',
  histogram_freq=1,
  write_images=True
)
keras_callbacks = [
  tensorboard
]


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
    for record in SeqIO.parse("/home/amirahmadipg/Documents/Programs/RNA/MasterThesis/scripts/H_test.fasta", "fasta"):
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

# 
# test
# 



## Add label to the test dataset and generate X_test=record.seq and Y_test=label
size_test = 0 
test_lst = []
test_samples = []
test_labels = []


with open(csv_test, 'w') as t:
    writer = csv.writer(t)
    for record in SeqIO.parse('/home/amirahmadipg/Documents/Programs/RNA/MasterThesis/scripts/H_test.fasta', 'fasta'):
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




def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(x):
    position = x[0]
    d_model = x[1]
    
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return pos_encoding



input_layer1 = layers.Input(shape=(max_train,))

embedding_layer = tf.keras.layers.Embedding(input_dim=5, output_dim=6, input_length=3000)(input_layer1)

positional_embedding = layers.Lambda(positional_encoding)([3000, 6])

add_embeddings = layers.Add()([embedding_layer, positional_embedding])

flatt_output = layers.Flatten()(add_embeddings)

length_of_one_rna = 18000 # output_dim*input_length

query0 = layers.Dense(length_of_one_rna, name='query_layer')(flatt_output)

values0 = layers.Dense(length_of_one_rna, name='values_layer')(flatt_output)

attention_kernel0 = layers.Attention()([query0, values0])


RNNS_input = layers.Reshape((3000, -1))(attention_kernel0)

# Add & Norm
# Add = tf.keras.layers.Add()([input_layer, RNNS_input])

# feed_forward
hidden = layers.Dense(10, activation='relu')(RNNS_input)
hidden = layers.Dense(15, activation='relu')(hidden)


GRU_layer = layers.GRU(2, return_sequences=True, recurrent_dropout=0.05)(hidden)
hidden = layers.Dense(2, activation='relu')(GRU_layer)
cf = layers.Dense(1, activation='sigmoid')(hidden)

adam = tf.keras.optimizers.Adam(
    learning_rate=0.009,
    beta_1=0.6,
    beta_2=0.6,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam")

classifier = models.Model(input_layer1, cf)

classifier.compile(loss='binary_crossentropy',
                   optimizer= adam,       
                   metrics=['accuracy'])


        # regressor.fit(sample, target, epochs=3, batch_size=32)
classifier.fit(padded_inputs, train_labels, epochs=5, batch_size=1, callbacks=keras_callbacks)



