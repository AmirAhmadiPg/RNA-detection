import os
import torch
import torch.nn as nn
from Bio import SeqIO
#from torch.utils.data import Dataset

# We move our tensor to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#print(os.getcwd())
#file = os.environ.get['DATA_PATH']#+'/H_train.fasta'
file = open('H_train.fasta')

letters = 'ACGT'
emb_dict = {letter: number+1 for number, letter in enumerate(letters)} #number+1 for emb because the padded_input_tensor is zero
#with open('H_test.fasta', 'r') as file:
#for record in SeqIO.parse(file, 'fasta'):
    #print(len(record))
#     integerized_seq = []
#     for index, letter, in enumerate(record.seq):
#         integerized_seq.append(emb_dict[letter])
#     x = torch.tensor(integerized_seq, dtype=torch.int)


def collate_seqs(integerized_samples): #padding
    batch_size = len(integerized_samples)
    integerized_seqs = [s[0] for s in integerized_samples]
    #print(integerized_seqs)
    maxlen = max([len(seq) for seq in integerized_seqs])
    padded_input_tensor = torch.zeros((maxlen, batch_size), dtype=torch.long)
    for i, s in enumerate(integerized_seqs):
        for j, v in enumerate(s):
            padded_input_tensor[j, i] = v
    label = torch.tensor([s[1] for s in integerized_samples])
    return padded_input_tensor, label

# --> DataLoader can do the batch computation for us
# Implement a custom Dataset:(integerized)
# inherit Dataset
# implement __init__ , __getitem__ , and __len__

class ClassificationDataset(torch.utils.data.Dataset): #An abstract class representing a Dataset.
    def __init__(self, file): #loading targets and integerized data
        self.samples = []
        self.targets = []
        for record in SeqIO.parse(file, 'fasta'):
            label_train = 0 if 'CDS' in record.id else 1
            y = torch.tensor(label_train, dtype=torch.int)
            self.targets.append(y)
            integerized_seq = []

            for index, letter, in enumerate(record.seq):
                integerized_seq.append(emb_dict[letter])
            x = torch.tensor(integerized_seq, dtype=torch.long)
            self.samples.append(x)

    def __getitem__(self, idx): #indexing
        # TODO transforms
        return self.samples[idx], self.targets[idx]

    def __len__(self):
        return len(self.samples)

# create dataset
ds = ClassificationDataset(file)

# get first sample and unpack
first_ds = ds[0]
features, label = first_ds
#print(features, label)


# Load whole dataset with DataLoader
# shuffle: shuffle data, good for training
dl = torch.utils.data.DataLoader(ds, collate_fn=collate_seqs, batch_size=4, shuffle=True)

# convert to an iterator and look at one random sample
dataiter = iter(dl)
data = dataiter.next()
features, labels = data
#print(features.shape, labels.shape)
#print(features, labels)

'''
# Hyper-parameters
input_size = maxlen
num_classes = 2
num_epochs = 2
batch_size = 4
learning_rate = 0.001

input_size =
#sequence_length = varied
hidden_size = 128
num_layers = 2

# Design model
model = nn.RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
'''

em = torch.nn.Embedding(5, 4, 0) # embed batch -> [L, B, H] 5:ACGT0, 4:batch_size, 0:padded-idx

for batch in dl:
    #print(batch[0])
    print(batch[0].size())
    #print(batch[0].max())
    print(em(batch[0]).size())
    #print(em(batch[0]))
    #print(batch)
    #print(batch[0].dtype)
    #print(f"Device tensor is stored on: {batch[0].device}")

    raise



    # TODO training goes here

'''
prediction = model(batch)
loss = crit(prediction, target)
'''
