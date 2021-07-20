# purpose: distinguish lncRNA from the cRNA
# 1. Design a model
# 2. Construct loss & optimizer
# 3. Training loop:
# - forward pass (call model to predict)
# - backward pass (calculate autograd)
# - update weights

import numpy as np
import torch
from Bio import SeqIO
from torch import utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import train
from torchsummary import summary

# We move our tensor to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# preparing dataset (generate integer_seq / batching & padding / embedding)
H_train = '/home/amirahmadipg/Documents/Programs/RNA/MasterThesis/scripts/H_train.fasta'
H_test = '/home/amirahmadipg/Documents/Programs/RNA/MasterThesis/scripts/H_test.fasta'
# define a dic
letters = 'ACGT'
emb_dict = {letter: number + 1 for number, letter in
            enumerate(letters)}  # number+1 for emb because the padded_input_tensor is zero


# padding
def collate_seqs(integerized_samples):
    batch_size = len(integerized_samples)
    integerized_seqs = [s[0] for s in integerized_samples]
    maxlen = max([len(seq) for seq in integerized_seqs])
    print(f'\n\n\n\n\n\n\n\n{maxlen}\n\n\n\n\n\n\n\n')
    padded_input_tensor = torch.zeros((maxlen, batch_size), dtype=torch.long)
    for i, s in enumerate(integerized_seqs):
        for j, v in enumerate(s):
            padded_input_tensor[j, i] = v
    label = torch.tensor([s[1] for s in integerized_samples])
    return padded_input_tensor, label


# An abstract class representing a Dataset.
class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, file_name):  # loading targets and integerized data
        self.samples = []
        self.targets = []
        with open(file_name)as fn:
            for record in SeqIO.parse(fn, 'fasta'):
                label_train = 0 if 'CDS' in record.id else 1
                y = torch.tensor(label_train, dtype=torch.long)
                self.targets.append(y)
                integerized_seq = []

                for index, letter, in enumerate(record.seq):
                    integerized_seq.append(emb_dict[letter])
                x = torch.tensor(integerized_seq, dtype=torch.long)
                self.samples.append(x)

    def __getitem__(self, idx):  # indexing
        return self.samples[idx], self.targets[idx]

    def __len__(self):
        return len(self.samples)


# create dataset
ds_train = ClassificationDataset(H_train)
ds_test = ClassificationDataset(H_test)

# Load whole dataset with DataLoader
# shuffle: shuffle data, good for training
dl_train = torch.utils.data.DataLoader(
    ds_train, collate_fn=collate_seqs, batch_size=train.batch_size, shuffle=True)
dl_test = torch.utils.data.DataLoader(
    ds_test, collate_fn=collate_seqs, batch_size=train.batch_size, shuffle=True)

# embedding
num_embeddings = 5
embedding_dim = 6

em = torch.nn.Embedding(num_embeddings, embedding_dim).to(device)

# Train the model
loss = 0
avg_loss = 0
all_losses = []
mean_losses = []
num_epochs = 20
n_total_steps = len(dl_train)
for epoch in range(num_epochs):
    for batch in tqdm(dl_train):
        n_samples = 0
        n_correct = 0
        sample, target = batch
        sample = sample.cuda()
        target = target.cuda()
        guess, loss = train.train_model(sample, target)
        all_losses.append(loss)

    avg_loss = sum(all_losses) / len(all_losses)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
#     mean_losses.append(avg_loss)

from tensorflow.keras import models, layers



n_samples = 0
n_correct = 0
sample, target = dl_train[0]

# save & load model
# save entire model
FIle = 'model_params'
torch.save(train.model, FIle)

# # load entire model
loaded_model = torch.load(FIle)
loaded_model.eval()

# # Test the model
loss_test = 0
avg_loss_test = 0
all_losses_test = []
mean_losses_test = []
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for batch in dl_test:
        sample, target = batch
        sample = sample.cuda()
        target = target.cuda()
        guess, loss_test = train.test_model(sample, target)
        all_losses_test.append(loss_test)

        n_samples += target.size(0)
        n_correct += (guess == target).sum().item()

    acc_test = 100.0 * n_correct / n_samples
    print(
        f'Accuracy of the network on the 7000 test RNA sequences: {acc_test:.4f} %')

print(all_losses)
print(len(all_losses))

x_train = np.arange(0, num_epochs)
plt.figure()
plt.plot(x_train, mean_losses, color='tab:blue', marker='o')

x_test = np.arange(0, len(dl_test))
plt.plot(x_test, all_losses_test, color='tab:orange', linestyle='--')
plt.show()
