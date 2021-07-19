'''
train.py published by Maryam on Jun 2021 includes train & test function
NOte: hyper-parameters of the model should be set in this script, then the model is being called
'''
import torch
import torch.nn as nn
import mymodels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# embedding
num_embeddings = 5
embedding_dim = 6

em = torch.nn.Embedding(num_embeddings, embedding_dim).to(device)


# Hyper-parameters
input_size = embedding_dim
batch_size = 125
learning_rate = 0.01
hidden_size = 100
num_classes = 2
num_layers = 3


model = mymodels.AttnRNN(input_size, 4, 1, 0.5).to(device)


criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
params = list(model.parameters())
print(len(params))
print(params)
print(params[0].size())


def train_model(sample, target):
    input_tensor = em(sample)  # input_tensor: (seq, batch_size, input_size)
    input_tensor = torch.reshape(input_tensor, (batch_size, sample.size(
        0), input_size)).to(device)   # x: (batch_size, seq, input_size),
    prediction = model(input_tensor)
    #print(f'prediction: {prediction}')
    loss = criterion(prediction, target)
    guess = torch.argmax(prediction, dim=1)
    #print(f'guess: {guess}, target: {target}')
    optimizer.zero_grad()  # Zero the gradients while training the network
    loss.backward()  # compute gradients
    optimizer.step()  # updates the parameters

    return guess, loss.item()


def test_model(sample, target):
    input_tensor = em(sample)
    input_tensor = torch.reshape(
        input_tensor, (batch_size, sample.size(0), input_size)).to(device)
    prediction = model(input_tensor)
    loss_train = criterion(prediction, target)
    guess = torch.argmax(prediction, dim=1)

    return guess, loss_train.item()
