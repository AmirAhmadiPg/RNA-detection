'''
mymodels.py published by Maryam on Jun 2021 includes networks used for training data
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Fully connected neural network
class basicRNN(nn.Module):
    # nn.RNN
    def __init__(self, input_size, hidden_size, num_classes):
        super(basicRNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, num_classes)
        #self.softmax = nn.LogSoftmax(dim=1)
        # self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)
        hidden = self.i2h(combined)
        self.relu = nn.ReLU
        output = self.i2o(combined)
        self.relu = nn.ReLU
        #output = self.softmax(output)
        # output = self.sigmoid(output)
        return output, hidden


# Fully connected neural network with one hidden layer
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # -> x needs to be: (batch_size, seq, input_size)
        self.RNN = nn.RNN(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(device)
        # x: (batch_size, seq, input_size), h0: (num_layers, batch_size, 128)

        # Forward propagate RNN
        # out: tensor of shape (batch_size, seq_length, hidden_size) containing the output features (h_t) from the last layer of the RNN, for each t
        out, _ = self.RNN(x, h0)
        # out: (batch_size, seq, 128)

        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (batch_size, hidden_size)

        out = self.fc(out)
        # out: (batch_size, num_classes)
        return out


class AttnRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout_p=0.1, max_length= 2702):
        super(AttnRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input):
        hidden = torch.zeros(self.num_layers, input.size(0),
                         self.hidden_size).to(device)
        
        embedded = self.dropout(input)

        attn_weights = F.softmax(
            self.attn(input))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 input.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)