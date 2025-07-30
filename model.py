# build and train everything with the neural network
import torch 
import torch.nn as nn
import torch.optim as optim

class PredictModel(nn.Module):

    # constructor
    # input_dim: Number of features at each time step.
    # hidden_dim: Number of features in the LSTM's hidden state.
    # num_layers: Number of stacked LSTM layers.
    # output_dim: Number of output features (e.g., 1 for predicting a single value).
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        # call the parent constructor so PyTorch can manage your model correctly.
        super(PredictModel, self).__init__()

        # the LSTM parameters to use later when creating the hidden state.
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # initialize an LSTM layer.
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True) # batch_first=True means input shape should be (batch_size, seq_len, input_dim).
        self.fc = nn.Linear(hidden_dim, output_dim) # Adds a fully connected layer that maps from LSTM's hidden output to your desired output size (e.g., one prediction per sequence).

    # this function defines how the model processes input x and generates output (i.e., prediction).
    def forward(self, x):
        # initialize the hidden state (h0) and cell state (c0) with zeros.
        # Shape explanation: (num_layers, batch_size, hidden_dim)
        # device=device puts the tensors on GPU if available.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)

        # feed input x into the LSTM.
        # (h0, c0) are the initial states. Using .detach() avoids tracking gradients through them.
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach())) # hn, cn are the final hidden and cell states.
        out = self.fc(out[:, -1, :]) # contains the LSTM output at each time step.

        return out