import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, input_size, output_size, cfg, type):
        super(SimpleMLP, self).__init__()
        if type == "actor":
            hidden_sizes = cfg["actor_hidden_dims"]
        elif type == "critic":
            hidden_sizes = cfg["critic_hidden_dims"]
        else:
            raise

        self.layers = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


class NormalizedSimpleMLP(SimpleMLP):
    def __init__(self, input_size, output_size, cfg, type):
        super(NormalizedSimpleMLP, self).__init__(input_size, output_size, cfg, type)
        self.layers.append(nn.Tanh())

        self.model = nn.Sequential(*self.layers)


class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, cfg, type):
        super(SimpleLSTM, self).__init__()
        if type == "actor":
            hidden_dim = cfg["actor_hidden_dims"]
            num_layers = cfg["actor_num_layers"]
            input_dim = 100
        elif type == "critic":
            hidden_dim = cfg["critic_hidden_dims"]
            num_layers = cfg["critic_num_layers"]
            input_dim = 103
        else:
            raise

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x, None)
        out = self.fc(out[:, -1, :])
        return out


class SimpleGPT(nn.Module):
    def __init__(self):
        super(SimpleGPT, self).__init__()

    def forward(self, x):
        pass