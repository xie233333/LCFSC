import warnings
warnings.simplefilter('default')
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

class GatedLinearUnit(nn.Module):
    """**The unit of gating operation that maps the input to the range of 0-1 and multiple original input through the
    sigmoid function.**

    """

    def __init__(self, input_size,
                 hidden_layer_size,
                 dropout_rate):
        """

        :param input_size: Number of features
        :param hidden_layer_size: The size of nn.Linear layer, global default is 160
        :param dropout_rate: The rate of linear layer parameters randomly discarded during training
        :param activation: activation function used to activate raw input, default is None
        """
        super(GatedLinearUnit, self).__init__()

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate

        if self.dropout_rate:
            self.dropout = nn.Dropout(p=self.dropout_rate)

        self.W4 = torch.nn.Linear(self.input_size, self.hidden_layer_size)
        self.W5 = torch.nn.Linear(self.input_size, self.hidden_layer_size)


        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if 'bias' not in n:
                torch.nn.init.xavier_uniform_(p)
            elif 'bias' in n:
                torch.nn.init.zeros_(p)

    def forward(self, x):
        output = self.sigmoid(self.W4(x)) * self.W5(x)

        return output


class GateAddNormNetwork(nn.Module):
    """**Units that adding gating output to skip connection improves generalization.**"""

    def __init__(self, input_size,
                 hidden_layer_size,
                 dropout_rate):
        """

        :param input_size: Number of features
        :param hidden_layer_size: The size of nn.Linear layer, global default is 160
        :param dropout_rate: The rate of linear layer parameters randomly discarded during training
        :param activation: activation function used to activate raw input, default is None
        """
        super(GateAddNormNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate

        self.GLU = GatedLinearUnit(self.input_size,
                                   self.hidden_layer_size,
                                   self.dropout_rate).cuda()

        self.LayerNorm = nn.LayerNorm(self.hidden_layer_size)

    def forward(self, x, skip):
        output = self.LayerNorm(self.GLU(x) + skip)

        return output


class GatedResidualNetwork(nn.Module):
    """**GRN main module, which divides all inputs into two ways, calculates the gating one way for linear mapping twice and
    passes the original input to GateAddNormNetwork together. ** """
    def __init__(self,
                 hidden_layer_size,
                 input_size=None,
                 output_size=None,
                 dropout_rate=None):
        """

        :param hidden_layer_size: The size of nn.Linear layer
        :param input_size: Number of features
        :param output_size: Number of features
        :param dropout_rate: The rate of linear layer parameters randomly discarded during training
        """
        super(GatedResidualNetwork, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size if input_size else self.hidden_layer_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        self.W1 = torch.nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.W2 = torch.nn.Linear(self.input_size, self.hidden_layer_size)

        if self.output_size:
            self.skip_linear = torch.nn.Linear(self.input_size, self.output_size)
            self.glu_add_norm = GateAddNormNetwork(self.hidden_layer_size,
                                                   self.output_size,
                                                   self.dropout_rate).cuda()
        else:
            self.glu_add_norm = GateAddNormNetwork(self.hidden_layer_size,
                                                   self.hidden_layer_size,
                                                   self.dropout_rate).cuda()

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if ('W2' in name or 'W3' in name) and 'bias' not in name:
                torch.nn.init.kaiming_normal_(p, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif ('skip_linear' in name or 'W1' in name) and 'bias' not in name:
                torch.nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                torch.nn.init.zeros_(p)

    def forward(self, x):
        n2 = F.elu(self.W2(x))
        n1 = self.W1(n2)

        if self.output_size:
            output = self.glu_add_norm(n1, self.skip_linear(x))
        else:
            output = self.glu_add_norm(n1, x)

        return output


class VariableSelectionNetwork(nn.Module):
    """**Feature selection module, which inputs a vector stitched into all features, takes the weights of each
    feature and multiply with the original input as output. ** """
    def __init__(self, hidden_layer_size,
                 dropout_rate,
                 output_size,
                 input_size):
        """

        :param hidden_layer_size: The size of nn.Linear layer, global default is 160
        :param dropout_rate: The rate of linear layer parameters randomly discarded during training
        :param output_size: Number of features
        :param input_size: Number of features
        """
        super(VariableSelectionNetwork, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.sigmoid = nn.Sigmoid()

        self.flattened_grn = GatedResidualNetwork(self.hidden_layer_size,
                                                  input_size=self.input_size,
                                                  output_size=self.output_size,
                                                  dropout_rate=self.dropout_rate, ).cuda()

    def forward(self, x, B_):
        embedding = x.view(B_, -1)

        flatten = torch.flatten(embedding, start_dim=1)
        mlp_outputs = self.flattened_grn(flatten)
        sparse_weights = F.softmax(mlp_outputs, dim=-1).mean(-2)
        return sparse_weights#combined, sparse_weights


class SimpleMLP(nn.Module):
    """**The module where the main model is defined. The model consists of GRN and a single layer neural network. The
    input discrete features are embedding and real valued features to the GRN module, and then obtains the feature
    weight and multiply the output to the single output through the single layer neural network, and then the loss is
    calculated with target. ** """
    def __init__(self, input_resolution, dim):
        """

        :param cat_num_classes: Number of category features
        :param real_num: Number of real valued features
        """
        super().__init__()
        self.input_size = input_resolution[0]*input_resolution[1]*dim # feature dimension
        self.lin_drop = nn.Dropout(0.25)
        self.sparse_weight = None
        self.output_size = input_resolution[0]*input_resolution[1]
        self.select_model = VariableSelectionNetwork(hidden_layer_size=self.input_size//16,
                                                input_size=self.input_size,
                                                output_size=self.output_size,
                                                dropout_rate=0.1).cuda() # give weights for each feature patch
        
    def forward(self, inputs, B_):
        x = inputs
        self.sparse_weight = self.select_model(x, B_)

        return self.sparse_weight


