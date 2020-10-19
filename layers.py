import torch
import numpy as np
from torch.nn import Parameter


class GraphConvolution(torch.nn.Module):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, dropout=0., sparse_inputs=False, act=torch.nn.functional.relu,
                 bias=False, featureless=False, norm=False, precalc=False, **kwargs):

        super(GraphConvolution, self).__init__(**kwargs)

        allowed_kwargs = {}
        for kwarg, _ in kwargs.items():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        self.vars = {}
        self.sparse_inputs = False

        self.dropout = dropout
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.norm = norm
        self.precalc = precalc

        # self.weight = Parameter(torch.Tensor(input_dim, output_dim))
        self.weight = torch.nn.Linear(input_dim, output_dim)
        if self.norm:
            self.bn = torch.nn.BatchNorm1d(output_dim)

    def forward(self, support, inputs):
        x = inputs

        # convolve
        if self.precalc:
            support = x
        else:
            support = dot(support, x, sparse=True)
            support = torch.cat((support, x), dim=1)

        # dropout
        support = torch.nn.functional.dropout(support, 1 - self.dropout)
        # output = dot(support, self.vars['weights'], sparse=self.sparse_inputs)
        # output = dot(support, self.weight, sparse=self.sparse_inputs)
        output = self.weight(support)
        if self.norm:
            res = self.act(self.bn(output))
            # res = torch.nn.functional.relu(self.bn(output))
            # output = layernorm(output, self.vars['offset'], self.vars['scale'])
        else:
            res = self.act(output)
            # res = torch.nn.functional.relu(output)

        # bias
        # if self.bias:
        #     output += self.vars['bias']

        return res


class StackedGCN(torch.nn.Module):
    """
    Multi-layer GCN model.
    """

    def __init__(self, args, input_channels, output_channels, **kwargs):
        """
        :param args: Arguments object.
        :input_channels: Number of features.
        :output_channels: Number of target features. 
        """
        super(StackedGCN, self).__init__()
        allowed_kwargs = {
            'name', 'logging', 'norm', 'precalc', 'num_layers'
        }
        for kwarg, _ in kwargs.items():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        self.layers = torch.nn.ModuleList()
        self.activations = None

        self.args = args
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.norm = kwargs.get('norm', False)
        self.precalc = kwargs.get('precalc', False)
        self.num_layers = kwargs.get('num_layers', 2)

        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layes based on the args.
        """
        self.layers.append(GraphConvolution(input_dim=self.input_channels if self.precalc else self.input_channels * 2,
                                            output_dim=self.args.hidden1,
                                            act=torch.nn.functional.relu,
                                            dropout=True,
                                            norm=self.norm,
                                            precalc=self.precalc))

        for _ in range(self.num_layers - 2):
            self.layers.append(GraphConvolution(input_dim=self.args.hidden1 * 2,
                                                output_dim=self.args.hidden1,
                                                act=torch.nn.functional.relu,
                                                dropout=True,
                                                norm=self.norm,
                                                precalc=False))

        self.layers.append(GraphConvolution(input_dim=self.args.hidden1 * 2,
                                            output_dim=self.output_channels,
                                            act=lambda x: x,
                                            dropout=True,
                                            norm=False,
                                            precalc=False))

        # self.layers = ListModule(*self.layers)

    def forward(self, support, features):
        # Build sequential layer model
        self.activations = []
        self.activations.append(features)
        for layer in self.layers:
            hidden = layer(support, self.activations[-1])
            self.activations.append(hidden)

        return self.activations[-1]


class GraphAttentionLayer(torch.nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, precalc=False):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.precalc = precalc

        self.W = torch.nn.Parameter(torch.zeros(size=(in_features, out_features)))
        torch.nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = torch.nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = torch.nn.LeakyReLU(self.alpha)

    def forward(self, features, support):
        x = features
        if self.precalc:
            support = x
        else:
            support = dot(support, x, sparse=True)
            support = torch.cat((support, x), dim=1)

        h = torch.mm(support, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(support > 0, e, zero_vec)
        attention = torch.nn.functional.softmax(attention, dim=1)
        attention = torch.nn.functional.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return torch.nn.functional.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(torch.nn.Module):
    """
    GAT Model
    """

    def __init__(self, args, input_channels, output_channels, **kwargs):
    # def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        allowed_kwargs = {
            'name', 'logging', 'multilabel', 'norm',
            'precalc', 'num_layers', 'dropout',
            'alpha', 'nheads'
        }
        for kwarg, _ in kwargs.items():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        self.layers = torch.nn.ModuleList()

        self.args = args
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.dropout = kwargs.get('dropout', 0.6)
        self.alpha = kwargs.get('alpha', 0.2)
        self.nheads = kwargs.get('nheads', 8)
        self.precalc = kwargs.get('precalc', False)
        self.num_layers = kwargs.get('num_layers', 2)

        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layes based on the args.
        """
        self.layers.append(GraphAttentionLayer(in_features=self.input_channels if self.precalc else self.input_channels * 2,
                                            out_features=self.args.hidden1,
                                            dropout=self.dropout,
                                            precalc=self.precalc,
                                               alpha=self.alpha,
                                               concat=True))

        for _ in range(self.num_layers - 2):
            self.layers.append(GraphAttentionLayer(in_features=self.args.hidden1 * 2,
                                                out_features=self.args.hidden1,
                                                dropout=self.dropout,
                                                precalc=False,
                                                   alpha=self.alpha,
                                                   concat=True))

        self.layers.append(GraphAttentionLayer(in_features=self.args.hidden1 * 2,
                                               out_features=self.output_channels,
                                               dropout=self.dropout,
                                               precalc=False,
                                               alpha=self.alpha,
                                               concat=False))
        # self.attentions = [GraphAttentionLayer(input_channels, self.args.hidden1, dropout=self.dropout,
        #                                        alpha=self.alpha, concat=True) for _ in range(self.nheads)]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)
        #
        # self.out_att = GraphAttentionLayer(self.args.hidden1 * self.nheads, output_channels, dropout=self.dropout, alpha=self.alpha, concat=False)

    def forward(self, support, features):
        # x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(features, support) for att in self.layers], dim=1)
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = torch.nn.functional.elu(self.out_att(features, support))
        return torch.nn.functional.log_softmax(x, dim=1)

    # def forward(self, x, adj):
        # x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        # x = torch.nn.functional.elu(self.out_att(x, adj))
        # return torch.nn.functional.log_softmax(x, dim=1)


class SimGraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, input_dim, output_dim, dropout=0., act=torch.nn.functional.relu, precalc=True):
        super(SimGraphConvolution, self).__init__()
        self.in_features = input_dim
        self.out_features = output_dim
        self.dropout = dropout
        self.act = act
        self.precalc = precalc
        # self.weight = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.weight = torch.nn.Linear(input_dim, output_dim)
        # self.reset_parameters()

    # def reset_parameters(self):
    #     torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, support, inputs):
        x = inputs
        hidden = torch.nn.functional.dropout(support, self.dropout)
        hidden = dot(hidden, x, sparse=False)
        output = self.weight(hidden)
        output = self.act(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNModelAE(torch.nn.Module):
    '''
    Variational Graph Auto-Encoders in Pytorch implementation
    https://github.com/zfjsail/gae-pytorch
    '''
    def __init__(self, input_channels, hidden_dim1, hidden_dim2, output_dim, dropout):
        super(GCNModelAE, self).__init__()
        self.input_channels = input_channels

        self.hidden1 = SimGraphConvolution(input_dim=self.input_channels,
                                       output_dim=hidden_dim1,
                                       act=torch.nn.functional.relu,
                                       dropout=dropout)

        self.embedding = SimGraphConvolution(input_dim=hidden_dim1,
                                       output_dim=hidden_dim2,
                                       dropout=dropout,
                                       act=lambda x: x)

        self.mlp = torch.nn.Linear(hidden_dim2, output_dim)

        self.reconstructions = InnerProductDecoder(dropout=dropout, act=lambda x: x)



    def forward(self, adj, x):
        hidden1 = self.hidden1(adj, x)
        embedding = self.embedding(adj, hidden1)
        reconstract = self.reconstructions(embedding)
        out = self.mlp(embedding)

        return embedding, reconstract, out


class InnerProductDecoder(torch.nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = torch.nn.functional.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


def dot(x, y, sparse=False):
    """Wrapper for torch.mm (sparse vs dense).
        torch.mm works for both sparse and dense in torch=0.4.1"""
    if sparse:
        res = torch.sparse.mm(x, y)
        # res = torch.mm(to_torch_sparse_tensor(x), y)
    else:
        res = torch.mm(x, y)
    return res

# def layernorm(x, offset, scale):
#     mean, variance = tf.nn.moments(x, axes=[1], keep_dims=True)
#     return tf.nn.batch_normalization(x, mean, variance, offset, scale, 1e-9)
#     mean = torch.mean(x)
#     variance = torch.std(x)
#     return torch.nn.functional.batch_norm(x)

##########

"""Implementations of different initialization methods."""


##########
def uniform(shape, scale=0.05):
    """Uniform init."""
    initial = torch.rand(shape, dtype=torch.float32) * scale - scale
    return torch.nn.Parameter(initial)


def glorot(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = torch.rand(shape, dtype=torch.float32) * init_range - init_range
    return torch.nn.Parameter(initial)


def zeros(shape):
    """All zeros."""
    initial = torch.zeros(shape, dtype=torch.float32)
    return torch.nn.Parameter(initial)


def ones(shape, name=None):
    """All ones."""
    initial = torch.ones(shape, dtype=torch.float32)
    return torch.nn.Parameter(initial)
