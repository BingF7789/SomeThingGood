import argparse

def parameter_parser():
    """
    A parser for command line parameters.
    Default is master script run on port 7789
    """

    parser = argparse.ArgumentParser(description="Run .")

    # Training Setup
    parser.add_argument('-seed', type=int, default=1, help='seed')
    parser.add_argument('-device', type=int, default=-1, help='-1 for cpu, 0...n for GPU number')

    # methods setting from tf
    parser.add_argument('-save_name', nargs="?", default='./mymodel.ckpt', help='Path for saving model')
    parser.add_argument('-logDir', default='./log/default.txt', help='Path for log info')
    parser.add_argument('-dataset', default='ppi', help='Dataset string.')
    parser.add_argument('-data_prefix', default='data/', help='Datapath prefix.')
    parser.add_argument('-lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('-epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('-slave_ep', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
    parser.add_argument('-hidden1', type=int, default=2048, help='Number of units in hidden layer 1.')
    parser.add_argument('-dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
    parser.add_argument('-weight_decay', type=int, default=0, help='Weight for L2 loss on embedding matrix.')
    parser.add_argument('-early_stopping', type=int, default=1000, help='Tolerance for early stopping (# of epochs).')
    parser.add_argument('-num_clusters', type=int, default=50, help='Number of clusters.')
    parser.add_argument('-bsize', type=int, default=1, help='Number of clusters for each batch.')
    parser.add_argument('-num_clusters_val', type=int, default=5, help='Number of clusters for validation.')
    parser.add_argument('-num_clusters_test', type=int, default=1, help='Number of clusters for test.')
    parser.add_argument('-num_layers', type=int, default=5, help='Number of GCN layers.')
    parser.add_argument('-diag_lambda', type=float, default=1, help='A positive number for diagonal enhancement, -1 indicates normalization without diagonal enhancement')
    parser.add_argument('-multilabel', type=int, default=1, help='Multilabel or multiclass.')
    parser.add_argument('-layernorm', type=int, default=1, help='Whether to use layer normalization.')
    parser.add_argument('-precalc', type=int, default=1, help='Whether to pre-calculate the first layer (AX preprocessing).')
    parser.add_argument('-validation', default=True, help='Print validation accuracy after each epoch.')

    # Methods setting from pytorch
    parser.add_argument('--agg', type=str, default='avg', help='averaging strategy')
    parser.add_argument('-frac', type=float, default=1, help='the fraction of clients')
    parser.add_argument('--dp', type=float, default=0.001, help='differential privacy')
    parser.add_argument('-val_freq', type=int, default=10, help='validation freq')

    parser.set_defaults(layers=[16, 16, 16])

    return parser.parse_args()

