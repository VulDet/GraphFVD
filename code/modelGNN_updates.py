import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F
from torch.nn.modules.module import Module
from dgl.nn import TypedLinear
from dgl import function as fn
import scipy.sparse as sp

att_op_dict = {
    'sum': 'sum',
    'mul': 'mul',
    'concat': 'concat'
}

"""GatedGNN with residual connection"""
class ReGGNN(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, num_GNN_layers, dropout, act=nn.functional.relu,
                 residual=True, att_op='mul', alpha_weight=1.0):
        super(ReGGNN, self).__init__()
        self.num_GNN_layers = num_GNN_layers
        self.residual = residual
        self.att_op = att_op
        self.alpha_weight = alpha_weight
        self.out_dim = hidden_size
        if self.att_op == att_op_dict['concat']:
            self.out_dim = hidden_size * 2

        self.emb_encode = nn.Linear(feature_dim_size, hidden_size).double()
        self.dropout_encode = nn.Dropout(dropout)
        self.z0 = nn.Linear(hidden_size, hidden_size).double()
        self.z1 = nn.Linear(hidden_size, hidden_size).double()
        self.r0 = nn.Linear(hidden_size, hidden_size).double()
        self.r1 = nn.Linear(hidden_size, hidden_size).double()
        self.h0 = nn.Linear(hidden_size, hidden_size).double()
        self.h1 = nn.Linear(hidden_size, hidden_size).double()
        self.soft_att = nn.Linear(hidden_size, 1).double()
        self.ln = nn.Linear(hidden_size, hidden_size).double()
        self.act = act

    def gatedGNN(self, x, adj):
        a = torch.matmul(adj, x)
        # update gate
        z0 = self.z0(a.double())
        z1 = self.z1(x.double())
        z = torch.sigmoid(z0 + z1)
        # reset gate
        r = torch.sigmoid(self.r0(a.double()) + self.r1(x.double()))
        # update embeddings
        h = self.act(self.h0(a.double()) + self.h1(r.double() * x.double()))

        return h * z + x * (1 - z)

    def forward(self, inputs, adj, mask):
        x = inputs
        x = self.dropout_encode(x)
        x = self.emb_encode(x.double())
        x = x * mask
        for idx_layer in range(self.num_GNN_layers):
            if self.residual:
                x = x + self.gatedGNN(x.double(), adj.double()) * mask.double()  # add residual connection, can use a weighted sum
            else:
                x = self.gatedGNN(x.double(), adj.double()) * mask.double()
        # soft attention
        soft_att = torch.sigmoid(self.soft_att(x))
        x = self.act(self.ln(x))
        x = soft_att * x * mask
        # sum/mean and max pooling

        # sum and max pooling
        if self.att_op == att_op_dict['sum']:
            graph_embeddings = torch.sum(x, 1) + torch.amax(x, 1)
        elif self.att_op == att_op_dict['concat']:
            graph_embeddings = torch.cat((torch.sum(x, 1), torch.amax(x, 1)), 1)
        else:
            graph_embeddings = torch.sum(x, 1) * torch.amax(x, 1)

        return graph_embeddings  

"""GCNs with residual connection"""
class ReGCN(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, num_GNN_layers, dropout, num_relations=3, num_bases=3, act=nn.functional.relu,
                 residual=True, att_op="mul", alpha_weight=1.0):
        super(ReGCN, self).__init__()
        self.num_GNN_layers = num_GNN_layers
        self.residual = residual
        self.att_op = att_op
        self.alpha_weight = alpha_weight
        self.hidden_size = hidden_size
        self.out_dim = hidden_size
        if self.att_op == att_op_dict['concat']:
            self.out_dim = hidden_size * 2

        self.gnnlayers = torch.nn.ModuleList()
        for layer in range(self.num_GNN_layers):
            if layer == 0:
                self.gnnlayers.append(RelationalGraphConvLayer(feature_dim_size, hidden_size, num_bases, num_relations, dropout, bias=True))  # bias=False
            else:
                self.gnnlayers.append(RelationalGraphConvLayer(hidden_size, hidden_size, num_bases, num_relations, dropout, bias=True))
        self.soft_att0 = nn.Linear(32, 1).double()
        self.ln0 = nn.Linear(1, 32).double()

        self.q = nn.Linear(hidden_size, hidden_size).double()
        self.k = nn.Linear(hidden_size, hidden_size).double()
        self.v = nn.Linear(hidden_size, hidden_size).double()
        self.soft_att = nn.Linear(hidden_size, 1).double()
        self.ln = nn.Linear(hidden_size, hidden_size).double()
        self.act = act
       
    def forward(self, inputs, adj, mask):
        x = inputs
        x = x.permute(0, 1, 3, 2)
        attn_token = torch.tanh(self.soft_att0(x.double()).double())
        attn_token = torch.nn.functional.softmax(self.ln0(attn_token), dim=3)
        x = attn_token * x
        x = torch.sum(x, 3)

        for idx_layer in range(self.num_GNN_layers):
            if idx_layer == 0:
                x = self.gnnlayers[idx_layer](adj, x) * mask
            else:
                if self.residual:
                    x = 0.4 * x + 0.6 * self.gnnlayers[idx_layer](adj, x) * mask  # Residual Connection, can use a weighted sum
                else:
                    x = self.gnnlayers[idx_layer](adj, x) * mask
        # soft attention
        query = self.q(x.double()).double()
        key = self.k(x.double()).double()
        value = self.v(x.double()).double()
        attn_node = torch.matmul(query, key.permute(0, 2, 1))
        attn_node = attn_node / math.sqrt(self.hidden_size)
        attn_probs = torch.nn.functional.softmax(attn_node, dim=-1)
        context = torch.matmul(attn_probs, value)
        graph_embeddings = torch.sum(context, 1) * torch.amax(context, 1)

        return graph_embeddings


"""GatedGNN"""
class GGGNN(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, num_GNN_layers, dropout, act=nn.functional.relu):
        super(GGGNN, self).__init__()
        self.num_GNN_layers = num_GNN_layers
        self.emb_encode = nn.Linear(feature_dim_size, hidden_size).double()
        self.dropout_encode = nn.Dropout(dropout)
        self.z0 = nn.Linear(hidden_size, hidden_size).double()
        self.z1 = nn.Linear(hidden_size, hidden_size).double()
        self.r0 = nn.Linear(hidden_size, hidden_size).double()
        self.r1 = nn.Linear(hidden_size, hidden_size).double()
        self.h0 = nn.Linear(hidden_size, hidden_size).double()
        self.h1 = nn.Linear(hidden_size, hidden_size).double()
        self.soft_att = nn.Linear(hidden_size, 1).double()
        self.ln = nn.Linear(hidden_size, hidden_size).double()
        self.act = act

    def gatedGNN(self, x, adj):
        a = torch.matmul(adj, x)
        # update gate
        z0 = self.z0(a.double())
        z1 = self.z1(x.double())
        z = torch.sigmoid(z0 + z1)
        # reset gate
        r = torch.sigmoid(self.r0(a.double()) + self.r1(x.double()))
        # update embeddings
        h = self.act(self.h0(a.double()) + self.h1(r.double() * x.double()))

        return h * z + x * (1 - z)

    def forward(self, inputs, adj, mask):
        x = inputs
        x = self.dropout_encode(x)
        x = self.emb_encode(x.double())
        x = x * mask
        for idx_layer in range(self.num_GNN_layers):
            x = self.gatedGNN(x.double(), adj.double()) * mask.double()
        return x


""" Simple GCN layer, similar to https://arxiv.org/abs/1609.02907 """
class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout, act=torch.relu, bias=False):
        super(GraphConvolution, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.act = act
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        x = self.dropout(input)
        support = torch.matmul(x.double(), self.weight.double())
        output = torch.matmul(adj.double(), support.double())
        if self.bias is not None:
            output = output + self.bias
        return self.act(output)


class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, n_layers=2, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.relu = torch.relu
        self.dropout = dropout
        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations))
        for i in range(n_layers - 2):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations))
        self.convs.append(RGCNConv(hidden_channels, out_channels, num_relations))

    def forward(self, x, edge_index, edge_type):
        for conv, norm in zip(self.convs, self.norms):
            x = norm(conv(x, edge_index, edge_type))
            x = self.relu(x)
            x = self.dropout(x, p=self.dropout, training=self.training)
        return x

class RelationalGraphConvLayer(Module):
    def __init__(
        self, input_size, output_size, num_bases, num_rel, dropout, bias=False):
        super(RelationalGraphConvLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_bases = num_bases
        self.num_rel = num_rel
        self.act = torch.relu
        self.dropout = nn.Dropout(dropout)
        # R-GCN weights
        if num_bases > 0:
            self.w_bases = Parameter(torch.FloatTensor(self.num_bases, self.input_size, self.output_size))
            self.w_rel = Parameter(torch.FloatTensor(self.num_rel, self.num_bases))
        else:
            self.w = Parameter(torch.FloatTensor(self.num_rel, self.input_size, self.output_size))
        # R-GCN bias
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.output_size))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.num_bases > 0:
            nn.init.xavier_uniform_(self.w_bases.data)
            nn.init.xavier_uniform_(self.w_rel.data)
        else:
            nn.init.xavier_uniform_(self.w.data)
        if self.bias is not None:
            nn.init.xavier_uniform_(self.bias.data.unsqueeze(0))


    def forward(self, A, X):
        X = self.dropout(X)
        self.w = (
            torch.einsum("rb, bio -> rio", (self.w_rel, self.w_bases))
            if self.num_bases > 0
            else self.w
        )
        weights = self.w.view(self.w.shape[0] * self.w.shape[1], self.w.shape[2])
        # Each relations * Weight
        supports = []
        A = A.permute(1, 0, 2, 3)
        for i in range(self.num_rel):
            supports.append(torch.matmul(A[i].double(), X.double()))

        tmp = torch.cat(supports, dim=2)
        out = torch.matmul(tmp.double(), weights.double())  # shape(#node, output_size)

        if self.bias is not None:
            out += self.bias.unsqueeze(0)
        return self.act(out)


def to_sparse(x):
    """converts dense tensor x to sparse format"""
    x_typename = torch.typename(x).split(".")[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def csr2tensor(A, cuda):
    coo = A.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    if cuda:
        out = torch.sparse.FloatTensor(i, v, torch.Size(shape)).cuda()
    else:
        out = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return out

class RelGraphConv(torch.nn.Module):
    r"""Relational graph convolution layer from `Modeling Relational Data with Graph
    Convolutional Networks <https://arxiv.org/abs/1703.06103>`__

    It can be described in as below:

    .. math::

       h_i^{(l+1)} = \sigma(\sum_{r\in\mathcal{R}}
       \sum_{j\in\mathcal{N}^r(i)}e_{j,i}W_r^{(l)}h_j^{(l)}+W_0^{(l)}h_i^{(l)})

    where :math:`\mathcal{N}^r(i)` is the neighbor set of node :math:`i` w.r.t. relation
    :math:`r`. :math:`e_{j,i}` is the normalizer. :math:`\sigma` is an activation
    function. :math:`W_0` is the self-loop weight.

    The basis regularization decomposes :math:`W_r` by:

    .. math::

       W_r^{(l)} = \sum_{b=1}^B a_{rb}^{(l)}V_b^{(l)}

    where :math:`B` is the number of bases, :math:`V_b^{(l)}` are linearly combined
    with coefficients :math:`a_{rb}^{(l)}`.

    The block-diagonal-decomposition regularization decomposes :math:`W_r` into :math:`B`
    number of block diagonal matrices. We refer :math:`B` as the number of bases.

    The block regularization decomposes :math:`W_r` by:

    .. math::

       W_r^{(l)} = \oplus_{b=1}^B Q_{rb}^{(l)}

    where :math:`B` is the number of bases, :math:`Q_{rb}^{(l)}` are block
    bases with shape :math:`R^{(d^{(l+1)}/B)*(d^{l}/B)}`.

    Parameters
    ----------
    in_feat : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    out_feat : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.
    num_rels : int
        Number of relations.
    regularizer : str, optional
        Which weight regularizer to use ("basis", "bdd" or ``None``):

         - "basis" is for basis-decomposition.
         - "bdd" is for block-diagonal-decomposition.
         - ``None`` applies no regularization.

        Default: ``None``.
    num_bases : int, optional
        Number of bases. It comes into effect when a regularizer is applied.
        If ``None``, it uses number of relations (``num_rels``). Default: ``None``.
        Note that ``in_feat`` and ``out_feat`` must be divisible by ``num_bases``
        when applying "bdd" regularizer.
    bias : bool, optional
        True if bias is added. Default: ``True``.
    activation : callable, optional
        Activation function. Default: ``None``.
    self_loop : bool, optional
        True to include self loop message. Default: ``True``.
    dropout : float, optional
        Dropout rate. Default: ``0.0``
    layer_norm: bool, optional
        True to add layer norm. Default: ``False``

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import RelGraphConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 10)
    >>> conv = RelGraphConv(10, 2, 3, regularizer='basis', num_bases=2)
    >>> etype = th.tensor([0,1,2,0,1,2])
    >>> res = conv(g, feat, etype)
    >>> res
    tensor([[ 0.3996, -2.3303],
            [-0.4323, -0.1440],
            [ 0.3996, -2.3303],
            [ 2.1046, -2.8654],
            [-0.4323, -0.1440],
            [-0.1309, -1.0000]], grad_fn=<AddBackward0>)
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_rels,
                 regularizer=None,
                 num_bases=None,
                 bias=True,
                 activation=None,
                 self_loop=True,
                 dropout=0.0,
                 layer_norm=False):
        super().__init__()
        if regularizer is not None and num_bases is None:
            num_bases = num_rels
        self.linear_r = TypedLinear(in_feat, out_feat, num_rels, regularizer, num_bases)
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.layer_norm = layer_norm

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # TODO(minjie): consider remove those options in the future to make
        #   the module only about graph convolution.
        # layer norm
        if self.layer_norm:
            self.layer_norm_weight = nn.LayerNorm(out_feat, elementwise_affine=True)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def message(self, edges):
        """Message function."""
        m = self.linear_r(edges.src['h'], edges.data['etype'], self.presorted)
        if 'norm' in edges.data:
            m = m * edges.data['norm']
        return {'m' : m}

    def forward(self, feat, adj, etypes, norm=None, *, presorted=False):
        """Forward computation.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        feat : torch.Tensor
            A 2D tensor of node features. Shape: :math:`(|V|, D_{in})`.
        etypes : torch.Tensor or list[int]
            An 1D integer tensor of edge types. Shape: :math:`(|E|,)`.
        norm : torch.Tensor, optional
            An 1D tensor of edge norm value.  Shape: :math:`(|E|,)`.
        presorted : bool, optional
            Whether the edges of the input graph have been sorted by their types.
            Forward on pre-sorted graph may be faster. Graphs created
            by :func:`~dgl.to_homogeneous` automatically satisfy the condition.
            Also see :func:`~dgl.reorder_graph` for sorting edges manually.

        Returns
        -------
        torch.Tensor
            New node features. Shape: :math:`(|V|, D_{out})`.
        """
        self.presorted = presorted
        with g.local_scope():
            g.srcdata['h'] = feat
            if norm is not None:
                g.edata['norm'] = norm
            g.edata['etype'] = etypes
            # message passing
            g.update_all(self.message, fn.sum('m', 'h'))
            # apply bias and activation
            h = g.dstdata['h']
            if self.layer_norm:
                h = self.layer_norm_weight(h)
            if self.bias:
                h = h + self.h_bias
            if self.self_loop:
                h = h + feat[:g.num_dst_nodes()] @ self.loop_weight
            if self.activation:
                h = self.activation(h)
            h = self.dropout(h)
            return h

weighted_graph = False
print('using default unweighted graph')
