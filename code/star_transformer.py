## Ref: fastNLP
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def seq_len_to_mask(seq_len, max_len=None):
    r"""
    将一个表示sequence length的一维数组转换为二维的mask，不包含的位置为0。
    转变 1-d seq_len到2-d mask.
    .. code-block::
    
        >>> seq_len = torch.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.size())
        torch.Size([14, 15])
        >>> seq_len = np.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.shape)
        (14, 15)
        >>> seq_len = torch.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len, max_len=100)
        >>>print(mask.size())
        torch.Size([14, 100])
    :param np.ndarray,torch.LongTensor seq_len: shape将是(B,)
    :param int max_len: 将长度pad到这个长度。默认(None)使用的是seq_len中最长的长度。但在nn.DataParallel的场景下可能不同卡的seq_len会有
        区别，所以需要传入一个max_len使得mask的长度是pad到该长度。
    :return: np.ndarray, torch.Tensor 。shape将是(B, max_length)， 元素类似为bool或torch.uint8
    """
    if isinstance(seq_len, np.ndarray):
        assert len(np.shape(seq_len)) == 1, f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        max_len = int(max_len) if max_len else int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)

    elif isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask
    
def get_embeddings(init_embed, padding_idx=None):
    r"""
    根据输入的init_embed返回Embedding对象。如果输入是tuple, 则随机初始化一个nn.Embedding; 如果输入是numpy.ndarray, 则按照ndarray
    的值将nn.Embedding初始化; 如果输入是torch.Tensor, 则按该值初始化nn.Embedding; 如果输入是fastNLP中的embedding将不做处理
    返回原对象。
    :param init_embed: 可以是 tuple:(num_embedings, embedding_dim), 即embedding的大小和每个词的维度;也可以传入
        nn.Embedding 对象, 此时就以传入的对象作为embedding; 传入np.ndarray也行，将使用传入的ndarray作为作为Embedding初始化;
        传入torch.Tensor, 将使用传入的值作为Embedding初始化。
    :param padding_idx: 当传入tuple时，padding_idx有效
    :return nn.Embedding:  embeddings
    """
    if isinstance(init_embed, tuple):
        res = nn.Embedding(
            num_embeddings=init_embed[0], embedding_dim=init_embed[1], padding_idx=padding_idx)
        nn.init.uniform_(res.weight.data, a=-np.sqrt(3 / res.weight.data.size(1)),
                         b=np.sqrt(3 / res.weight.data.size(1)))
    elif isinstance(init_embed, nn.Module):
        res = init_embed
    elif isinstance(init_embed, torch.Tensor):
        res = nn.Embedding.from_pretrained(init_embed, freeze=False)
    elif isinstance(init_embed, np.ndarray):
        init_embed = torch.tensor(init_embed, dtype=torch.float32)
        res = nn.Embedding.from_pretrained(init_embed, freeze=False)
    else:
        raise TypeError(
            'invalid init_embed type: {}'.format((type(init_embed))))
    return res

class StarTransformer(nn.Module):
    r"""
    Star-Transformer 的encoder部分。 输入3d的文本输入, 返回相同长度的文本编码
    paper: https://arxiv.org/abs/1902.09113
    """

    def __init__(self, hidden_size, num_layers, num_head, head_dim, dropout=0.1, max_len=None):
        r"""
        
        :param int hidden_size: 输入维度的大小。同时也是输出维度的大小。
        :param int num_layers: star-transformer的层数
        :param int num_head: head的数量。
        :param int head_dim: 每个head的维度大小。
        :param float dropout: dropout 概率. Default: 0.1
        :param int max_len: int or None, 如果为int，输入序列的最大长度，
            模型会为输入序列加上position embedding。
            若为`None`，忽略加上position embedding的步骤. Default: `None`
        """
        super(StarTransformer, self).__init__()
        self.iters = num_layers

        self.norm = nn.ModuleList([nn.LayerNorm(hidden_size, eps=1e-6) for _ in range(self.iters)])
        # self.emb_fc = nn.Conv2d(hidden_size, hidden_size, 1)
        self.emb_drop = nn.Dropout(dropout)
        self.ring_att = nn.ModuleList(
            [_MSA1(hidden_size, nhead=num_head, head_dim=head_dim, dropout=0.0)
             for _ in range(self.iters)])
        self.star_att = nn.ModuleList(
            [_MSA2(hidden_size, nhead=num_head, head_dim=head_dim, dropout=0.0)
             for _ in range(self.iters)])

        if max_len is not None:
            self.pos_emb = nn.Embedding(max_len, hidden_size)
        else:
            self.pos_emb = None

    def forward(self, data, mask):
        r"""
        :param FloatTensor data: [batch, length, hidden] 输入的序列
        :param ByteTensor mask: [batch, length] 输入序列的padding mask, 在没有内容(padding 部分) 为 0,
            否则为 1
        :return: [batch, length, hidden] 编码后的输出序列
                [batch, hidden] 全局 relay 节点, 详见论文
        """

        def norm_func(f, x):
            # B, H, L, 1
            return f(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        B, L, H = data.size()
        mask = (mask.eq(False))  # flip the mask for masked_fill_
        smask = torch.cat([torch.zeros(B, 1, ).byte().to(mask), mask], 1)

        embs = data.permute(0, 2, 1)[:, :, :, None]  # B H L 1
        if self.pos_emb:
            P = self.pos_emb(torch.arange(L, dtype=torch.long, device=embs.device) \
                             .view(1, L)).permute(0, 2, 1).contiguous()[:, :, :, None]  # 1 H L 1
            embs = embs + P
        embs = norm_func(self.emb_drop, embs)
        nodes = embs
        relay = embs.mean(2, keepdim=True)
        ex_mask = mask[:, None, :, None].expand(B, H, L, 1)
        r_embs = embs.view(B, H, 1, L)
        for i in range(self.iters):
            ax = torch.cat([r_embs, relay.expand(B, H, 1, L)], 2)
            nodes = F.leaky_relu(self.ring_att[i](norm_func(self.norm[i], nodes), ax=ax))
            # nodes = F.leaky_relu(self.ring_att[i](nodes, ax=ax))
            relay = F.leaky_relu(self.star_att[i](relay, torch.cat([relay, nodes], 2), smask))

            nodes = nodes.masked_fill_(ex_mask, 0)

        nodes = nodes.view(B, H, L).permute(0, 2, 1)

        return nodes, relay.view(B, H)


class _MSA1(nn.Module):
    def __init__(self, nhid, nhead=10, head_dim=10, dropout=0.1):
        super(_MSA1, self).__init__()
        # Multi-head Self Attention Case 1, doing self-attention for small regions
        # Due to the architecture of GPU, using hadamard production and summation are faster than dot production when unfold_size is very small
        self.WQ = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WK = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WV = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WO = nn.Conv2d(nhead * head_dim, nhid, 1)

        self.drop = nn.Dropout(dropout)

        # print('NUM_HEAD', nhead, 'DIM_HEAD', head_dim)
        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, nhead, head_dim, 3

    def forward(self, x, ax=None):
        # x: B, H, L, 1, ax : B, H, X, L append features
        nhid, nhead, head_dim, unfold_size = self.nhid, self.nhead, self.head_dim, self.unfold_size
        B, H, L, _ = x.shape

        q, k, v = self.WQ(x), self.WK(x), self.WV(x)  # x: (B,H,L,1)

        if ax is not None:
            aL = ax.shape[2]
            ak = self.WK(ax).view(B, nhead, head_dim, aL, L)
            av = self.WV(ax).view(B, nhead, head_dim, aL, L)
        q = q.view(B, nhead, head_dim, 1, L)
        k = F.unfold(k.view(B, nhead * head_dim, L, 1), (unfold_size, 1), padding=(unfold_size // 2, 0)) \
            .view(B, nhead, head_dim, unfold_size, L)
        v = F.unfold(v.view(B, nhead * head_dim, L, 1), (unfold_size, 1), padding=(unfold_size // 2, 0)) \
            .view(B, nhead, head_dim, unfold_size, L)
        if ax is not None:
            k = torch.cat([k, ak], 3)
            v = torch.cat([v, av], 3)

        alphas = self.drop(F.softmax((q * k).sum(2, keepdim=True) / np.sqrt(head_dim), 3))  # B N L 1 U
        att = (alphas * v).sum(3).view(B, nhead * head_dim, L, 1)

        ret = self.WO(att)

        return ret


class _MSA2(nn.Module):
    def __init__(self, nhid, nhead=10, head_dim=10, dropout=0.1):
        # Multi-head Self Attention Case 2, a broadcastable query for a sequence key and value
        super(_MSA2, self).__init__()
        self.WQ = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WK = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WV = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WO = nn.Conv2d(nhead * head_dim, nhid, 1)

        self.drop = nn.Dropout(dropout)

        # print('NUM_HEAD', nhead, 'DIM_HEAD', head_dim)
        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, nhead, head_dim, 3

    def forward(self, x, y, mask=None):
        # x: B, H, 1, 1, 1 y: B H L 1
        nhid, nhead, head_dim, unfold_size = self.nhid, self.nhead, self.head_dim, self.unfold_size
        B, H, L, _ = y.shape

        q, k, v = self.WQ(x), self.WK(y), self.WV(y)

        q = q.view(B, nhead, 1, head_dim)  # B, H, 1, 1 -> B, N, 1, h
        k = k.view(B, nhead, head_dim, L)  # B, H, L, 1 -> B, N, h, L
        v = v.view(B, nhead, head_dim, L).permute(0, 1, 3, 2)  # B, H, L, 1 -> B, N, L, h
        pre_a = torch.matmul(q, k) / np.sqrt(head_dim)
        if mask is not None:
            pre_a = pre_a.masked_fill(mask[:, None, None, :], -float('inf'))
        alphas = self.drop(F.softmax(pre_a, 3))  # B, N, 1, L
        att = torch.matmul(alphas, v).view(B, -1, 1, 1)  # B, N, 1, h -> B, N*h, 1, 1
        return self.WO(att)

class StarTransEnc(nn.Module):
    r"""
    带word embedding的Star-Transformer Encoder
    """

    def __init__(self, embed,
                 hidden_size,
                 num_layers,
                 num_head,
                 head_dim,
                 max_len,
                 emb_dropout,
                 dropout):
        r"""
        
        :param embed: 单词词典, 可以是 tuple, 包括(num_embedings, embedding_dim), 即
            embedding的大小和每个词的维度. 也可以传入 nn.Embedding 对象,此时就以传入的对象作为embedding
        :param hidden_size: 模型中特征维度.
        :param num_layers: 模型层数.
        :param num_head: 模型中multi-head的head个数.
        :param head_dim: 模型中multi-head中每个head特征维度.
        :param max_len: 模型能接受的最大输入长度.
        :param emb_dropout: 词嵌入的dropout概率.
        :param dropout: 模型除词嵌入外的dropout概率.
        """
        super(StarTransEnc, self).__init__()
        self.embedding = get_embeddings(embed)
        emb_dim = self.embedding.embedding_dim
        self.emb_fc = nn.Linear(emb_dim, hidden_size)
        # self.emb_drop = nn.Dropout(emb_dropout)
        self.encoder = StarTransformer(hidden_size=hidden_size,
                                       num_layers=num_layers,
                                       num_head=num_head,
                                       head_dim=head_dim,
                                       dropout=dropout,
                                       max_len=max_len)

    def forward(self, x, mask):
        r"""
        :param FloatTensor x: [batch, length, hidden] 输入的序列
        :param ByteTensor mask: [batch, length] 输入序列的padding mask, 在没有内容(padding 部分) 为 0,
            否则为 1
        :return: [batch, length, hidden] 编码后的输出序列
                [batch, hidden] 全局 relay 节点, 详见论文
        """
        x = self.embedding(x)
        x = self.emb_fc(x)
        nodes, relay = self.encoder(x, mask)
        return nodes, relay


class _Cls(nn.Module):
    def __init__(self, in_dim, num_cls, hid_dim, dropout=0.1):
        super(_Cls, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, num_cls),
        )

    def forward(self, x):
        h = self.fc(x)
        return h


class STSeqCls(nn.Module):
    r"""
    用于分类任务的Star-Transformer
    """

    def __init__(self, embed, num_cls,
                 hidden_size=300,
                 num_layers=4,
                 num_head=8,
                 head_dim=32,
                 max_len=512,
                 cls_hidden_size=600,
                 emb_dropout=0.1,
                 dropout=0.1, ):
        r"""
        
        :param embed: 单词词典, 可以是 tuple, 包括(num_embedings, embedding_dim), 即
            embedding的大小和每个词的维度. 也可以传入 nn.Embedding 对象, 此时就以传入的对象作为embedding
        :param num_cls: 输出类别个数
        :param hidden_size: 模型中特征维度. Default: 300
        :param num_layers: 模型层数. Default: 4
        :param num_head: 模型中multi-head的head个数. Default: 8
        :param head_dim: 模型中multi-head中每个head特征维度. Default: 32
        :param max_len: 模型能接受的最大输入长度. Default: 512
        :param cls_hidden_size: 分类器隐层维度. Default: 600
        :param emb_dropout: 词嵌入的dropout概率. Default: 0.1
        :param dropout: 模型除词嵌入外的dropout概率. Default: 0.1
        """
        super(STSeqCls, self).__init__()
        self.enc = StarTransEnc(embed=embed,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                num_head=num_head,
                                head_dim=head_dim,
                                max_len=max_len,
                                emb_dropout=emb_dropout,
                                dropout=dropout)
        self.cls = _Cls(hidden_size, num_cls, cls_hidden_size, dropout=dropout)
        self.max_len = max_len

    def forward(self, words, seq_len):
        r"""
        :param words: [batch, seq_len] 输入序列
        :param seq_len: [batch,] 输入序列的长度
        :return output: [batch, num_cls] 输出序列的分类的概率
        """
        mask = seq_len_to_mask(seq_len)
        nodes, relay = self.enc(words, mask)
        y = 0.5 * (relay + nodes.max(1)[0])
        output = self.cls(y)  # [bsz, n_cls]
        return output

class STSeqCls_Pred(nn.Module):
    r"""
    用于分类任务的Star-Transformer
    """

    def __init__(self, embed, num_cls,
                 hidden_size=300,
                 num_layers=4,
                 num_head=8,
                 head_dim=32,
                 max_len=512,
                 cls_hidden_size=600,
                 emb_dropout=0.1,
                 dropout=0.1, ):
        super(STSeqCls_Pred, self).__init__()
        self.enc = StarTransEnc(embed=embed,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                num_head=num_head,
                                head_dim=head_dim,
                                max_len=max_len,
                                emb_dropout=emb_dropout,
                                dropout=dropout)
        self.cls = _Cls(hidden_size, num_cls, cls_hidden_size, dropout=dropout)
        self.max_len = max_len

    def forward(self, words, seq_len):
        r"""
        :param words: [batch, seq_len] 输入序列
        :param seq_len: [batch,] 输入序列的长度
        :return output: [batch, num_cls] 输出序列的分类的概率
        """
        mask = seq_len_to_mask(seq_len, max_len=50)
        nodes, relay = self.enc(words, mask)
        y = 0.5 * (relay + nodes.max(1)[0])
        output = self.cls(y)  # [bsz, n_cls]
        return output

class STCNNSeqCls(nn.Module):
    r"""
    用于分类任务的Star-Transformer-CNN
    """

    def __init__(self, embed, num_cls,
                 hidden_size=300,
                 num_layers=4,
                 num_head=8,
                 head_dim=32,
                 max_len=512,
                 cls_hidden_size=600,
                 emb_dropout=0.1,
                 dropout=0.1, ):
        r"""
        
        :param embed: 单词词典, 可以是 tuple, 包括(num_embedings, embedding_dim), 即
            embedding的大小和每个词的维度. 也可以传入 nn.Embedding 对象, 此时就以传入的对象作为embedding
        :param num_cls: 输出类别个数
        :param hidden_size: 模型中特征维度. Default: 300
        :param num_layers: 模型层数. Default: 4
        :param num_head: 模型中multi-head的head个数. Default: 8
        :param head_dim: 模型中multi-head中每个head特征维度. Default: 32
        :param max_len: 模型能接受的最大输入长度. Default: 512
        :param cls_hidden_size: 分类器隐层维度. Default: 600
        :param emb_dropout: 词嵌入的dropout概率. Default: 0.1
        :param dropout: 模型除词嵌入外的dropout概率. Default: 0.1
        """
        super(STCNNSeqCls, self).__init__()
        self.enc = StarTransEnc(embed=embed,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                num_head=num_head,
                                head_dim=head_dim,
                                max_len=max_len,
                                emb_dropout=emb_dropout,
                                dropout=dropout)
        self.dropout = nn.Dropout(0.1)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size//2, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size//2, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size//2, kernel_size=5),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.cls = _Cls(250, num_cls, cls_hidden_size, dropout=dropout)

    def forward(self, words, seq_len):
        r"""
        :param words: [batch, seq_len] 输入序列
        :param seq_len: [batch,] 输入序列的长度
        :return output: [batch, num_cls] 输出序列的分类的概率
        """
        mask = seq_len_to_mask(seq_len)
        nodes, relay = self.enc(words, mask)
        nodes = nodes.permute(0,2,1)
        conv1_out = self.conv1(nodes).squeeze(2)
        conv2_out = self.conv2(nodes).squeeze(2)
        conv3_out = self.conv3(nodes).squeeze(2)
        out = torch.cat([conv1_out,conv1_out,conv1_out,relay],1)
        output = self.cls(self.dropout(out))
        return output

class STLSTMSeqCls(nn.Module):
    r"""
    用于分类任务的Star-Transformer-LSTM
    """

    def __init__(self, embed, num_cls,
                 hidden_size=300,
                 num_layers=4,
                 num_head=8,
                 head_dim=32,
                 max_len=512,
                 cls_hidden_size=600,
                 emb_dropout=0.1,
                 dropout=0.1, ):
        r"""
        
        :param embed: 单词词典, 可以是 tuple, 包括(num_embedings, embedding_dim), 即
            embedding的大小和每个词的维度. 也可以传入 nn.Embedding 对象, 此时就以传入的对象作为embedding
        :param num_cls: 输出类别个数
        :param hidden_size: 模型中特征维度. Default: 300
        :param num_layers: 模型层数. Default: 4
        :param num_head: 模型中multi-head的head个数. Default: 8
        :param head_dim: 模型中multi-head中每个head特征维度. Default: 32
        :param max_len: 模型能接受的最大输入长度. Default: 512
        :param cls_hidden_size: 分类器隐层维度. Default: 600
        :param emb_dropout: 词嵌入的dropout概率. Default: 0.1
        :param dropout: 模型除词嵌入外的dropout概率. Default: 0.1
        """
        super(STLSTMSeqCls, self).__init__()
        self.enc = StarTransEnc(embed=embed,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                num_head=num_head,
                                head_dim=head_dim,
                                max_len=max_len,
                                emb_dropout=emb_dropout,
                                dropout=dropout)
        self.cls = _Cls(hidden_size, num_cls, cls_hidden_size)
        self.rnn = nn.LSTM(hidden_size,
                            hidden_size=100,
                            num_layers=2,
                            bidirectional=True,
                            batch_first=True,
                            dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.fc_rnn = nn.Linear(100 * 3, num_cls)

    def forward(self, words, seq_len):
        r"""
        :param words: [batch, seq_len] 输入序列
        :param seq_len: [batch,] 输入序列的长度
        :return output: [batch, num_cls] 输出序列的分类的概率
        """
        mask = seq_len_to_mask(seq_len)
        nodes, relay = self.enc(words, mask)
        # 按照句子长度从大到小排序
        bert_sentence, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(nodes, seq_len)
        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(bert_sentence, sorted_seq_lengths.cpu(),
                                                            batch_first=True)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # output = [ batch size,sent len, hidden_dim * bidirectional]
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        output = output[desorted_indices]
        hidden = hidden[:,desorted_indices]

        batch_size, max_seq_len, hidden_dim = output.shape
        hidden = torch.mean(torch.reshape(hidden, [batch_size, -1, hidden_dim]), dim=1)
        output = torch.sum(output, dim=1)
        fc_input = self.dropout(output + hidden)
        # fc_input += relay.repeat(1,2)
        fc_input = torch.concat([fc_input, relay],dim=1)
        out = self.fc_rnn(fc_input)

        return out