# 3.训练train

wenet toolkit进行训练流程时，需为train.py使用时配置"--config xxx.yaml"，可以选择transformer、conformer等不同的模型结构。本片笔记以transformer为例，记录解读wenet toolkit的训练流程。

## 3.1 初始化总体模型框架
在train.py的169行调用init_asr_model()初始化模型，初始化模型的实际操作是在wenet.transformer.asr_model完成。
```
from wenet.transformer.asr_model import init_asr_model
# train.py ---> line 169
# Init asr model from configs
model = init_asr_model(configs)
```

## 3.2 初始化总体模型细节

#### init()
在asr_model.py文件,class ASRModel()类,__init__()方法,初始化模型总体框架。模型由四个部分组成：
- encoder,
- decoder,
- ctc,
- criterion_att
```
class ASRModel(torch.nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""
    def __init__(...):

        self.encoder = encoder
        self.decoder = decoder
        self.ctc = ctc
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
```

#### forward()
在forward方法(),定义数据的流向，计算方式。首先,进行encode.然后,同时并行执行CTC和decode.最后,加权求和ctc和decode得到最终loss.
```
    def forward(...) :
        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        # 2a. Attention-decoder branch
        if self.ctc_weight != 1.0:
            loss_att, acc_att = self._calc_att_loss(encoder_out, encoder_mask,
                                                    text, text_lengths)
        else:
            loss_att = None

        # 2b. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, text,
                                text_lengths)
        else:
            loss_ctc = None

        if loss_ctc is None:
            loss = loss_att
        elif loss_att is None:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 -
                                                 self.ctc_weight) * loss_att
        return loss, loss_att, loss_ctc
```

## 3.3 encode

### 3.3.1 encode总体结构

#### init()
下面代码是encoder.py的TransformerEncoder类的init函数()，它展示wenet的encoder结构。

首先，最外层是一个List,modulelist里面使用for循环生成结构一致的子结构---transformerencoderlayer,每一个子结构称为block。

然后，transformerencoderlayer由2大块组成，MultiHeadedAttention、PositionwiseFeedForward。PositionwiseFeedForward简单的全连接层，详细可以直接看源代码。

最后，如果选择Conformerencoderlayer作为block,而不是Transformerencoderlayer，子结构的子块可以从MultiHeadedAttention、RelPositionMultiHeadedAttention二者择其一。

MultiHeadedAttention、PositionwiseFeedForward的具体实现和计算需要看from wenet.transformer.encoder_layer import TransformerEncoderLayer的代码。

"attention is all you need"原文,Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two
sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network

```
class TransformerEncoder(BaseEncoder):
    """Transformer encoder module."""
    def __init__(...):
        """ Construct TransformerEncoder
        See Encoder for the meaning of each parameter.
        """
        super().__init__()
        self.encoders = torch.nn.ModuleList([
            TransformerEncoderLayer(
                output_size,
                MultiHeadedAttention(attention_heads, output_size,
                                     attention_dropout_rate),
                PositionwiseFeedForward(output_size, linear_units,
                                        dropout_rate), dropout_rate,
                normalize_before, concat_after) for _ in range(num_blocks)
        ])
```
#### forward()
1.执行mask待补充

2.执行TransformerEncoderLayer(MultiHeadedAttention, PositionwiseFeedForward)

3.重复运行6次，跑完6个block
```
    def forward(...) :
        """Embed positions in tensor."""
        masks = ~make_pad_mask(xs_lens).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks  # (B, 1, T/subsample_rate)
        chunk_masks = add_optional_chunk_mask()
        for layer in self.encoders:
            xs, chunk_masks, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks
```

### 3.3.2 TransformerEncoderLayer

#### init()

self.self_attn = 对应encoder里面的MultiHeadedAttention,
self.feed_forward = 对应encoder里面的PositionwiseFeedForward,
还有一些其他简单计算
```
    def __init__(...):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size, eps=1e-12)
        self.norm2 = nn.LayerNorm(size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        # concat_linear may be not used in forward fuction,
        # but will be saved in the *.pt
        self.concat_linear = nn.Linear(size + size, size)
```
#### forward()

下面展示wenet的encoder_lyayer.py代码中transformer的实际计算过程，其实现与《attention is all you need》略有修改。
* wenet使用transformer作为encoder时，无需在input emb加入position emb。
* wenet构造encoder时，有self.normalize_before参数可选，前置layer_norm在残差之前，或者是后置layer_nromwenet example提供的yaml配置文件中，使用的是前置layer_norm，放置于残差residual之前。

```
# 按example配置文件简化后的执行顺序
# 第一次 layer norm
residual = x
x = self.norm1(x)
# 计算attention, 然后第一次残差
x_q = x
x = residual + self.dropout(self.self_attn(x_q, x, x, mask))
# 第二次 layer norm
residual = x
x = self.norm2(x)
# 计算feed_forward，然后第二次残差
x = residual + self.dropout(self.feed_forward(x))
```

![transformer](https://github.com/woqulrlr/wenet-learning/blob/main/transformer.jpg)

#### 3.3.2.1 MultiHeadedAttention

#### init()

增加几个linear层，李沐视频中解释，经过lineaer投影模型有更多的参数可以学习。
```
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)
```
#### forward_qkv()
此处代码对应下面右图,multi_head实际实现通过view改变数据shape,变形成multi_head所需形式.[4,69,256]===>[4,69,(4,64)]
```
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
```
#### attention1 : torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

#### attention2 :

再次通过view完成多头的合并,multi_head合并，shape[4, 4, 69, 64]===>[4, 69, 256]

```
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0)  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (x.transpose(1, 2).contiguous().view(n_batch, -1,
                                                 self.h * self.d_k)
             )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)
```


![multi_head_attention](https://github.com/woqulrlr/wenet-learning/blob/main/multi_head_attention.jpg)

![multi_head_attention](https://github.com/woqulrlr/wenet-learning/blob/main/attention_formula.jpg)


#### 3.3.2.2 PositionwiseFeedForward

linear,activation,dropout,linear

```
    def __init__(self,
                 idim: int,
                 hidden_units: int,
                 dropout_rate: float,
                 activation: torch.nn.Module = torch.nn.ReLU()):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))
```