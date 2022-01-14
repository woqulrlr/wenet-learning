本篇主要内容是wenet模型的代码笔记、工程向的阅读笔记

# 1.* 数据采集

# 2.* 数据处理

# 3.训练train

wenet toolkit进行训练流程时，需为train.py使用时配置"--config xxx.yaml"，可以选择transformer、conformer等不同的模型结构。
本片笔记以transformer为例，记录解读wenet toolkit的训练流程。


## 3.1 train encode 代码outside

### 模型第一层：
模型层级：train.py--->asr_model.py--->encoder.py--->encoder_layer.py--->attention.py
asr_model.py、encoder.py、encoder_lyayer.py、attention.py的__init__函数，负责初始化模型类；forward函数，负责控制数据流向、计算。
### 模型第二层：
下面代码是asr_model.py的init_asr_model()函数，wenet模型主要由3大块组成，encoder(ConformerEncoder/TransformerEncoder)、decoder(TransformerDecoder/BiTransformerDecoder)、ctc：
```
    ctc = CTC(vocab_size, encoder.output_size())
    model = ASRModel(
        vocab_size=vocab_size,
        encoder=encoder,
        decoder=decoder,
        ctc=ctc,
        **configs['model_conf'],
    )
```
### 模型第三层：
下面代码是encoder.py的TransformerEncoder类的init函数()，它展示wenet的encoder结构。
首先，modulelist里面使用for循环生成结构一致的子结构---transformerencoderlayer,每一个子结构称为block。
然后，transformerencoderlayer由2大块组成，MultiHeadedAttention、PositionwiseFeedForward。PositionwiseFeedForward简单的全连接层，详细可以直接看源代码。
最后，如果选择Conformerencoderlayer作为block,而不是Transformerencoderlayer，子结构的子块可以从MultiHeadedAttention、RelPositionMultiHeadedAttention二者择其一。
```
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

### 模型第四层：
下面展示wenet的encoder_lyayer.py代码中transformer的实际计算过程，其实现与《attention is all you need》略有修改。
* wenet使用transformer作为encoder时，无需在input emb加入position emb。
* wenet构造encoder时，有self.normalize_before参数可选，前置layer_norm在残差之前，或者是后置layer_nromwenet example提供的yaml配置文件中，使用的是前置layer_norm，放置于残差residual之前。

```
# 按example配置文件简化后的执行顺序
# 第一次 layer norm
x = self.norm1(x)
# 计算attention, 然后第一次残差
x_q = x
x = residual + self.dropout(self.self_attn(x_q, x, x, mask))
# 第二次 layer norm
x = self.norm2(x)
# 计算feed_forward，然后第二次残差
x = residual + self.dropout(self.feed_forward(x))
```

![transformer](https://github.com/woqulrlr/wenet-learning/blob/main/transformer.jpg)

### 模型第五层：
下面展示wenet的attention.py代码中attention的实际计算过程。
```
#此处代码对应下面右图，数据变形成multi_head所需形式
#multi_head实际实现通过view改变数据shape，计算完成attention后，再通过view合并多头(attention is all you need中写作concat(head1,head2,...))
#数据切分成multi_head所需的shape.此处数据shape变化为[4，69，256]===>[4，69，（4，64）]
n_batch = query.size(0)
q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
q = q.transpose(1, 2)  # (batch, head, time1, d_k)
k = k.transpose(1, 2)  # (batch, head, time2, d_k)
v = v.transpose(1, 2)  # (batch, head, time2, d_k)

#此处代码对应左图Scaled Dot-Product Attention的计算
# query, key, value经过全连接投影得到q, k, v;李沐视频中解释，经过lineaer投影模型有更多的参数可以学习
q, k, v = self.forward_qkv(query, key, value)
# 计算scores，对应公式中的QK(T)
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

# 加入mask，因为训练过程，所以加入mask
mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
scores = scores.masked_fill(mask, -float('inf'))
# 计算softmax,对应公式的softmax
attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
# dropout
p_attn = self.dropout(attn)
# 对应公式dot V
x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
# multi_head合并，shape[4, 4, 69, 64]===>[4, 69, 256]
x = (x.transpose(1, 2).contiguous().view(n_batch, -1,self.h * self.d_k))  # (batch, time1, d_model)
```
![multi_head_attention](https://github.com/woqulrlr/wenet-learning/blob/main/multi_head_attention.jpg)
![multi_head_attention](https://github.com/woqulrlr/wenet-learning/blob/main/attention_formula.jpg)

***************************************************************
decode部分

attention&RNN/LSTM的比较

混合精度mixed precision training、多卡训练ddp、自动混合精度amp
***************************************************************

## 3.2 train decode 代码outside
模型层级：asr_model.py--->decode.py--->decode_layer.py--->attention.py
在asr_model.py代码主要步骤有以下3步，encoder部分已经在3.1说明。接下来对decoder和CTC branch部分进行解析和代码解读。
step1: encoder
step2a: decoder
step2b: CTC branch

### step2a: decoder
#### decode.py代码主要实现4件事情，
1.生成tgt_mask矩阵，此处生成的mask矩阵大小为[Batch，Len，Len],Len指训练音频样本对应的字符长度。
2.运算decode(x,tgt_mask,menory,memory_mask)，其中x指训练样本对应的字符串的embedding结果。
3.norm
4.linear，将decode结果从256，投影为vocab_size的大小。

#### decode_layer.py主要实现3个步骤，以伪代码的形式在下方写出,并给出《attention is all you need》图片对照参考。
在wenet中norm的执行顺序被前置了。
```
step1 : {
    norm
    attention(tgt_q,tgt,tgt,tgt_q_mask) #tgt_q = tgt = embed(字符)
    dropout + residual
},
step2 : {
    norm
    attention(x, memory, memory, memory_mask) #x=step1 attention的输出进行residual后的结果，memory=encode的结果
    dropout + residual
}
step3 : {
    norm
    feed_forward
    dropout + residual
}
```

![multi_head_attention](https://github.com/woqulrlr/wenet-learning/blob/main/transformer2.jpg)

#### attention.py已在3.1进行解析，不再次做解析。
完成step2a: decoder后，decode结果 与 训练样本对应的实际字符串计算loss，loss的计算方式使用LabelSmoothingLoss。如果decode_layer采用BiTransformerDecoder，则计算"从左往右decode"和"从右往左decode"两次解码，两次loss，两次loss根据权重进行加权求和为最终loss。权重为超参数。

### step2b: CTC branch主要进行两个计算
```
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
        # ys_hat: (B, L, D) -> (L, B, D)
        ys_hat = ys_hat.transpose(0, 1)
        ys_hat = ys_hat.log_softmax(2)
        loss = self.ctc_loss(ys_hat, ys_pad, hlens, ys_lens)
```

***************************************************************
mask矩阵逐个打开？

什么是LabelSmoothingLoss？

train 和 recognition，decode都是逐个预测的吗？

multi-attention的训练数据，是从什么维度进行切分的：multi-head输入attention时，feat的shape[batch,time,faet_dimention]，切分从feat_dimention
***************************************************************

# 4.识别recognize

## 4.1 attention

使用训练一致的encoder，
使用multi-head作decoder,
使用decode结果作为最后结果，
decode是逐帧解码的，逐帧解码与train的区别是：decode下一步的解码，都需要前一步block的解码结果，decode的输出作为query的输入（query、key、value）。

## 4.2 ctc_greedy_search


使用训练一致的encoder，
使用训练过的linear作decoder（此linear在训练时用ctc loss进行训练），
使用decode结果作为最后结果，
不是逐帧解码的，一次性得到encode对应的所有结果。

## 4.3 ctc_prefix_beam_search

同上4.2解码方式，附加一个beam search解码。

## 4.4 attention_rescoring

(1)encode
(2)CTC解码encode
(3)BiTransformer解码encode，CTC解码 + encode embedding作为输入。

# 5.模型抽取jit

# 6.部署deploy
