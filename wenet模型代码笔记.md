本篇主要内容是wenet模型的代码笔记、工程向的阅读笔记

# 1.* 数据采集

# 2.* 数据处理

# 3.训练train

wenet toolkit进行训练流程时，需为train.py使用时配置"--config xxx.yaml"，可以选择transformer、conformer等不同的模型结构。
本片笔记以transformer为例，记录解读wenet toolkit的训练流程。


## 3.1 train代码outside

### 模型第一层：
模型层级：train.py--->asr_model.py--->encoder.py--->encoder_lyayer.py--->attention.py
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
然后，transformerencoderlayer由2大块组成，MultiHeadedAttention、PositionwiseFeedForward。
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
#数据切分成multi_head所需的shape.此处数据shape变化为4，69，256===>4，69，（4，64）
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
return self.forward_attention(v, scores, mask)
```
![multi_head_attention](https://github.com/woqulrlr/wenet-learning/blob/main/multi_head_attention.jpg)
![multi_head_attention](https://github.com/woqulrlr/wenet-learning/blob/main/attention_formula.jpg)


PositionwiseFeedForward，解释一下




multiple-attention的计算&代码实现

transformer

attention&RNN/LSTM的比较

混合精度mixed precision training、多卡训练ddp、自动混合精度amp




# 4.识别recognize

# 5.模型抽取jit

# 6.部署deploy
