本篇主要内容是wenet模型的代码笔记、工程向的阅读笔记
源码地址：https://github.com/wenet-e2e/wenet

# 1.* 数据采集

# 2.* 数据处理

wenet数据的处理部分可以分成两部分。一是做inference/recognize时的数据处理流程。另一个是train时的数据处理流程。

## 2.1 recognize

## 2.2.1 train-dataloader
## 2.2.2 train-Collate_fn
Collate_fn是pytorch的dataloader数据导入模块中的数据处理函数。用户可以使用自定义的collate_fn来实现自定义数据批处理。批处理后的数据将作为dataloader的输出。wenet的Collate_fn详细见于，wenet--->dataset--->datset.py--->CollateFunc--->__call__。下面是CollateFunc的数据处理实现：
```
_extract_feature
    _load_wav_with_speed(speed_perturb,optinoal)
    _waveform_distortion
feature_dither(optinoal)
_spec_substitute(optinoal)
_spec_augmentation(optinoal)
padding
```

#### _load_wav_with_speed（speed_perturb）
使用pytorch的sox的接口进行音频数据增强，实现音频变速。
更多具体内容可查看https://pytorch.org/audio/stable/sox_effects.html。

#### _waveform_distortion
通过自定义的函数实现各种失真模拟。模拟的种类有：
```
'gain_db'
'max_distortion'
'fence_distortion'
'jag_distortion'
'poly_distortion'
'quad_distortion'
'none_distortion'
```

#### feature_dither
```
a = random.uniform(0, self.feature_dither)
xs = [x + (np.random.random_sample(x.shape) - 0.5) * a for x in xs]
```
xs是一个batch,x是一个sample。
np.random.random_sample生成一个[0,1)的随机数。x.shape是x即sample的长度。
x + (np.random.random_sample(x.shape) - 0.5) * a ： 生成范围在([0,1) - 0.5) * a的随机数，数量与x.shape一致，并将随机数与x的每一个元素，一一对应相加(类似element wise)。

#### _spec_substitute
或将频谱图，音频整段后移。如0-30frame的音频，截取10-20frame,替换15-25frame的音频。

#### * _spec_augmentation
对频谱图进行随机修改，将某段频谱图置0。(未完整)

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

```
# ctc_probs概率矩阵，概率矩阵shape[ctc解码字符长度，vocabulary]，
# 例：torch.Size([148, 4233])
ctc_probs = self.ctc.log_softmax(encoder_out)

# cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
# prefix：完成beam_search输出的字符；blank_ending_score：空/停顿的概率；none_blank_ending_score：非空/非停顿的概率
cur_hyps = [(tuple(), (0.0, -float('inf')))]

# 2. CTC beam search step by step
for t in range(0, maxlen):# 例 maxlen = 148
    logp = ctc_probs[t]  # (vocab_size,),torch.Size([4233]),每个字的概率
    
    # key: prefix, value (pb, pnb), default value(-inf, -inf)
    # pb：prob of blank；pnb：prob of no blank
    
    #例： {(2995,): (-0.00016307625787703728, -inf), (70,): (-10.18041968259945, -inf), (2995, 2553): (-10.916154154024202, -inf), (254,): (-11.645881483306317, -inf), (2995, 254): (-11.696770409974455, -inf)}
    # (2995,254，xxx,xxx)字符串
    # (-11.696770409974455, -inf)，第一个是n_pb, 第二个是n_pnb
    next_hyps = defaultdict(lambda: (-float('inf'), -float('inf'))) 
    
    
    # 2.1 First beam prune: select topk best
    top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)，选top5;top_k_logp 概率，top_k_index 字
    for s in top_k_index:#循环top5的字
        s = s.item()# 字
        ps = logp[s].item()# 概率
        for prefix, (pb, pnb) in cur_hyps:
            last = prefix[-1] if len(prefix) > 0 else None
            if s == 0:  # blank
                n_pb, n_pnb = next_hyps[prefix]
                n_pb = log_add([n_pb, pb + ps, pnb + ps])
                next_hyps[prefix] = (n_pb, n_pnb)
            elif s == last:
                #  Update *ss -> *s;
                n_pb, n_pnb = next_hyps[prefix]
                n_pnb = log_add([n_pnb, pnb + ps])
                next_hyps[prefix] = (n_pb, n_pnb)
                # Update *s-s -> *ss, - is for blank
                n_prefix = prefix + (s, )
                n_pb, n_pnb = next_hyps[n_prefix]
                n_pnb = log_add([n_pnb, pb + ps])
                next_hyps[n_prefix] = (n_pb, n_pnb)
            else:
                n_prefix = prefix + (s, )
                n_pb, n_pnb = next_hyps[n_prefix] #产生默认值（-inf,inf）,next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
                n_pnb = log_add([n_pnb, pb + ps, pnb + ps]) #>??????,为什么这样更新
                next_hyps[n_prefix] = (n_pb, n_pnb) #更新next_porb_no_blank的概率值； 不更新next_prob_blank

    # 2.2 Second beam prune
    next_hyps = sorted(next_hyps.items(),
                       key=lambda x: log_add(list(x[1])),
                       reverse=True)
    cur_hyps = next_hyps[:beam_size]
hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]
return hyps, encoder_out
```

## 4.4 attention_rescoring
```
(1)encode

(2)CTC解码encode

(3)BiTransformer解码encode，CTC解码 + encode embedding作为输入。

(4)score计算
    1. for 5句CTC
    2. for 每句CTC的charlist
        3. left to right transformer, for 每句CTC的charlist的，每个字 score_left_to_right
        4. right to left transformer, for 每句CTC的charlist的，每个字 score_right_to_left
    5. weight*score_left_to_right + weight*score_right_to_left
    6. CTC句子级score + (weight*score_left_to_right + weight*score_right_to_left)
```

## 4.5 shallow fusion LM

https://zhuanlan.zhihu.com/p/74696938

https://bbs.huaweicloud.com/blogs/269842

# 5.模型抽取jit

# 6.部署deploy
