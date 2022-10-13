# 3.训练train

wenet toolkit进行训练流程时，需为train.py使用时配置"--config xxx.yaml"，可以选择transformer、conformer等不同的模型结构。本片笔记以transformer为例，记录解读wenet toolkit的训练流程。

## 3.1 初始化总体模型框架
在train.py的169行调用init_asr_model()初始化模型，调用模型的实际操作是在wenet.transformer.asr_model完成。
```
from wenet.transformer.asr_model import init_asr_model
# train.py ---> line 169
# Init asr model from configs
model = init_asr_model(configs)
```

## 3.2 初始化总体模型细节

#### init()
在asr_model.py文件,class ASRModel()类,__init__()方法,初始化模型总体框架。模型由encoder,decoder,ctc,criterion_att四个部分组成。
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

#### init()
下面代码是encoder.py的TransformerEncoder类的init函数()，它展示wenet的encoder结构。

首先，最外层是一个List,modulelist里面使用for循环生成结构一致的子结构---transformerencoderlayer,每一个子结构称为block。

然后，transformerencoderlayer由2大块组成，MultiHeadedAttention、PositionwiseFeedForward。PositionwiseFeedForward简单的全连接层，详细可以直接看源代码。

最后，如果选择Conformerencoderlayer作为block,而不是Transformerencoderlayer，子结构的子块可以从MultiHeadedAttention、RelPositionMultiHeadedAttention二者择其一。

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
1.mask待补充
2.跑完6个block
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