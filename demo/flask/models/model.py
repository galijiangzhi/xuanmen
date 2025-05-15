import torch.nn as nn
import torch
import torch.nn.functional as F
class seq2seq(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        self.encode_embedding = nn.Embedding(input_dim, emb_dim)  # 将每个词扩充为emb_dim维
        self.decode_embedding = nn.Embedding(output_dim, emb_dim)
        self.encode = nn.LSTM(emb_dim, hidden_dim, n_layers)
        self.decode = nn.LSTM(emb_dim, hidden_dim, n_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, tar):
        # src: [batch_size, src_len]
        # tar: [batch_size, trg_len]

        # 编码器部分
        encode_embedded = self.encode_embedding(src)  # [batch_size, src_len, emb_dim]
        encode_embedded = encode_embedded.permute(1, 0, 2)  # [src_len, batch_size, emb_dim]
        _, (hidden, cell) = self.encode(encode_embedded)

        # 解码器部分
        batch_size = tar.shape[0]  # 3
        trg_len = tar.shape[1]  # 9
        output_dim = self.fc.out_features  # 181
        # print(output_dim)

        # 准备输出张量
        outputs = torch.zeros(trg_len, batch_size, output_dim).to(src.device)  # 9x3x181

        # 初始输入是<sos> token，这里假设tar已经包含<sos>作为第一个token
        input = tar[:, 0]  # 取第一个token作为初始输入 [batch_size]

        for t in range(1, trg_len):
            # 嵌入输入
            embedded = self.decode_embedding(input).unsqueeze(0)  # [1, batch_size, emb_dim]
            # print(f'embedded:{embedded.size()}')
            # print(f'hidden:{hidden.size()}')
            # 通过解码器
            output, (hidden, cell) = self.decode(embedded, (hidden, cell))

            # 预测下一个token
            pred = self.fc(output.squeeze(0))
            outputs[t] = pred

            # 下一个输入是真实目标(teacher forcing)或预测结果
            # 这里使用teacher forcing，传入真实目标
            input = tar[:, t]

        return outputs.permute(1, 0, 2)  # [batch_size, trg_len, output_dim]

    def predict(self, src, sos_token_idx=0, eos_token_idx=1, max_len=9):
        """
        自回归预测（不需要输入tar）
        :param src: 输入序列 [batch_size, src_len]
        :param sos_token_idx: <sos>的索引
        :param eos_token_idx: <eos>的索引（可选）
        :param max_len: 最大生成长度
        :return: 预测序列 [batch_size, max_len]
        """
        # 编码器部分
        encode_embedded = self.encode_embedding(src).permute(1, 0, 2)
        _, (hidden, cell) = self.encode(encode_embedded)

        # 解码器初始化
        batch_size = src.size(0)
        outputs = torch.zeros(batch_size, max_len).long().to(src.device)
        input = torch.full((batch_size,), sos_token_idx, dtype=torch.long).to(src.device)

        # 自回归解码
        for t in range(max_len):
            embedded = self.decode_embedding(input).unsqueeze(0)  # [1, batch_size, emb_dim]
            output, (hidden, cell) = self.decode(embedded, (hidden, cell))
            pred = self.fc(output.squeeze(0)).argmax(-1)  # [batch_size]

            outputs[:, t] = pred
            input = pred  # 使用预测结果作为下一输入

            # 如果所有序列都生成<eos>则提前停止
            if eos_token_idx is not None and (pred == eos_token_idx).all():
                break

        return outputs


class Seq2SeqWithMultiHeadAttention(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, output_dim, n_layers, num_heads):
        super(Seq2SeqWithMultiHeadAttention, self).__init__()
        self.encode_embedding = nn.Embedding(input_dim, emb_dim)  # 输入嵌入层
        self.decode_embedding = nn.Embedding(output_dim, emb_dim)  # 输出嵌入层
        self.encoder = nn.LSTM(emb_dim, hidden_dim, n_layers, batch_first=False)
        self.decoder = nn.LSTM(emb_dim + hidden_dim, hidden_dim, n_layers, batch_first=False)

        # 多头注意力相关参数
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by the number of heads"
        self.head_dim = hidden_dim // num_heads

        # 注意力权重矩阵
        self.W_q = nn.Linear(hidden_dim, hidden_dim)  # Query 线性变换
        self.W_k = nn.Linear(hidden_dim, hidden_dim)  # Key 线性变换
        self.W_v = nn.Linear(hidden_dim, hidden_dim)  # Value 线性变换
        self.fc = nn.Linear(hidden_dim, output_dim)

    def attention(self, hidden, encoder_outputs):
        """
        多头注意力机制
        :param hidden: 解码器的隐藏状态 [batch_size, hidden_dim]
        :param encoder_outputs: 编码器的所有隐藏状态 [src_len, batch_size, hidden_dim]
        :return: 上下文向量 [batch_size, hidden_dim], 注意力权重 [batch_size, src_len]
        """
        batch_size = hidden.size(0)
        src_len = encoder_outputs.size(0)

        # 将编码器输出和解码器隐藏状态映射到 Q、K、V
        Q = self.W_q(hidden).view(batch_size, 1, self.num_heads, self.head_dim).permute(0, 2, 1,
                                                                                        3)  # [batch_size, num_heads, 1, head_dim]
        K = self.W_k(encoder_outputs.permute(1, 0, 2)).view(batch_size, src_len, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3)  # [batch_size, num_heads, src_len, head_dim]
        V = self.W_v(encoder_outputs.permute(1, 0, 2)).view(batch_size, src_len, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3)  # [batch_size, num_heads, src_len, head_dim]

        # 计算点积注意力分数
        energy = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch_size, num_heads, 1, src_len]
        attn_weights = F.softmax(energy, dim=-1)  # [batch_size, num_heads, 1, src_len]

        # 加权求和得到上下文向量
        context = torch.matmul(attn_weights, V)  # [batch_size, num_heads, 1, head_dim]
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, 1,
                                                                self.hidden_dim)  # [batch_size, 1, hidden_dim]
        context = context.squeeze(1)  # [batch_size, hidden_dim]

        return context, attn_weights

    def forward(self, src, tar):
        # src: [batch_size, src_len]
        # tar: [batch_size, trg_len]

        # 编码器部分
        encode_embedded = self.encode_embedding(src)  # [batch_size, src_len, emb_dim]
        encode_embedded = encode_embedded.permute(1, 0, 2)  # [src_len, batch_size, emb_dim]
        encoder_outputs, (hidden, cell) = self.encoder(
            encode_embedded)  # encoder_outputs: [src_len, batch_size, hidden_dim]

        # 解码器部分
        batch_size = tar.shape[0]
        trg_len = tar.shape[1]
        output_dim = self.fc.out_features

        # 准备输出张量
        outputs = torch.zeros(trg_len, batch_size, output_dim).to(src.device)  # [trg_len, batch_size, output_dim]

        # 初始输入是<sos> token
        input = tar[:, 0]  # 取第一个token作为初始输入 [batch_size]

        for t in range(1, trg_len):
            # 嵌入输入
            embedded = self.decode_embedding(input).unsqueeze(0)  # [1, batch_size, emb_dim]

            # 计算多头注意力上下文向量
            context, _ = self.attention(hidden[-1], encoder_outputs)  # context: [batch_size, hidden_dim]

            # 将上下文向量与嵌入输入拼接
            rnn_input = torch.cat((embedded, context.unsqueeze(0)), dim=2)  # [1, batch_size, emb_dim + hidden_dim]

            # 通过解码器
            output, (hidden, cell) = self.decoder(rnn_input, (hidden, cell))

            # 预测下一个token
            pred = self.fc(output.squeeze(0))  # [batch_size, output_dim]
            outputs[t] = pred

            # 下一个输入是真实目标(teacher forcing)或预测结果
            input = tar[:, t]  # 使用真实目标

        return outputs.permute(1, 0, 2)  # [batch_size, trg_len, output_dim]

    def predict(self, src, sos_token_idx=0, eos_token_idx=1, max_len=9):
        """
        自回归预测（不需要输入tar）
        :param src: 输入序列 [batch_size, src_len]
        :param sos_token_idx: <sos>的索引
        :param eos_token_idx: <eos>的索引（可选）
        :param max_len: 最大生成长度
        :return: 预测序列 [batch_size, max_len]
        """
        # 编码器部分
        encode_embedded = self.encode_embedding(src).permute(1, 0, 2)  # [src_len, batch_size, emb_dim]
        encoder_outputs, (hidden, cell) = self.encoder(
            encode_embedded)  # encoder_outputs: [src_len, batch_size, hidden_dim]

        # 解码器初始化
        batch_size = src.size(0)
        outputs = torch.zeros(batch_size, max_len).long().to(src.device)
        input = torch.full((batch_size,), sos_token_idx, dtype=torch.long).to(src.device)

        for t in range(max_len):
            # 嵌入输入
            embedded = self.decode_embedding(input).unsqueeze(0)  # [1, batch_size, emb_dim]

            # 计算多头注意力上下文向量
            context, _ = self.attention(hidden[-1], encoder_outputs)  # context: [batch_size, hidden_dim]

            # 将上下文向量与嵌入输入拼接
            rnn_input = torch.cat((embedded, context.unsqueeze(0)), dim=2)  # [1, batch_size, emb_dim + hidden_dim]

            # 通过解码器
            output, (hidden, cell) = self.decoder(rnn_input, (hidden, cell))

            # 预测下一个token
            pred = self.fc(output.squeeze(0)).argmax(-1)  # [batch_size]

            outputs[:, t] = pred
            input = pred  # 使用预测结果作为下一输入

            # 如果所有序列都生成<eos>则提前停止
            if eos_token_idx is not None and (pred == eos_token_idx).all():
                break

        return outputs