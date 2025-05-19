import torch
import torch.nn as nn
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import os
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import jieba


class CustomDataset(Dataset):
    def __init__(self,all_data):
        self.data = all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx][1]
        label = self.data[idx][0]
        return sample, label
def xuanmen_train(train_data_path,train_data_label_path,multi_head_bool,model_path,input_dim,emb_dim,hidden_dim,hand_model,seq_len,sampling_interval,n_layers,batch_size,train_epoch,num_heads=0):
    '''

    :param train_data_path:  训练数据集地址
    :param train_data_label_path:  数据集标签txt地址
    :return:
    '''


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 数据集参数
    data_dir = train_data_path  # 数据集源文件根目录
    max_length = seq_len//sampling_interval
    if hand_model:
        max_length = max_length//2

    end_token = (input_dim - 2)  # 源序列结束符号
    pad_token = (input_dim - 1)  # 源填充符号

    npy_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])
    labels = open(train_data_label_path,encoding='utf-8').read().splitlines()  # 假设每行是一个标签
    labels = [i.split()[1] for i in labels]
    labels = [i.replace('\ufeff', '') for i in labels]
    samples = [np.load(os.path.join(data_dir, f), allow_pickle=True) for f in npy_files]
    jiange = 5

    allall = []
    for i in samples:  # 便利每个类
        allall.append(np.array([np.array(j[::jiange]) for j in i],dtype=object))
    samples = np.array(allall,dtype=object)

    # 中文按字符分词（如需分词需修改为jieba等）
    tokenizer = lambda x: list(jieba.cut(x))

    # 构建词表（添加特殊标记）
    def yield_tokens(texts):
        for text in texts:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(
        yield_tokens(labels),
        specials=["<start>", "<pad>", "<unk>", "<end>"]
    )
    vocab.set_default_index(vocab["<unk>"])

    # 转换为序列并添加<end>标记
    sequences = [torch.tensor([vocab["<start>"]] + vocab(tokenizer(text)) + [vocab["<end>"]]) for text in labels]

    # 统一填充长度（填充<pad>）
    padded_sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=vocab["<pad>"]
    )
    idx2word = vocab.get_itos()

    all_data = []
    for i in range(len(samples)):
        input_seq = padded_sequences[i]  # 获取对应的输入序列

        for j in samples[i]:
            # 将j转为tensor（如果不是）
            j_tensor = torch.tensor(j) if not isinstance(j, torch.Tensor) else j.clone().detach()

            # 1. 先添加结束符41（计入1500长度内）
            j_with_end = torch.cat([j_tensor, torch.tensor([end_token], dtype=j_tensor.dtype)])

            # 2. 处理长度
            if len(j_with_end) > max_length:
                # 如果超长：截断到1499再加结束符
                j_processed = torch.cat([j_with_end[:max_length - 1],
                                         torch.tensor([end_token], dtype=j_tensor.dtype)])
            elif len(j_with_end) < max_length:
                pad_needed = max_length - len(j_with_end)
                padding = torch.full((pad_needed,), pad_token, dtype=j_tensor.dtype)
                j_processed = torch.cat([j_with_end, padding])
            else:
                # 刚好1500
                j_processed = j_with_end

            # 验证长度
            assert len(j_processed) == max_length, f"长度错误：{len(j_processed)} != {max_length}"

            # 添加到最终数据
            all_data.append([input_seq, j_processed])

    print(f'词表大小：{len(vocab.get_stoi())}')

    output_dim = len(vocab.get_stoi())  # 输出词汇表大小（需你确认）

    dataset = CustomDataset(all_data)
    # 定义划分比例（例如80%训练，20%测试）
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # 随机划分
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子确保可复现
    )

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    print(output_dim,hidden_dim,input_dim,emb_dim)
    if multi_head_bool:
        model = Seq2SeqWithMultiHeadAttention(
            input_dim=input_dim,  # 输入词汇表大小
            emb_dim=emb_dim,  # 词向量维度
            hidden_dim=hidden_dim,  # LSTM隐藏层维度
            output_dim=output_dim,  # 输出词汇表大小（需你确认）
            n_layers=2,
            num_heads=8
        ).to(device)
    else:
        model = seq2seq(
            input_dim=input_dim,  # 输入词汇表大小
            emb_dim=emb_dim,  # 词向量维度
            hidden_dim=hidden_dim,  # LSTM隐藏层维度
            output_dim=output_dim,  # 输出词汇表大小（需你确认）
            n_layers=n_layers
        ).to(device)

    print(model)

    train(
        model=model,
        train_loader=train_loader,  # 你的DataLoader
        test_loader=test_loader,
        optimizer=optim.Adam(model.parameters(), lr=0.001),
        criterion=CrossEntropyLoss(ignore_index=1),  # 假设填充符index=1
        # criterion=CrossEntropyLoss(),
        epochs=train_epoch,
        device=device,
        max_bleu=0,
        output_dim = output_dim,
        save_path=model_path,
        test_bool=True,
        train_bool=True
    )


def train(model, train_loader, test_loader,optimizer, criterion, epochs, device, max_bleu,output_dim,save_path,test_bool=False, train_bool=True):
    model.train()
    model.to(device)
    max_bleu = max_bleu
    for epoch in range(epochs):
        epoch_loss = 0
        if train_bool:
            for batch_idx, (src, trg) in enumerate(train_loader):
                src = src.to(device)  # [batch_size, 424]
                trg = trg.to(device)  # [batch_size, 10]

                optimizer.zero_grad()

                # 前向传播（模型自动处理teacher forcing）
                output = model(src, trg)  # [batch_size, 10, output_dim]

                # 计算损失（忽略<sos>和padding）
                output = output[:, 1:].reshape(-1, output_dim)  # 忽略<sos>，形状变为[batch_size*9, output_dim]
                trg = trg[:, 1:].reshape(-1)  # 忽略<sos>，形状变为[batch_size*9]
                loss = criterion(output, trg)
                # print(output)
                # print(output.size())
                # print(trg)
                # print(trg.size())
                # return 0

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()

                if batch_idx % 10 == 0:
                    print(f'Epoch: {epoch + 1:03d} | Batch: {batch_idx:03d} | Loss: {loss.item():.4f}')
        test_num = 0
        test_bleu = float(0)
        test_loss = 0
        test_jingque = 0
        test_zhaohui = 0
        test_f1 = 0
        if test_bool:
            for batch_idx, (src, trg) in enumerate(test_loader):
                # print(src)
                # print(trg)
                src = src.to(device)
                trg = trg.to('cpu').tolist()
                # print('0000000000000',src)
                output = model.predict(src)
                # print(output.size())
                output = output.tolist()
                for i in range(len(output)):
                    out = [0] + output[i][:8]
                    # print(out)
                    tlist = [trg[i]]
                    # print(tlist)
                    score = sentence_bleu(tlist, out, weights=(0.5, 0.5))
                    test_bleu += score
                    test_num += 1
        print(f'Epoch: {epoch + 1:03d} | Avg Loss: {epoch_loss / len(train_loader):.4f}')
        if test_bool:
            # print(f'test bleu = {test_bleu/test_num},精确率: {(test_jingque/test_num):.4f}, 召回率: {(test_zhaohui/test_num):.4f}, F1: {(test_f1/test_num):.4f}')
            if (test_bleu / test_num) > max_bleu:
                max_bleu = (test_bleu / test_num)

                parent_dir = os.path.dirname(save_path)

                # 如果父目录不存在，则创建
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)  # exist_ok=True 避免目录已存在时报错
                    print(f"父目录已创建：{parent_dir}")
                else:
                    print(f"父目录已存在或无需创建：{parent_dir}")
                torch.save(model, save_path)
        print(f'max_bleu={max_bleu}')


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
                                                                self.hidden_dim)  # [batch_size, 1, hidden_dim(head_dim*num_heads)]
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

