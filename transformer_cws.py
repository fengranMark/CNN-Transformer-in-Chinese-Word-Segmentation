#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re, sys, os, codecs
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
sys.path.append('../code')
import score
import time

if torch.cuda.is_available():
    torch.cuda.manual_seed(812)
    torch.manual_seed(812)
    device = torch.device('cuda')
else:
    torch.manual_seed(812)
    device = torch.device('cpu')


config = {'hidden_dimensions':200,
          'embedding_dimensions':100, 
          'learning_rate':0.001, 
          'keep_dropout':0.2, 
          'emb_dropout':0.2,
          'hidden_layers':4, 
          'n_heads':4,
          'k_dims':50, # d_model / heads
          'v_dims':50,
          'kernel':3,
          'pretrained':False, 
          'shuffle':True, 
          'epochs':50, 
          'batch_size':32, 
          'avg_bathch':False, 
          'tagset_size':4, 
          'CRF':True, 
          'max_len':60, 
          'bi_gram':True, 
          'bi_embedding_dimensions':100,
          'special_token_file':'../data/pku/specialToken.txt', 
          'train_file':'../data/pku/pku_train_file.txt', 
          'valid_file':'../data/pku/pku_valid_file.txt', 
          'dict_file':'../data/pku/pku_udict.txt', 
          'bi_dict_file':'../data/pku/pku_bdict.txt', 
          'model_path':'../data/pku/model/',              
          'test_file':'../data/pku/pku_test_file.txt', 
          'file_for_test':'../data/medical/pku_test_input.txt',
          'test_model':'', 
          'update_model':'',
          'module':'train', 
          'output_file':'../data/pku/pku_output.txt'
     }

class Dataset(Dataset):
    def __init__(self, x_, y_, bi_, pos_):
        self.x_data = x_
        self.y_data = y_
        self.bi_data = bi_
        self.len = len(x_)
        self.pos_data = pos_
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.bi_data[index], self.pos_data[index]

    def __len__(self):
        return self.len
        
def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    inputs = list()
    labels = list()
    seq_len = list()
    bi_grams = list()
    pos_id = list()
    for key in data:
        x, y, z, pos = key
        inputs.append(x)
        labels.append(y)
        seq_len.append(len(x))
        bi_grams.append(z)
        pos_id.append(pos)
    datax = pad_sequence([torch.from_numpy(np.array(x)) for x in inputs], batch_first=True, padding_value=0).long()
    datay = pad_sequence([torch.from_numpy(np.array(y)) for y in labels], batch_first=True, padding_value=0).long()
    dataz = pad_sequence([torch.from_numpy(np.array(z)) for z in bi_grams], batch_first=True, padding_value=0).long()
    datapos = pad_sequence([torch.from_numpy(np.array(pos)) for pos in pos_id], batch_first=True, padding_value=0).long()
    data =  [datax, datay, dataz, datapos, seq_len]
    return data
    
def word2label(words, ntag):
    labels = []
    if ntag == 4:
        for word in words:
            if len(word) == 1:
                labels.append(4)
            elif len(word) == 2:
                labels.append(1)
                labels.append(2)
            else:
                labels.append(1)
                n2 = len(word)-2
                labels = labels + [3] * n2
                labels.append(2)
    elif ntag == 2:
        for word in words:
            if len(word) == 1:
                labels.append(2)
            else:
                labels = labels + [1] * (len(word)-1)
                labels.append(2)
    return labels

def getdata(data, ntag=4, maxlen=60, specialToken=[]):
    sents = list()
    labels = list()
    for sent in data:
        Left = 0
        sent = sent.split()
        for idx,word in enumerate(sent):
            if word not in specialToken:
                if len(re.sub('\W','',word,flags=re.U))==0:
                    if idx > Left:
                        slen = len(list(''.join(sent[Left:idx])))
                        if slen <= maxlen:
                            sents.append(list(''.join(sent[Left:idx])))
                            labels.append(word2label(sent[Left:idx], ntag))
                    Left = idx+1
        if Left!=len(sent):
            slen = len(list(''.join(sent[Left:])))
            if slen <= maxlen:
                sents.append(list(''.join(sent[Left:])))
                labels.append(word2label(sent[Left:], ntag))
    return sents, labels

def splitdata(data, specialToken=[]):
    sents = list()
    #print (specialToken)
    for sent in data:
        Left = 0
        sent = list(sent)
        for idx,c in enumerate(sent):
            if c not in specialToken:
                if len(re.sub('\W','',c,flags=re.U))==0:
                    if idx > Left:
                        sents.append(list(''.join(sent[Left:idx])))
                        sents.append(c)
                    else:
                        sents.append(c)
                    Left = idx+1
        if Left!=len(sent):
            sents.append(list(''.join(sent[Left:])))
        sents.append('\n')
    return sents
    
def prepareData(config):
    #uni-gram or bi-gram model data
    with open(config['special_token_file'], 'r', encoding='utf8') as f:
        specialToken = f.read().splitlines()
    if config['module'] == 'train':
        with open(config['train_file'], 'r', encoding='utf8') as f:
            data = f.read().splitlines()
        sents, labels = getdata(data, ntag=config['tagset_size'], specialToken=specialToken)
    elif config['module'] == 'test':
        with open(config['file_for_test'], 'r', encoding='utf8') as f:
            data = f.read().splitlines()
        sents = splitdata(data, specialToken=specialToken)
    udict = dict()
    with open(config['dict_file'], 'r', encoding='utf8') as f:
        data = f.read().splitlines()
    for key in data:
        keyL = key.split('\t')
        udict[keyL[0]] = int(keyL[1])
    config['vocab_size'] = len(udict)+1 #0 padding mask
    bdict = dict()
    if config['bi_gram']:
        with open(config['bi_dict_file'], 'r', encoding='utf8') as f:
            data = f.read().splitlines()
        for key in data:
            keyL = key.split('\t')
            bdict[keyL[0]] = int(keyL[1])
        config['bi_vocab_size'] = len(bdict)+1 #0   
    else:
        config['bi_vocab_size'] = 0
    idx = list()
    bi_idx = list()
    pos_idx = list()
    for sent in sents:
        idxs = list()
        bi_idxs = list()
        pos_idxs = list()
        if type(sent) != list:
            continue
        for index, c in enumerate(sent):
            if c in udict:
                idxs.append(udict[c])
            else:
                idxs.append(udict['U']) 
            pos_idxs.append(index)
            if config['bi_gram']:
                if index == len(sent)-1:
                    bw = c + 'E'
                else:
                    bw = c + sent[index+1]
                if bw in bdict:
                    bi_idxs.append(bdict[bw])
                else:
                    bi_idxs.append(bdict['U'])
            else:
                pass
        idx.append(idxs)
        bi_idx.append(bi_idxs)
        pos_idx.append(pos_idxs)
    if config['module'] == 'train':
        train_data = Dataset(idx, labels, bi_idx, pos_idx)
        train_data_loader = DataLoader(dataset=train_data, batch_size=config['batch_size'], shuffle=config['shuffle'], collate_fn=collate_fn)
        return train_data_loader
    elif config['module'] == 'test':
        return sents, idx, bi_idx, pos_idx
      
def write(data, filename):
    with open(filename, 'w', encoding='utf8') as f:
        for sent in data:
            for word in sent:
                f.write(word + ' ')
            f.write('\n')
    print ('the file has been writen successfully! ')
    
def repairResult(sents, results):
    new_results = list()
    i = 0
    j = 0
    word_result = list()
    while i < len(sents):
        if type(sents[i]) != list:
            if sents[i] == '\n':
                new_results.append(word_result)
                word_result = list()
            else:
                word_result.append(sents[i])
            i += 1
        elif len(sents[i]) != len(results[j]):
            s = ''
            for c in sents[i]:
                s = s + c
            print ('the result does not match with the sent of ' + str(s))
            return -1
        else:
            word = ''
            for k in range(len(sents[i])):
                if results[j][k] == 4:
                    word_result.append(sents[i][k])
                    word = ''
                elif results[j][k] == 1:
                    if word != '':
                        word_result.append(word)
                        word = ''
                    word = sents[i][k]
                elif results[j][k] == 2:
                    word += sents[i][k]
                    word_result.append(word)
                    word = ''
                elif results[j][k] == 3:
                    word += sents[i][k]
                else:
                    print ('the result has a wrong tag')
                    continue
                    return -1
            if word != '':
                word_result.append(word)
                word = ''
            #new_results.append(word_result)
            i += 1
            j += 1
    if len(word_result) != 0:
        new_results.append(word_result)
    #output_results = resumeResult(new_results, split_list, continuous)
    return new_results

def load_dense_drop_repeat(inputs):
    vocab_size, size = 0, 0
    vocab = {}
    vocab["i2w"], vocab["w2i"] = [], {}
    count = 0
    with codecs.open(inputs, "r", "utf-8") as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                vocab_size = int(line.strip().split()[0])
                size = int(line.rstrip().split()[1])
                matrix = np.zeros(shape=(vocab_size, size), dtype=np.float32)
                continue
            vec = line.strip().split()
            if not vocab["w2i"].__contains__(vec[0]):
                vocab["w2i"][vec[0]] = count
                matrix[count, :] = np.array([float(x) for x in vec[1:]])
                count += 1
    for w, i in vocab["w2i"].items():
        vocab["i2w"].append(w)
    return matrix

#matrix_word = load_dense_drop_repeat("../data/sgns.renmin.char")
#matrix_bigram = load_dense_drop_repeat("../data/sgns.renmin.bigram")
#config['unigram_prevocab'] = len(matrix_word)
#config['bigram_prevocab'] = len(matrix_bigram)
#print("embedding load finish!")


# In[2]:


class TextSlfAttnNet(nn.Module):
    ''' 自注意力模型 '''

    def __init__(self, config):
        super(TextSlfAttnNet, self).__init__()
        self.embedding_drop = nn.Dropout(p = config['emb_dropout'], inplace=True)
        # unigram向量
        if config['pretrained']:
            self.word_embedding = nn.Embedding(config['unigram_prevocab'], config['embedding_dimensions'])
            self.word_embedding.weight.data.copy_(torch.from_numpy(matrix_word))
            self.word_embedding.weight.requires_grad = True
        else:
            self.word_embedding = nn.Embedding(config['vocab_size'], config['embedding_dimensions'])
        # bigram向量
        if config['bi_gram']:
            if config['pretrained']:
                self.bi_embedding = nn.Embedding(config['bigram_prevocab'], config['bi_embedding_dimensions'])
                self.bi_embedding.weight.data.copy_(torch.from_numpy(matrix_bigram))
                self.bi_embedding.weight.requires_grad = True
                config['embedding_dimensions'] = config['embedding_dimensions'] + config['bi_embedding_dimensions']
            else:
                self.bi_embedding = nn.Embedding(config['bi_vocab_size'], config['embedding_dimensions'])
                config['embedding_dimensions'] = config['embedding_dimensions'] + config['bi_embedding_dimensions']
        else:
            self.bi_embedding = None
            config['embedding_dimensions'] = config['embedding_dimensions']
        self.char_encode = CNNChar(config['embedding_dimensions'] // 2, config['embedding_dimensions'] // 2, config['kernel'], dropout=config['emb_dropout'])
        # 位置向量
        self.pos_embedding = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(config['max_len'], config['embedding_dimensions'], padding_idx=0),
            freeze=True)
        # 多个编码层
        self.layer_stack = nn.ModuleList([
            EncoderLayer(config['embedding_dimensions'], config['hidden_dimensions'], config['n_heads'], config['k_dims'], config['v_dims'], dropout=config['keep_dropout'])
            for _ in range(config['hidden_layers'])
        ])
        self.crf = config['CRF']       
        self.average_batch = config['avg_bathch']
        if self.crf:
            #self.hidden2tag = nn.Linear(config['embedding_dimensions'], config['tagset_size']+3)
            self.fc_out = nn.Sequential(
            nn.Dropout(config['keep_dropout']),
            nn.Linear(config['embedding_dimensions'], config['hidden_dimensions']),
            nn.ReLU(inplace=True),
            nn.Dropout(config['keep_dropout']),
            nn.Linear(config['hidden_dimensions'], config['tagset_size']+3),
        )
            self.START_TAG_IDX, self.END_TAG_IDX = -2, -1
            init_transitions = torch.randn(config['tagset_size']+3, config['tagset_size']+3)
            init_transitions[:, self.START_TAG_IDX] = -10000.
            init_transitions[self.END_TAG_IDX, :] = -10000.
            self.transitions = nn.Parameter(init_transitions)
        else:
            self.fc_out = nn.Sequential(
            nn.Dropout(config['keep_dropout']),
            nn.Linear(config['embedding_dimensions'], config['hidden_dimensions']),
            nn.ReLU(inplace=True),
            nn.Dropout(config['keep_dropout']),
            nn.Linear(config['hidden_dimensions'], config['tagset_size']+3),
        )
            #self.hidden2tag = nn.Linear(config['embedding_dimensions'], config['tagset_size']+1)
        #fc_out's input [Batch_size, seq_len, d_model
        
    def log_sum_exp(self, vec, m_size):
        _, idx = torch.max(vec, 1)
        max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)
        return max_score.view(-1, m_size) + torch.log(torch.sum(
                torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)
    
    def _forward_alg(self, feats, mask=None):
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(-1)

        mask = mask.transpose(1, 0).contiguous()
        ins_num = batch_size * seq_len
        feats = feats.transpose(1, 0).contiguous().view(
            ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        scores = feats + self.transitions.view(
            1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        seq_iter = enumerate(scores)
        _, inivalues = seq_iter.__next__()
        partition = inivalues[:, self.START_TAG_IDX, :].clone().view(batch_size, tag_size, 1)
        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(
                batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            cur_partition = self.log_sum_exp(cur_values, tag_size)
            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, tag_size)
            masked_cur_partition = cur_partition.masked_select(mask_idx)
            if masked_cur_partition.dim() != 0:
                mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)
                partition.masked_scatter_(mask_idx, masked_cur_partition)
        cur_values = self.transitions.view(1, tag_size, tag_size).expand(
            batch_size, tag_size, tag_size) + partition.contiguous().view(
                batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        cur_partition = self.log_sum_exp(cur_values, tag_size)
        final_partition = cur_partition[:, self.END_TAG_IDX]
        return final_partition.sum(), scores
    
    def _score_sentence(self, scores, mask, tags):
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(-1)
        new_tags = autograd.Variable(torch.LongTensor(batch_size, seq_len)).to(device)
        for idx in range(seq_len):
            if idx == 0:
                new_tags[:, 0] = (tag_size - 2) * tag_size + tags[:, 0]
            else:
                new_tags[:, idx] = tags[:, idx-1] * tag_size + tags[:, idx]

        end_transition = self.transitions[:, self.END_TAG_IDX].contiguous().view(
            1, tag_size).expand(batch_size, tag_size)
        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()
        end_ids = torch.gather(tags, 1, length_mask-1)
        end_energy = torch.gather(end_transition, 1, end_ids)
        new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len, batch_size, 1)
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(
            seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0))
        gold_score = tg_energy.sum() + end_energy.sum()
        return gold_score
    
    def neg_log_likelihood_loss(self, feats, mask, tags):
        batch_size = feats.size(0)
        forward_score, scores = self._forward_alg(feats, mask)
        gold_score = self._score_sentence(scores, mask, tags)
        if self.average_batch:
            return (forward_score - gold_score) / batch_size
        return forward_score - gold_score
    
    def calculate_loss(self, inputs, bi_inputs, labels, pos_id, seq_len):
        loss_function = nn.NLLLoss(ignore_index=0)
        embeds = self.word_embedding(inputs)
        uni_char_embeds = self.char_encode(embeds) # CNN
        if config['bi_gram']:
            bi_embeds = self.bi_embedding(bi_inputs)
            bi_char_embeds = self.char_encode(bi_embeds) # CNN
            #sent_inputs = torch.cat((embeds, bi_embeds), dim=2) # dim = 2
            sent_inputs = torch.cat((uni_char_embeds, bi_char_embeds), dim=2)
        else:
            sent_inputs = embeds

        #print('sent_inputs:', sent_inputs.shape)
        #sentence_length = sent_inputs.size()[1]
        #pos_id = torch.LongTensor(np.array([i for i in range(sentence_length)])).to(device)
        pos_inputs = self.pos_embedding(pos_id)
        #print('pos_inputs', pos_inputs.shape)
        # batch_size * sen_len * embedding_size
        new_inputs = sent_inputs + pos_inputs
        new_inputs = self.embedding_drop(new_inputs)
        #print('new_inputs', new_inputs.shape)
        for layer in self.layer_stack:
            inputs, _ = layer(new_inputs) # inputs [B, S, D]
            
        #outs = self.hidden2tag(inputs)
        #print('before_fc_out', inputs.shape)
        outs = self.fc_out(inputs) # [batch_size, seq_len, tag_size]
        #print("outs_after_fc", outs.shape)
        if self.crf:
            plen = inputs.size(1)
            #seq_len = seq_len.tolist()
            mask = torch.tensor([[1]*slen + [0]*(plen-slen) for slen in seq_len], dtype=torch.bool).to(device)
            loss = self.neg_log_likelihood_loss(outs, mask, labels)
            return loss
        else:   
            outs = outs.view(inputs.size(0) * inputs.size(1), -1)
            #print("before_softmax_loss", outs.shape)
            scores = F.log_softmax(outs, 1)
            #print("after_softmax_loss", scores.shape)
            #print("after_softmax_loss", labels.view(inputs.size(0) * inputs.size(1)))
            loss = loss_function(scores, labels.view(inputs.size(0) * inputs.size(1)))
            return loss
        
    def _viterbi_decode(self, feats, mask=None):
        """
        Args:
            feats: size=(batch_size, seq_len, self.target_size+2)
            mask: size=(batch_size, seq_len)

        Returns:
            decode_idx: (batch_size, seq_len), viterbi decode结果
            path_score: size=(batch_size, 1), 每个句子的得分
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(-1)

        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        feats = feats.transpose(1, 0).contiguous().view(
            ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)

        scores = feats + self.transitions.view(
            1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        seq_iter = enumerate(scores)
        # record the position of the best score
        back_points = list()
        partition_history = list()
        mask = (1 - mask.long()).bool()
        _, inivalues = seq_iter.__next__()
        partition = inivalues[:, self.START_TAG_IDX, :].clone().view(batch_size, tag_size, 1)
        partition_history.append(partition)

        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(
                batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            partition, cur_bp = torch.max(cur_values, 1)
            partition_history.append(partition.unsqueeze(-1))

            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, tag_size), 0)
            back_points.append(cur_bp)

        partition_history = torch.cat(partition_history).view(
            seq_len, batch_size, -1).transpose(1, 0).contiguous()

        last_position = length_mask.view(batch_size, 1, 1).expand(batch_size, 1, tag_size) - 1
        last_partition = torch.gather(
            partition_history, 1, last_position).view(batch_size, tag_size, 1)

        last_values = last_partition.expand(batch_size, tag_size, tag_size) +             self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)
        pad_zero = autograd.Variable(torch.zeros(batch_size, tag_size)).long().to(device)
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)

        pointer = last_bp[:, self.END_TAG_IDX]
        insert_last = pointer.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, tag_size)
        back_points = back_points.transpose(1, 0).contiguous()

        back_points.scatter_(1, last_position, insert_last)

        back_points = back_points.transpose(1, 0).contiguous()

        decode_idx = autograd.Variable(torch.LongTensor(seq_len, batch_size)).to(device)
        decode_idx[-1] = pointer.data
        for idx in range(len(back_points)-2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1))
            decode_idx[idx] = pointer.view(-1).data
        path_score = None
        decode_idx = decode_idx.transpose(1, 0)
        return path_score, decode_idx
    
    def forward(self, inputs, bi_inputs, pos_id, seq_len):
        embeds = self.word_embedding(inputs)
        uni_char_embeds = self.char_encode(embeds) # CNN
        if config['bi_gram']:
            bi_embeds = self.bi_embedding(bi_inputs)
            bi_char_embeds = self.char_encode(bi_embeds) # CNN
            #sent_inputs = torch.cat((embeds, bi_embeds), dim=2) # dim = 2
            sent_inputs = torch.cat((uni_char_embeds, bi_char_embeds), dim=2)
        else:
            sent_inputs = embeds

        #sentence_length = sent_inputs.size()[1]
        #pos_id = torch.LongTensor(np.array([i for i in range(sentence_length)])).to(device)
        pos_inputs = self.pos_embedding(pos_id)
        # batch_size * sen_len * embedding_size
        new_inputs = sent_inputs + pos_inputs # [barch_size, seq_len, embedding_dimensions]
        new_inputs = self.embedding_drop(new_inputs)

       # new_inputs = torch.cat((sent_inputs, pos_inputs), dim=2)
        for layer in self.layer_stack:
            inputs, _ = layer(new_inputs) # [barch_size, seq_len, embedding_dimensions]
        
        #outs = self.hidden2tag(inputs)
        outs = self.fc_out(inputs)
        if self.crf:
            plen = inputs.size(1)
            #seq_len = seq_len.tolist()
            mask = torch.tensor([[1]*slen + [0]*(plen-slen) for slen in seq_len]).to(device)
            path_score, tag_space = self._viterbi_decode(outs, mask)
        else:
            tag_space = F.softmax(outs, dim=2)
            tag_space = tag_space.argmax(dim=2)
        return tag_space


# In[3]:


class EncoderLayer(nn.Module):
    '''编码层'''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        '''

        :param d_model: 模型输入维度
        :param d_inner: 前馈神经网络隐层维度
        :param n_head:  多头注意力
        :param d_k:     键向量
        :param d_v:     值向量
        :param dropout:
        '''
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        '''

        :param enc_input:
        :param non_pad_mask:
        :param slf_attn_mask:
        :return:
        '''
        #输入为q k v mask
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        #print("multi_enc_output (before_ffn)", enc_output.shape)
        #print("enc_slf_attn (attention_matrix)", enc_slf_attn.shape)
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        #print("ffn_enc_output (shape == multi_enc_output)", enc_output.shape)
        if non_pad_mask is not None:
            enc_output *= non_pad_mask
        return enc_output, enc_slf_attn


# In[4]:


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()
        
    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context, attention


# In[5]:


class MultiHeadAttention(nn.Module):
    '''
        “多头”注意力模型
    '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        '''

        :param n_head: “头”数
        :param d_model: 输入维度
        :param d_k: 键向量维度
        :param d_v: 值向量维度
        :param dropout:
        '''
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        # 产生 查询向量q，键向量k， 值向量v
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        #初始化为正态分布
        #nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        #nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        #nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        #self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.attention = Scaled_Dot_Product_Attention()
        #channel方向上做归一化
        self.layer_normal = nn.LayerNorm(d_model)
        #生成前向向量
        self.fc = nn.Linear(n_head * d_v, d_model)
        #传播时方差不变
        #nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        '''
        计算多头注意力
        :param q: 用于产生  查询向量
        :param k: 用于产生  键向量
        :param v:  用于产生 值向量
        :param mask:
        :return:
        '''
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # (n*b) x lq x dk
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        # (n*b) x lk x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        # (n*b) x lv x dv
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        # mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        #
        output, attn = self.attention(q, k, v)
        #print('attn_output', output.shape)
        # (n_heads * batch_size) * lq * dv
        output = output.view(n_head, sz_b, len_q, d_v)
        # batch_size * len_q * (n_heads * dv)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = self.layer_normal(output + residual)
        return output, attn


# In[6]:


class PositionwiseFeedForward(nn.Module):
    '''
        前馈神经网络
    '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        '''

        :param d_in:    输入维度
        :param d_hid:   隐藏层维度
        :param dropout:
        '''
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_normal = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_normal(output + residual)
        return output

class CNNChar(nn.Module):
    def __init__(self, char_embeds, char_hidden, kernel_size, dropout=0.1):
        super(CNNChar, self).__init__()
        self.char_cnn = nn.Conv1d(char_embeds, char_hidden, kernel_size=kernel_size, padding=1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs):
        #inputs : [batch_size*seq_len_ofchar*d_model_ofchar]
        encoder_output = inputs.transpose(1,2)
        encoder_output = F.relu(self.char_cnn(encoder_output))
        encoder_output = encoder_output.transpose(1,2)
        encoder_output = self.dropout(encoder_output)
        return encoder_output

class MLP(nn.Module):
    def __init__(self, d_model, d_hidden, dropout=0.1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.layerNorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs):
        #inputs : [batch_size*seq_len*d_model]
        residual = inputs
        encoder_output = F.relu(self.fc1(inputs))
        encoder_output = F.relu(self.fc2(encoder_output))
        encoder_output = self.dropout(encoder_output)
        encoder_output = self.layerNorm(residual + encoder_output)
        return encoder_output

# In[7]:


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    '''
    计算位置向量
    :param n_position:      位置的最大值
    :param d_hid:           位置向量的维度，和字向量维度相同（要相加求和）
    :param padding_idx: 
    :return: 
    '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


# In[8]:

def score_shell(word_dict, gold_file, result_file, output_file):
    cmd = 'perl score ' + str(word_dict) + ' ' + str(gold_file) + ' ' + str(result_file) + '>' + str(output_file)
    res = os.system(cmd)
    return res
# In[9]:


def train(config):
    print ('get the training data')       
    train_data_loader = prepareData(config)
    print ('The training data is %d batches with %d batch sizes'%(len(train_data_loader), config['batch_size']))
    print ('------------------------------------------')
    print ('get the config information')
    print (config)
    print ('------------------------------------------')
    print ('get the test data')
    config['train_file'] = config['test_file']
    config['shuffle'] = True
    test_data_loader = prepareData(config)
    writeover = 0
    print ('------------------------------------------')
    print ('Train step! The model runs on ' + str(device))
    if config['update_model']:
        model = TextSlfAttnNet(config).to(device)
        load_model_name = config['update_model']
        print('load update model name is :' + load_model_name)
        checkpoint = torch.load(load_model_name)
        model.load_state_dict(checkpoint['net'])
    else:
        model = TextSlfAttnNet(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr = config['learning_rate'])
    #optimizer = optim.SGD(model.parameters(), lr = config['learning_rate'], momentum=0.95)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    loss_list = dict()
    F_log = 95.14
    for epoch in range(config['epochs']):
        start = time.time()
        model.train()
        model.zero_grad()
        total_loss = 0
        batch_step = 0
        for batch, data in enumerate(train_data_loader):
            batch_step += 1
            inputs, labels, bi_inputs, pos_id, seq_len = data
            #if not bigram, the bi_inputs is None    
            #loss_fn = nn.CrossEntropyLoss()
            #loss = loss_fn(inputs.to(device), labels.to(device))
            loss = model.calculate_loss(inputs.to(device), bi_inputs.to(device), labels.to(device), pos_id.to(device), seq_len)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            total_loss += loss.detach()
            print ("\rEpoch: %d ! the process is in %d of %d ! "%(epoch+1, batch+1, len(train_data_loader)), end='')
        scheduler.step()
        loss_avg = total_loss / batch_step
        loss_list[epoch] = loss_avg
        print ("The loss is %f ! "%(loss_avg))
    
        #valid process
        model.eval()
        with torch.no_grad():
            result_matrix_list = []
            gold_matrix_list = []
            for batch, data in enumerate(test_data_loader):
                inputs, labels, bi_inputs, pos_id, seq_len = data
                tag_space = model(inputs.to(device), bi_inputs.to(device), pos_id.to(device), seq_len)           
                result_matrix = tag_space.tolist()
                result_matrix = [result_matrix[i][:eof] for i, eof in enumerate(seq_len)]  
                labels = labels.tolist()
                labels = [labels[i][:eof] for i, eof in enumerate(seq_len)]
                result_matrix_list += result_matrix
                gold_matrix_list += labels
                
            P, R, F = score.score(result_matrix_list, gold_matrix_list)
            acc = score.accuracy(result_matrix_list, gold_matrix_list)
            end = time.time()
            print ('score| P:%.2f R:%.2f F:%.2f A:%.2f Time:%.2f'%(P, R, F, acc, end - start))
            if F > F_log:
                F_log = F
                writeover = 1
            else:
                writeover = 0
        if config['model_path'] and writeover == 1:
            state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            model_name = os.path.join(config['model_path'],'best.pkl')
            torch.save(state, model_name)
            print ('\n the epoch %d is saved successfully, named %s !'%(epoch+1, model_name))
            


# In[10]:


def test(config):
    print ('get the test data') 
    config['module'] = 'test'    
    sents, idx, bi_idx, pos_id = prepareData(config)
    print ('------------------------------------------')    
    model = TextSlfAttnNet(config).to(device)
    load_model_name = config['test_model']
    print ('model name is : ' + load_model_name)
    checkpoint = torch.load(load_model_name)
    model.load_state_dict(checkpoint['net'])
    model.eval()
    results = list()
    with torch.no_grad():
        for i in range(len(idx)):
            inputs = torch.tensor(idx[i]).unsqueeze(0)
            bi_inputs = torch.tensor(bi_idx[i]).unsqueeze(0)
            pos = torch.tensor(pos_id[i]).unsqueeze(0)
            seq_len = [len(idx[i])]
            result = model(inputs.to(device), bi_inputs.to(device), pos.to(device), seq_len).squeeze(0).tolist()
            results.append(result)
    #print (sents, results)
    new_results = repairResult(sents, results)
    #print (new_results)
    write(new_results, config['output_file'])
    print ('finish!')


# In[ ]:


if __name__ == '__main__':
    train(config)
    print ('main function finish!')
    #test(config)


# In[ ]:




