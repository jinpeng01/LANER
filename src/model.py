import torch
import torch.nn as nn
from torch.nn import functional as F
# from transformers import BertModel, BertTokenizer
from transformers import AutoConfig
from transformers import AutoModelWithLMHead
from src.utils import load_embedding
import numpy as np

import logging

logger = logging.getLogger()


def del_tensor_ele(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index + 1:]
    return torch.cat((arr1, arr2), dim=0)

class BertTagger(nn.Module):
    def __init__(self, params):
        super(BertTagger, self).__init__()
        self.num_tag = params.num_tag
        self.hidden_dim = params.hidden_dim
        self.target_embedding_dim = params.target_embedding_dim
        config = AutoConfig.from_pretrained(params.model_name)
        config.output_hidden_states = True
        # self.bert = BertModel.from_pretrained("bert-base-cased")
        self.model = AutoModelWithLMHead.from_pretrained(params.model_name, config=config)
        if params.ckpt != "":
            logger.info("Reloading model from %s" % params.ckpt)
            model_ckpt = torch.load(params.ckpt)
            self.model.load_state_dict(model_ckpt)

        self.target_sequence = params.target_sequence
        self.target_type = params.target_type
        self.connect_label_background = params.connect_label_background


        if self.target_sequence == True:



            self.target_embedding = nn.Embedding(self.num_tag + 2, self.target_embedding_dim, padding_idx=0)

            self.to_target_emb = nn.Linear(self.hidden_dim + 2 * self.target_embedding_dim, self.target_embedding_dim)

            if self.target_type == 'RNN':
                self.RNNencoder = nn.LSTM(int(self.target_embedding_dim), int(self.target_embedding_dim), batch_first=True,
                                          bidirectional=True)
                self.RNNoutLinear = nn.Linear(2 * self.target_embedding_dim, self.hidden_dim)
                self.BERTtoTarget2 = nn.Linear(self.hidden_dim * 2, 2 * self.target_embedding_dim)
                self.BERTtoTarget = nn.Linear(self.hidden_dim, 2 * self.target_embedding_dim)


            self.se_linear = nn.Linear(self.hidden_dim * 2 + self.target_embedding_dim * 2, self.num_tag)
            self.se_linear_first = nn.Linear(self.hidden_dim * 2 + self.target_embedding_dim * 2, self.num_tag)



        else:

            self.linear = nn.Linear(self.hidden_dim, self.num_tag)

    def dot_attention(self, current_word_embedding, total_y_embedding):

        attention_weight = current_word_embedding @ total_y_embedding.permute(0, 2, 1)
        attention_weight = torch.softmax(attention_weight, dim=-1)
        relation_information = attention_weight @ total_y_embedding

        return relation_information

    def forward(self, X, y=None):
        outputs = self.model(X)  # a tuple ((bsz,seq_len,hidden_dim), (bsz, hidden_dim))
        outputs = outputs[1][-1]  # (bsz, seq_len, hidden_dim)
        hcl_loss = 0

        if self.target_sequence == True:

            y_modified = torch.where(y < -1, self.num_tag + 1, y).to(outputs.device)
            y_embedding = self.target_embedding(y_modified)
            bsz, seq_len, dim = outputs.shape
            predcits = []

            init_zero = torch.zeros([bsz, 2 * self.target_embedding_dim], dtype=torch.float32, device='cuda')

            for i in range(seq_len):
                if i == 0:
                    current_word_re = torch.cat([outputs[:, i, :], init_zero], dim=1)
                    current_label_embedding = self.to_target_emb(current_word_re)


                    current_word_re = torch.cat([outputs[:, i, :],
                                                 outputs[:, i, :],
                                                 current_label_embedding.squeeze(),
                                                 current_label_embedding.squeeze()], dim=-1)
                    predict = self.se_linear_first(current_word_re)


                else:
                    total_y_embedding = y_embedding[:, :i, :]

                    if self.target_type == 'RNN':
                        output_lstm, (hn, cn) = self.RNNencoder(total_y_embedding)
                        relation_information = output_lstm[:, -1, :]

                        ##label_background
                        label_memory = self.RNNoutLinear(relation_information)
                        label_background = self.dot_attention(label_memory.unsqueeze(dim=1), outputs).squeeze()

                        ##label_context
                        if self.connect_label_background:
                            output_memory = self.BERTtoTarget2(torch.cat([outputs[:, i, :], label_background], dim=-1))
                        else:
                            output_memory = self.BERTtoTarget(outputs[:, i, :])
                        label_context = self.dot_attention(output_memory.unsqueeze(dim=1), output_lstm).squeeze()

                    total_word_re = torch.cat([outputs[:, i, :], label_background, label_context], dim=-1)
                    predict = self.se_linear(total_word_re)

                predcits.append(predict)

            prediction2 = torch.stack(predcits, dim=1)

        else:
            prediction2 = self.linear(outputs)


        return prediction2, hcl_loss

    def test(self, X):

        outputs = self.model(X)  # a tuple ((bsz,seq_len,hidden_dim), (bsz, hidden_dim))
        outputs = outputs[1][-1]  # (bsz, seq_len, hidden_dim)

        if self.target_sequence == True:
            bsz, seq_len, dim = outputs.shape
            predcits = []
            init_zero = torch.zeros([bsz, 2 * self.target_embedding_dim], dtype=torch.float32, device='cuda')
            total_predict = None

            for i in range(seq_len):

                if i == 0:
                    current_word_re = torch.cat([outputs[:, i, :], init_zero], dim=1)
                    current_label_embedding = self.to_target_emb(current_word_re)
                    if len(outputs[:, i, :].shape) == len(current_label_embedding.shape):
                        current_word_re = torch.cat(
                                [outputs[:, i, :], outputs[:, i, :], current_label_embedding,
                                 current_label_embedding], dim=-1)
                    else:
                        current_word_re = torch.cat(
                                [outputs[:, i, :], outputs[:, i, :], current_label_embedding.squeeze(),
                                 current_label_embedding.squeeze()], dim=-1)

                    predict = self.se_linear_first(current_word_re)

                else:
                    total_y_embedding = self.target_embedding(total_predict)

                    if self.target_type == 'RNN':
                        output_lstm, (hn, cn) = self.RNNencoder(total_y_embedding)
                        relation_information = output_lstm[:, -1, :]
                        label_memory = self.RNNoutLinear(relation_information)
                        label_background = self.dot_attention(label_memory.unsqueeze(dim=1), outputs).squeeze()
                        if len(label_background.shape) == 1:
                            label_background = label_background.unsqueeze(dim=0)

                        if self.connect_label_background:
                            output_memory = self.BERTtoTarget2(torch.cat([outputs[:, i, :], label_background], dim=-1))
                        else:
                            output_memory = self.BERTtoTarget(outputs[:, i, :])

                        label_context = self.dot_attention(output_memory.unsqueeze(dim=1), output_lstm).squeeze()
                        if len(label_context.shape) == 1:
                            label_context = label_context.unsqueeze(dim=0)

                    if len(outputs[:, i, :].shape) == len(label_background.shape):
                        total_word_re = torch.cat([outputs[:, i, :], label_background, label_context], dim=-1)
                    else:
                        total_word_re = torch.cat(
                            [outputs[:, i, :], label_background.unsqueeze(dim=0), label_context.unsqueeze(dim=0)],
                            dim=-1)

                    predict = self.se_linear(total_word_re)



                current_predict = predict.data.cpu().numpy()
                current_predict = np.argmax(current_predict, axis=1)
                current_predict2 = torch.tensor(current_predict, dtype=torch.long, device='cuda')

                if total_predict == None:
                    total_predict = current_predict2.unsqueeze(dim=1)
                else:
                    total_predict = torch.cat([total_predict, current_predict2.unsqueeze(dim=1)], dim=-1)

                predcits.append(predict)

            prediction2 = torch.stack(predcits, dim=1)
        else:

            prediction2 = self.linear(outputs)

        return prediction2


class BiLSTMTagger(nn.Module):
    def __init__(self, params):
        super(BiLSTMTagger, self).__init__()


        self.num_tag = params.num_tag
        self.hidden_dim = params.hidden_dim
        config = AutoConfig.from_pretrained(params.model_name)
        config.output_hidden_states = True
        # self.bert = BertModel.from_pretrained("bert-base-cased")
        self.model = AutoModelWithLMHead.from_pretrained(params.model_name, config=config)
        if params.ckpt != "":
            logger.info("Reloading model from %s" % params.ckpt)
            model_ckpt = torch.load(params.ckpt)
            self.model.load_state_dict(model_ckpt)

        self.linear = nn.Linear(self.hidden_dim, self.num_tag)
        self.crf_layer = CRF(params.num_tag)

    def forward(self, X, return_hiddens=False,y=None):
        """
        Input:
            X: (bsz, seq_len)
        Output:
            prediction: (bsz, seq_len, num_tag)
            lstm_hidden: (bsz, seq_len, hidden_size)
        """
        outputs = self.model(X)  # a tuple ((bsz,seq_len,hidden_dim), (bsz, hidden_dim))
        outputs = outputs[1][-1]  # (bsz, seq_len, hidden_dim)

        prediction = self.linear(outputs)

        # import pdb
        # pdb.set_trace()

        if return_hiddens:
            return prediction, outputs
        else:
            return prediction,None

    def crf_decode(self, inputs, lengths):
        """ crf decode
        Input:
            inputs: (bsz, seq_len, num_entity)
            lengths: lengths of x (bsz, )
        Ouput:
            crf_loss: loss of crf
        """
        prediction = self.crf_layer(inputs)
        prediction = [prediction[i, :length].data.cpu().numpy() for i, length in enumerate(lengths)]

        return prediction

    def crf_loss(self, inputs, lengths, y):
        """ create crf loss
        Input:
            inputs: (bsz, seq_len, num_entity)
            lengths: lengths of x (bsz, )
            y: label of entity value (bsz, seq_len)
        Ouput:
            crf_loss: loss of crf
        """
        padded_y = self.pad_label(lengths, y)
        crf_loss = self.crf_layer.loss(inputs, padded_y)

        return crf_loss

    def pad_label(self, lengths, y):
        bsz = len(lengths)
        max_len = torch.max(lengths)
        padded_y = torch.LongTensor(bsz, max_len).fill_(0)
        for i in range(bsz):
            length = lengths[i]
            y_i = y[i]
            padded_y[i, 0:length] = torch.LongTensor(y_i)

        padded_y = padded_y.cuda()
        return padded_y



class CRF(nn.Module):
    """
    Implements Conditional Random Fields that can be trained via
    backpropagation.
    """

    def __init__(self, num_tags):
        super(CRF, self).__init__()

        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.Tensor(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.stop_transitions = nn.Parameter(torch.randn(num_tags))

        nn.init.xavier_normal_(self.transitions)

    def forward(self, feats):
        # Shape checks
        if len(feats.shape) != 3:
            raise ValueError("feats must be 3-d got {}-d".format(feats.shape))

        return self._viterbi(feats)

    def loss(self, feats, tags):
        """
        Computes negative log likelihood between features and tags.
        Essentially difference between individual sequence scores and
        sum of all possible sequence scores (partition function)
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
            tags: Target tag indices [batch size, sequence length]. Should be between
                    0 and num_tags
        Returns:
            Negative log likelihood [a scalar]
        """
        # Shape checks
        if len(feats.shape) != 3:
            raise ValueError("feats must be 3-d got {}-d".format(feats.shape))

        if len(tags.shape) != 2:
            raise ValueError('tags must be 2-d but got {}-d'.format(tags.shape))

        if feats.shape[:2] != tags.shape:
            raise ValueError('First two dimensions of feats and tags must match ', feats.shape, tags.shape)

        sequence_score = self._sequence_score(feats, tags)
        partition_function = self._partition_function(feats)
        log_probability = sequence_score - partition_function

        # -ve of l()
        # Average across batch
        return -log_probability.mean()

    def _sequence_score(self, feats, tags):
        """
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
            tags: Target tag indices [batch size, sequence length]. Should be between
                    0 and num_tags
        Returns: Sequence score of shape [batch size]
        """

        batch_size = feats.shape[0]

        # Compute feature scores
        feat_score = feats.gather(2, tags.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

        # print(feat_score.size())

        # Compute transition scores
        # Unfold to get [from, to] tag index pairs
        tags_pairs = tags.unfold(1, 2, 1)

        # Use advanced indexing to pull out required transition scores
        indices = tags_pairs.permute(2, 0, 1).chunk(2)
        trans_score = self.transitions[indices].squeeze(0).sum(dim=-1)

        # Compute start and stop scores
        start_score = self.start_transitions[tags[:, 0]]
        stop_score = self.stop_transitions[tags[:, -1]]

        return feat_score + start_score + trans_score + stop_score

    def _partition_function(self, feats):
        """
        Computes the partitition function for CRF using the forward algorithm.
        Basically calculate scores for all possible tag sequences for
        the given feature vector sequence
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
        Returns:
            Total scores of shape [batch size]
        """
        _, seq_size, num_tags = feats.shape

        if self.num_tags != num_tags:
            raise ValueError('num_tags should be {} but got {}'.format(self.num_tags, num_tags))

        a = feats[:, 0] + self.start_transitions.unsqueeze(0)  # [batch_size, num_tags]
        transitions = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags] from -> to

        for i in range(1, seq_size):
            feat = feats[:, i].unsqueeze(1)  # [batch_size, 1, num_tags]
            a = self._log_sum_exp(a.unsqueeze(-1) + transitions + feat, 1)  # [batch_size, num_tags]

        return self._log_sum_exp(a + self.stop_transitions.unsqueeze(0), 1)  # [batch_size]

    def _viterbi(self, feats):
        """
        Uses Viterbi algorithm to predict the best sequence
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
        Returns: Best tag sequence [batch size, sequence length]
        """
        _, seq_size, num_tags = feats.shape

        if self.num_tags != num_tags:
            raise ValueError('num_tags should be {} but got {}'.format(self.num_tags, num_tags))

        v = feats[:, 0] + self.start_transitions.unsqueeze(0)  # [batch_size, num_tags]
        transitions = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags] from -> to
        paths = []

        for i in range(1, seq_size):
            feat = feats[:, i]  # [batch_size, num_tags]
            v, idx = (v.unsqueeze(-1) + transitions).max(1)  # [batch_size, num_tags], [batch_size, num_tags]

            paths.append(idx)
            v = (v + feat)  # [batch_size, num_tags]

        v, tag = (v + self.stop_transitions.unsqueeze(0)).max(1, True)

        # Backtrack
        tags = [tag]
        for idx in reversed(paths):
            tag = idx.gather(1, tag)
            tags.append(tag)

        tags.reverse()
        return torch.cat(tags, 1)

    def _log_sum_exp(self, logits, dim):
        """
        Computes log-sum-exp in a stable way
        """
        max_val, _ = logits.max(dim)
        return max_val + (logits - max_val.unsqueeze(dim)).exp().sum(dim).log()
