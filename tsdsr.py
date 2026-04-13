import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import FeatureSeqEmbLayer, VanillaAttention, FeedForward, MultiHeadAttention
from recbole.model.loss import BPRLoss
import copy
import math


class FrequencyLayer(nn.Module):
    def __init__(self, hidden_dropout_prob, hidden_size, c):
        super(FrequencyLayer, self).__init__()
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.c = c // 2 + 1

        self.beta = nn.Parameter(torch.randn(1, 1, 1))



    def forward(self, input_tensor):
        
        batch, seq_len, hidden = input_tensor.shape
        
        # 检查序列长度是否有效（FFT 需要至少 2 个元素）
        if seq_len < 2:
            hidden_states = self.out_dropout(input_tensor)
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
            return hidden_states
        
        try:
            # 尝试在 GPU 上执行 FFT
            x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
            low_pass = x[:]
            low_pass[:, self.c:, :] = 0
            low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')
            
            high_pass = input_tensor - low_pass
            sequence_emb_fft = low_pass + (self.beta**2) * high_pass

            hidden_states = self.out_dropout(sequence_emb_fft)
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
            return hidden_states
        except RuntimeError as e:
            # 如果 cuFFT 失败，在 CPU 上执行 FFT
            if 'cuFFT' in str(e) or 'CUFFT' in str(e):
                device = input_tensor.device
                input_cpu = input_tensor.cpu()
                
                x = torch.fft.rfft(input_cpu, dim=1, norm='ortho')
                low_pass = x[:]
                fft_output_len = seq_len // 2 + 1
                c_idx = min(self.c, fft_output_len)
                if c_idx < fft_output_len:
                    low_pass[:, c_idx:, :] = 0
                low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')
                low_pass = low_pass.to(device)
                
                high_pass = input_tensor - low_pass
                sequence_emb_fft = low_pass + (self.beta**2) * high_pass

                hidden_states = self.out_dropout(sequence_emb_fft)
                hidden_states = self.LayerNorm(hidden_states + input_tensor)
                return hidden_states
            else:
                raise e
      
        

class TVRecGFTFrequencyLayer(nn.Module):
    """
    TV-Rec style frequency filter for item sequence only:
    - Learnable basis/c_t in graph spectral domain.
    - Residual + dropout + LayerNorm, consistent with tsdsr blocks.
    """

    def __init__(self, hidden_dropout_prob, hidden_size, max_seq_length, M=32, K=50, layer_norm_eps=1e-12):
        super(TVRecGFTFrequencyLayer, self).__init__()
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.seq_len = max_seq_length
        self.K = int(max(1, K))
        self.pad_seq_len = self.seq_len + self.K - 1
        self.M = int(max(1, M))

        # Learnable spectral parameters, following TV-Rec parameterization.
        self.basis = nn.Parameter(torch.randn(self.M, self.K + 1, 2, dtype=torch.float32) * 1e-3)
        self.c_t = nn.Parameter(torch.randn(self.pad_seq_len, self.M, dtype=torch.float32) * 1e-3)

        gft, igft, L = self._transform_matrix()
        self.register_buffer("gft", gft, persistent=False)
        self.register_buffer("igft", igft, persistent=False)
        self.register_buffer("L", L, persistent=False)

    def _adjacency_matrix(self):
        A = torch.zeros((self.pad_seq_len, self.pad_seq_len), dtype=torch.float32)
        A[0, -1] = 1.0
        A[1:, :-1].fill_diagonal_(1.0)
        return A

    def _transform_matrix(self):
        A = self._adjacency_matrix()
        eigvals, eigvecs = torch.linalg.eig(A)
        L = torch.stack([eigvals ** i for i in range(self.K + 1)], dim=1)  # [pad_seq_len, K+1]
        gft = torch.linalg.inv(eigvecs)  # GFT
        igft = eigvecs  # inverse GFT
        return gft, igft, L

    def _pad_tensor(self, tensor):
        # tensor: [B, L, D], pad along sequence axis with K-1 zeros.
        pad_item = torch.zeros_like(tensor[:, : self.K - 1, :])
        return torch.cat([tensor, pad_item], dim=1)

    def orthogonal_regularization(self):
        # basis: [M, K+1, 2]
        B_real = self.basis[:, :, 0]
        B_imag = self.basis[:, :, 1]

        def _ortho_loss(matrix):
            matrix = F.normalize(matrix, p=2, dim=1)
            gram = matrix @ matrix.T
            I = torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)
            return ((gram - I) ** 2).sum()

        return _ortho_loss(B_real) + _ortho_loss(B_imag)

    def forward(self, input_tensor):
        # input_tensor: [B, L, D]
        batch, seq_len, _ = input_tensor.shape

        # RecBole usually uses fixed max_seq_length; fallback keeps robustness.
        if seq_len != self.seq_len:
            hidden_states = self.out_dropout(input_tensor)
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
            return hidden_states

        padded_tensor = self._pad_tensor(input_tensor)  # [B, pad_seq_len, D]
        x_tilde = self.gft @ padded_tensor.to(torch.complex64)  # [B, pad_seq_len, D]

        B = torch.view_as_complex(self.basis)  # [M, K+1]
        B = B / torch.norm(B, p=2, dim=1, keepdim=True).clamp_min(1e-7)
        H = self.c_t.to(torch.complex64) @ B  # [pad_seq_len, K+1]
        F_g = self.igft * (H @ self.L.T)  # [pad_seq_len, pad_seq_len]

        x = F_g @ x_tilde
        x = torch.real(x)[:, :seq_len, :]  # [B, L, D]

        hidden_states = self.out_dropout(x)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class tsdsr_MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, hidden_size,attribute_hidden_size,feat_num, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,fusion_type,max_len):
        super(tsdsr_MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.attribute_attention_head_size = [int(_ / n_heads) for _ in attribute_hidden_size]
        self.attribute_all_head_size = [self.num_attention_heads * _ for _ in self.attribute_attention_head_size]
        self.fusion_type = fusion_type
        self.max_len = max_len

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.query_p = nn.Linear(hidden_size, self.all_head_size)
        self.key_p = nn.Linear(hidden_size, self.all_head_size)

        self.feat_num = feat_num
        self.query_layers = nn.ModuleList(
            [copy.deepcopy(nn.Linear(attribute_hidden_size[_], self.attribute_all_head_size[_])) for _ in
             range(self.feat_num)])
        self.key_layers = nn.ModuleList(
            [copy.deepcopy(nn.Linear(attribute_hidden_size[_], self.attribute_all_head_size[_])) for _ in
             range(self.feat_num)])

        if self.fusion_type == 'concat':
            self.fusion_layer = nn.Linear(self.max_len*(2+self.feat_num), self.max_len)
        elif self.fusion_type == 'gate':
            self.fusion_layer = VanillaAttention(self.max_len, self.max_len)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)

        return x.permute(0, 2, 1, 3)

    def transpose_for_scores_attribute(self, x, i):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attribute_attention_head_size[i])
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor,attribute_table,position_embedding, attention_mask):
        item_query_layer = self.transpose_for_scores(self.query(input_tensor))
        item_key_layer = self.transpose_for_scores(self.key(input_tensor))
        item_value_layer = self.transpose_for_scores(self.value(input_tensor))

        pos_query_layer = self.transpose_for_scores(self.query_p(position_embedding))
        pos_key_layer = self.transpose_for_scores(self.key_p(position_embedding))

        item_attention_scores = torch.matmul(item_query_layer, item_key_layer.transpose(-1, -2))   # [B,h,L,L]
        pos_scores = torch.matmul(pos_query_layer, pos_key_layer.transpose(-1, -2))   # [B,h,L,L]

        # 去掉 cross-guided fusion，仅保留 item 与 position 的注意力
        attention_scores = item_attention_scores + pos_scores

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, item_value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class tsdsr_TransformerLayer(nn.Module):


    def __init__(
        self, n_heads, hidden_size,attribute_hidden_size,feat_num, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps,fusion_type,max_len,c,tvrec_M,tvrec_K,use_tvrec_item_gft,use_tvrec_attribute_gft
    ):
        super(tsdsr_TransformerLayer, self).__init__()
        self.multi_head_attention = tsdsr_MultiHeadAttention(
            n_heads, hidden_size,attribute_hidden_size, feat_num, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,fusion_type,max_len,
        )
        # filter_layer layout:
        #   [0] fused branch
        #   [1] item branch
        #   [2:] attribute branches
        # Only Ev and Eva go through GFT. Attribute branches bypass GFT.
        filter_layers = [
            TVRecGFTFrequencyLayer(
                hidden_dropout_prob=hidden_dropout_prob,
                hidden_size=hidden_size,
                max_seq_length=max_len,
                M=tvrec_M,
                K=tvrec_K,
                layer_norm_eps=layer_norm_eps,
            ),
            TVRecGFTFrequencyLayer(
                hidden_dropout_prob=hidden_dropout_prob,
                hidden_size=hidden_size,
                max_seq_length=max_len,
                M=tvrec_M,
                K=tvrec_K,
                layer_norm_eps=layer_norm_eps,
            ),
        ]
        self.filter_layer = nn.ModuleList(filter_layers)
        self.fusion_type = fusion_type
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)
        self.early_multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )

    def forward(self, hidden_states,attribute_embed, fused_embed, position_embedding, attention_mask):
        fd_attribute_embed = []

        # Ev
        fd_output = self.filter_layer[0](fused_embed)
        for i, filter_layer in enumerate(self.filter_layer):
            if i == 0:
                continue
            if i == 1:
                # Eva
                fd_attribute_embed.append(filter_layer(hidden_states).unsqueeze(2))

        hidden_states = fd_attribute_embed[0].squeeze(2)
        attribute_embed = attribute_embed

        attention_output = self.multi_head_attention(hidden_states,attribute_embed,position_embedding, attention_mask)
        early_attention_output = self.early_multi_head_attention(fd_output, attention_mask)
        fused_feedforward_output = self.feed_forward(early_attention_output)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output, fused_feedforward_output
    

    

class tsdsr_TransformerEncoder(nn.Module):

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        attribute_hidden_size=[64],
        feat_num=1,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12,
        fusion_type = 'sum',
        max_len = None,
        c = None,
        tvrec_M = 32,
        tvrec_K = 50,
        use_tvrec_item_gft = True,
        use_tvrec_attribute_gft = True,


    ):

        super(tsdsr_TransformerEncoder, self).__init__()
        layer = tsdsr_TransformerLayer(
            n_heads, hidden_size, attribute_hidden_size, feat_num, inner_size, hidden_dropout_prob, attn_dropout_prob,
            hidden_act, layer_norm_eps, fusion_type, max_len, c, tvrec_M, tvrec_K, use_tvrec_item_gft, use_tvrec_attribute_gft
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states,attribute_hidden_states, fused_hidden_states, position_embedding, attention_mask, output_all_encoded_layers=True):

        all_encoder_layers = []
        all_encoder_fused_layers = []
        for layer_module in self.layer:
            hidden_states, fused_hidden_states = layer_module(hidden_states, attribute_hidden_states, fused_hidden_states, position_embedding, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                all_encoder_fused_layers.append(fused_hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            all_encoder_fused_layers.append(fused_hidden_states)
        return all_encoder_layers, all_encoder_fused_layers

class tsdsr(SequentialRecommender):
    """
    tsdsr moves the side information from the input to the attention layer and decouples the attention calculation of
    various side information and item representation
    """

    def __init__(self, config, dataset):
        super(tsdsr, self).__init__(config, dataset)
        def _cfg(name, default):
            return config[name] if name in config else default

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.attribute_hidden_size = config['attribute_hidden_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']

        self.layer_norm_eps = config['layer_norm_eps']   # a parameter of nn.LayerNorm
        self.pooling_mode = config['pooling_mode']  # the way to calculate riched item embedding matrices

        self.selected_features = config['selected_features']  # categories
        self.device = config['device']
        self.num_feature_field = len(config['selected_features'])  # all default setting for all datasets use cate only
        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        self.fusion_type = config['fusion_type']

        self.lamdas = config['lamdas']
        self.attribute_predictor = config['attribute_predictor']
        self.lambda_ = config['lambda']
        self.align = config['align']
        # Alignment tuning module (enabled by default with conservative strength).
        self.use_align_tuning = bool(_cfg('use_align_tuning', True))
        self.align_temp_calib = float(_cfg('align_temp_calib', 0.05))
        self.align_target_smoothing = float(_cfg('align_target_smoothing', 0.02))
        # TV-Rec style item GFT settings (optional).
        self.tvrec_M = int(_cfg('M', 32))
        self.tvrec_K = int(_cfg('K', 50))
        self.reg_weight = float(_cfg('reg_weight', 0.0))
        self.orthogonal_reg_weight = float(_cfg('orthogonal_reg_weight', self.reg_weight))
        self.use_tvrec_item_gft = bool(_cfg('use_tvrec_item_gft', True))
        self.use_tvrec_attribute_gft = bool(_cfg('use_tvrec_attribute_gft', True))

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)  # is it reverse ?
        self.dataset = dataset
        self.feature_embed_layer_list = nn.ModuleList(
            [copy.deepcopy(FeatureSeqEmbLayer(dataset, self.attribute_hidden_size[_], [self.selected_features[_]], self.pooling_mode,self.device)) for _
             in range(len(self.selected_features))])  # is a layer that initialize feature embedding
        self.c = config['c']

        self.trm_encoder = tsdsr_TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            attribute_hidden_size=self.attribute_hidden_size,
            feat_num=len(self.selected_features),
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            fusion_type=self.fusion_type,
            max_len=self.max_seq_length,
            c=self.c,
            tvrec_M=self.tvrec_M,
            tvrec_K=self.tvrec_K,
            use_tvrec_item_gft=self.use_tvrec_item_gft,
            use_tvrec_attribute_gft=self.use_tvrec_attribute_gft
               
        )

        if self.fusion_type == 'concat':
            self.fusion_layer = nn.Linear(self.hidden_size*(len(self.selected_features)+2), self.hidden_size)   # 2+self.feat_num  2:item,position
        elif self.fusion_type == 'gate':
            self.fusion_layer = VanillaAttention(self.hidden_size, self.hidden_size)
        self.alpha = config['alpha']

        self.n_attributes = {}
        for attribute in self.selected_features:
            self.n_attributes[attribute] = len(dataset.field2token_id[attribute])
        if self.attribute_predictor == 'MLP':
            self.ap = nn.Sequential(nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
                                    nn.BatchNorm1d(num_features=self.hidden_size),
                                    nn.ReLU(),
                                    # final logits
                                    nn.Linear(in_features=self.hidden_size, out_features=self.n_attributes))
        elif self.attribute_predictor == 'linear':
            self.ap = nn.ModuleList([copy.deepcopy(nn.Linear(in_features=self.hidden_size, out_features=self.n_attributes[_]))
                 for _ in self.selected_features])

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
            self.attribute_loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)
        self.other_parameter_name = ['feature_embed_layer_list']

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)  # note that the input of this code have no side information_seq, seems in dataset
        # position embedding
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        feature_table = []
        for feature_embed_layer in self.feature_embed_layer_list:
            sparse_embedding, dense_embedding = feature_embed_layer(None, item_seq)
            sparse_embedding = sparse_embedding['item']  # [bs, L, 1, d_f]
            dense_embedding = dense_embedding['item']  # None
        # concat the sparse embedding and float embedding
            if sparse_embedding is not None:
                feature_table.append(sparse_embedding)
            if dense_embedding is not None:
                feature_table.append(dense_embedding)

        feature_emb = feature_table  # here is the cate emb [bs, L, 1, d_f]


        input_emb = item_emb  # [bs, L, d]
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        attribute_embed_cat = torch.cat(feature_emb, dim=2)
        attribute_embed_cat = torch.cat([input_emb.unsqueeze(2), attribute_embed_cat, position_embedding.unsqueeze(2)], dim=2)
        if self.fusion_type == 'sum':
            attribute_embed_cat = torch.sum(attribute_embed_cat, dim=-2) 
        elif self.fusion_type == 'concat':
            # [B,L,feat_num, d_f] -> [B,L,feat_num*d_f]
            table_shape = attribute_embed_cat.shape
            attribute_embed_cat = attribute_embed_cat.view(table_shape[:-2] + (table_shape[-2] * table_shape[-1],))  # [B,h,L,fea num,L]->[B,h,L,fea num*L]
            attribute_embed_cat = self.fusion_layer(attribute_embed_cat)  # [B,h,L,(fea_num+2)*L] -> [B,h,L,L]

        elif self.fusion_type == 'gate':
            attribute_embed_cat, _ = self.fusion_layer(attribute_embed_cat)

        extended_attention_mask = self.get_attention_mask(item_seq)  # [bs, 1, L, L]
        trm_output, fused_output = self.trm_encoder(input_emb, feature_emb, attribute_embed_cat, position_embedding, extended_attention_mask,
                                      output_all_encoded_layers=True)
        trm_output = trm_output[-1]  # the output of last layer [bs, L, d]
        fused_output = fused_output[-1]  # the output of last layer [bs, L, d]
        output = self.alpha * trm_output + (1 - self.alpha) * fused_output
        seq_output = self.gather_indexes(output, item_seq_len - 1)  # [bs, d]
        return seq_output, feature_emb

    def align_loss(self, X, A, item_seq, temperature=0.1):

        # Normalize the embeddings
        A = torch.cat(A, dim=2)
        A = torch.sum(A, dim=2)
        X_norm = F.normalize(X, p=2, dim=-1)
        A_norm = F.normalize(A, p=2, dim=-1)

        # Length-aware temperature calibration (when enabled).
        seq_mask = (item_seq > 0).float()
        valid_ratio = (seq_mask.sum(-1) / seq_mask.size(-1)).unsqueeze(-1).unsqueeze(-1).clamp_min(1e-8)
        if self.use_align_tuning and self.align_temp_calib > 0:
            temp = temperature * (1.0 + self.align_temp_calib * (1.0 - valid_ratio))
        else:
            temp = temperature

        # Calculate similarity matrices
        similarity_matrix_X = torch.matmul(X_norm, A_norm.transpose(-1, -2)) / temp
        similarity_matrix_A = torch.matmul(A_norm, X_norm.transpose(-1, -2)) / temp

        Y_X = F.softmax(similarity_matrix_X, dim=-1)
        Y_A = F.softmax(similarity_matrix_A, dim=-1)

        epsilon = 1e-6  
        ground_truth = (torch.abs(torch.matmul(A_norm, A_norm.transpose(-1, -2)) - 1) < epsilon).float().to(X.device)
        # Target smoothing over valid tokens (when enabled).
        if self.use_align_tuning and self.align_target_smoothing > 0:
            smooth = min(max(self.align_target_smoothing, 0.0), 0.5)
            pair_mask = (seq_mask.unsqueeze(-1) * seq_mask.unsqueeze(-2))
            gt = ground_truth * pair_mask
            gt_row_sum = gt.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            gt = gt / gt_row_sum
            uniform = pair_mask / pair_mask.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            ground_truth = (1.0 - smooth) * gt + smooth * uniform
        # Compute the loss
        mask = seq_mask.unsqueeze(-1)
        mask = mask @ mask.transpose(-1, -2)
        loss_X = ground_truth * torch.log(Y_X)
        loss_A = ground_truth * torch.log(Y_A)
        loss_X = loss_X * mask
        loss_A = loss_A * mask

        loss_X = - loss_X.sum(dim=-1) / seq_mask.sum(-1).unsqueeze(-1)
        loss_A = - loss_A.sum(dim=-1) / seq_mask.sum(-1).unsqueeze(-1)

        loss = (loss_X + loss_A).mean()

        return loss


    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, feature_emb = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        orthogonal_reg = self._tvrec_orthogonal_regularization()
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss + self.orthogonal_reg_weight * orthogonal_reg
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [bs, I]
            loss = self.loss_fct(logits, pos_items)
            if self.align:
                loss += self.lambda_* self.align_loss(test_item_emb[item_seq], feature_emb, item_seq)
            total_loss = loss + self.orthogonal_reg_weight * orthogonal_reg
            return total_loss

    def _tvrec_orthogonal_regularization(self):
        """
        Sum orthogonal regularization from TV-Rec style GFT filters
        across all transformer layers.
        """
        if self.orthogonal_reg_weight <= 0:
            return torch.zeros((), device=self.item_embedding.weight.device)
        reg = torch.zeros((), device=self.item_embedding.weight.device)
        for layer_module in self.trm_encoder.layer:
            # index 0: Ev, index 1: Eva.
            for branch_filter in layer_module.filter_layer:
                if hasattr(branch_filter, "orthogonal_regularization"):
                    reg = reg + branch_filter.orthogonal_regularization()
        # Normalize regularization scale to avoid drift when changing layer/feature counts.
        num_layers = max(1, int(self.n_layers))
        num_branches = max(1, len(self.trm_encoder.layer[0].filter_layer))
        return reg / float(num_layers * num_branches)

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output,feature_emb = self.forward(item_seq, item_seq_len)
        test_item = interaction[self.ITEM_ID]
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, feature_emb = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, item_num]
        return scores
