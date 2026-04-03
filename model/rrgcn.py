import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.fft as fft
try:
    from dgl.nn.pytorch import TAGConv
except ImportError:
    # 兼容旧版 DGL：若无 TAGConv，则回退到 GraphConv
    from dgl.nn.pytorch import GraphConv as TAGConv

# from rgcn.layers import RGCNBlockLayer as RGCNLayer
from model.layers import UnionRGCNLayer, CompGCNCovLayer
from model.hrgcn import HawkesRGCNLayer
from model.lie_regularizer import RelationAwareLieRegularizer
from model.temporal_trend_encoder import (
    TemporalTrendEncoder, PeriodicTrendTimeEmbedding, 
    TemporalGatingModule, AngleConstrainedLoss, TemporalContrastiveLoss
)
from src.utils import merge_graphs
#from src.model import BaseRGCN
from model.decoder import ConvTransE, ConvTransR, InteractE
from model.eventmodel import EventDecoderE, EventDecoderR
from model.focalloss import FocalLoss

class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, encoder_name="", opn="sub", rel_emb=None, use_cuda=False):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_basis = num_basis
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.skip_connect = skip_connect
        self.self_loop = self_loop
        self.encoder_name = encoder_name
        self.use_cuda = use_cuda
        self.skip_connect = skip_connect
        print("use layer :{}".format(encoder_name))
        self.rel_emb = rel_emb
        self.opn = opn
        # 构建 RGCN 各层
        self.build_model()
        # 构建初始特征
        self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):

            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    # 为每个节点初始化特征
    def create_features(self):
        return None

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g):
        if self.features is not None:
            g.ndata['id'] = self.features
        print("h before GCN message passing")
        print(g.ndata['h'])
        print("h behind GCN message passing")
        for layer in self.layers:
            layer(g)
        print(g.ndata['h'])
        return g.ndata.pop('h')



class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        #print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        elif self.encoder_name == 'compgcn':
            return CompGCNCovLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases, activation=act, dropout=self.dropout, opn=self.opn, rel_emb=self.rel_emb)
        elif self.encoder_name == 'hrgcn':
            return HawkesRGCNLayer(self.h_dim, self.h_dim, self.num_rels, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb, use_temporal_gating=False)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn" or self.encoder_name == "hrgcn":
            node_id = g.ndata['id'].squeeze()
            edge_id = g.edata['type'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r)
            return g.ndata.pop('h'), []
        if self.encoder_name == "compgcn": 
            node_id = g.ndata['id'].squeeze()
            edge_id = g.edata['type'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            g.edata['h'] = init_rel_emb[edge_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                x, r = layer(g, x, r)
            return x, r
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')



class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, 
                 theta=1, entity_prediction=False, relation_prediction=False, raw_input=False, use_cuda=False,
                 gpu = 0, alpha=10.0,
                 use_rel_context_prior=True, rel_prior_weight=0.3,
                 use_lie_reg=True, lie_p=3.0, lie_ent_weight=0.005, lie_rel_weight=0.01, lie_pair_weight=0.01,
                 use_relation_dynamics=False, use_copy_generation=False, copy_gen_alpha=0.5,
                 use_temporal_trend=False, temporal_gating=False, time_embedding_alpha=0.5,
                 angle_degree=10.0, temporal_temperature=0.07, use_angle_constraint=False,
                 use_temporal_contrastive=False, angle_constraint_weight=0.1, temporal_contrastive_weight=0.1):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.graph_h_dim = h_dim*2
        self.layer_norm = layer_norm
        self.h = None
        self.h_0 = None
        self.graph_h = None
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.theta = theta
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.use_cuda = use_cuda
        self.gpu = gpu
        self.alpha = alpha
        self.use_rel_context_prior = use_rel_context_prior
        self.rel_prior_weight = rel_prior_weight
        self.use_relation_dynamics = use_relation_dynamics
        self.use_copy_generation = use_copy_generation
        self.copy_gen_alpha = copy_gen_alpha
        
        # 周期趋势时序编码配置
        self.use_temporal_trend = use_temporal_trend
        self.temporal_gating = temporal_gating
        self.use_angle_constraint = use_angle_constraint
        self.use_temporal_contrastive = use_temporal_contrastive
        self.angle_constraint_weight = angle_constraint_weight
        self.temporal_contrastive_weight = temporal_contrastive_weight
        
        # 若启用则初始化时序趋势编码器
        if self.use_temporal_trend:
            self.temporal_trend_encoder = TemporalTrendEncoder(
                num_entities=num_ents,
                h_dim=h_dim,
                max_history_len=sequence_len,
                alpha_balance=time_embedding_alpha,
                angle_degree=angle_degree,
                temperature=temporal_temperature,
                use_gating=temporal_gating,
                use_angle_loss=use_angle_constraint,
                use_contrastive_loss=use_temporal_contrastive,
                dropout=dropout
            )
        
        # Lie 群正则化配置
        self.use_lie_reg = use_lie_reg
        if self.use_lie_reg:
            self.lie_regularizer = RelationAwareLieRegularizer(
                dim=h_dim,
                p=lie_p,
                ent_weight=lie_ent_weight,
                rel_weight=lie_rel_weight,
                pair_weight=lie_pair_weight
            )

        # 关系嵌入采用 2|R| 维度，包含正向关系与反向关系
        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()      #对所有的关系做嵌入
        torch.nn.init.xavier_normal_(self.emb_rel)

        # 实体基础嵌入，在 forward 中可被社区图增强后的表示替换
        self.emb_ent = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()      #对实体做嵌入
        torch.nn.init.normal_(self.emb_ent)

        #注意力中的网络层
        # self.w_q = torch.nn.Parameter(torch.Tensor(self.graph_h_dim, 64), requires_grad=True).float()
        # torch.nn.init.xavier_normal_(self.w_q)
        # self.w_k = torch.nn.Parameter(torch.Tensor(self.graph_h_dim, 64), requires_grad=True).float()
        # torch.nn.init.xavier_normal_(self.w_k)
        # self.w_v = torch.nn.Parameter(torch.Tensor(self.graph_h_dim, 64), requires_grad=True).float()
        # torch.nn.init.xavier_normal_(self.w_v)

        # self.loss_r = torch.nn.CrossEntropyLoss()
        # self.loss_e = torch.nn.CrossEntropyLoss()
        self.loss_e = FocalLoss(gamma=2)
        self.loss_r = FocalLoss(gamma=2)

        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda)
        #self.classgcn = GraphConv(h_dim, h_dim)
        self.classgcn = TAGConv(h_dim, h_dim)

        self.node2graph = nn.Linear(h_dim, self.graph_h_dim)
        self.node2graph_gate = nn.Sequential(nn.Linear(h_dim, 1), nn.Sigmoid())
        self.reset_gate = nn.Linear(h_dim, 1)
        self.reset_gate1 = nn.Linear(self.num_ents, 1)

        # 解码器
        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        elif decoder_name == "film":
            self.decoder_ob = EventDecoderE(h_dim, num_ents)
            self.rdecoder = EventDecoderR(h_dim, num_rels)
        elif decoder_name =='interacte':
            self.decoder_ob = InteractE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError 

        # 关系上下文先验：使用 (主体, 客体) 拼接特征生成关系先验偏置
        # 该项以加性偏置方式作用于关系 logits，为解码器引入结构先验
        if self.use_rel_context_prior:
            self.rel_prior = nn.Sequential(
                nn.Linear(2 * h_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(feat_dropout),
                nn.Linear(h_dim, 2 * num_rels)
            )
            
        # 设备处理
        if use_cuda and isinstance(gpu, int) and gpu >= 0:
            self.device = f'cuda:{gpu}'
        else:
            self.device = 'cpu'

        # 默认置空，避免条件分支未触发时属性不存在
        self.global_rel_dynamics = None
        self.copy_gen_decoder = None
        
        # ERD-Net创新模块初始化
        if self.use_relation_dynamics:
            try:
                from model.relation_dynamics import GlobalRelationDynamics
                self.global_rel_dynamics = GlobalRelationDynamics(
                    num_ents, num_rels, h_dim, self.device
                )
                print("✅ ERD-Net全局关系动态模块初始化成功")
            except Exception as e:
                print(f"⚠️ GlobalRelationDynamics初始化失败: {e}")
                self.use_relation_dynamics = False
                self.global_rel_dynamics = None
                
        if self.use_copy_generation and self.relation_prediction:
            try:
                from model.copy_generation_decoder import CopyGenerationRelationDecoder
                self.copy_gen_decoder = CopyGenerationRelationDecoder(
                    h_dim, num_rels, alpha=self.copy_gen_alpha,
                    device=self.device, dropout=feat_dropout
                )
                print("✅ ERD-Net复制-生成机制初始化成功")
            except Exception as e:
                print(f"⚠️ CopyGenerationRelationDecoder初始化失败: {e}")
                self.use_copy_generation = False
                self.copy_gen_decoder = None


    def forward(self, g_list, class_g, use_cuda):
        # 步骤1：社区图增强实体初始表示（对应论文中的社区/类别结构融合）
        if class_g is None:
            current_ent_emb = self.emb_ent
        else:
            if use_cuda:
                class_g = class_g.to(self.gpu)
            current_ent_emb = self.emb_ent
            current_ent_emb = self.classgcn(class_g, current_ent_emb)
            current_ent_emb = F.normalize(current_ent_emb) if self.layer_norm else current_ent_emb
            self.emb_ent = torch.nn.Parameter(current_ent_emb)

        history_embs = []
        graph_h_list = []
        
        # 步骤2：初始化关系表示，后续可被关系动态模块更新
        current_rel_emb = F.normalize(self.emb_rel) if self.layer_norm else self.emb_rel
        rel_embs_history = []
        
        # 步骤3：将历史窗口中的多快照合并为单图，并在边上保留重映射时间戳
        history_graph = merge_graphs(self.num_ents, g_list, use_cuda, self.gpu)
        if use_cuda:
            history_graph = history_graph.to(self.gpu)
        
        # 检查历史图是否有线图和概率矩阵信息
        has_line_graph = any(hasattr(g, 'line_graph_obj') and g.line_graph_obj is not None for g in g_list)
        has_pm_pd = any(hasattr(g, 'pm_pd') and g.pm_pd is not None for g in g_list)
        
        if has_line_graph and has_pm_pd:
            # 如果有线图和概率矩阵，将它们附加到合并的历史图中
            # 这里我们使用最后一个图的线图和概率矩阵作为代表
            for g in reversed(g_list):
                if hasattr(g, 'line_graph_obj') and g.line_graph_obj is not None:
                    history_graph.line_graph_obj = g.line_graph_obj
                    if use_cuda:
                        history_graph.line_graph_obj = history_graph.line_graph_obj.to(self.gpu)
                    break
            
            for g in reversed(g_list):
                if hasattr(g, 'pm_pd') and g.pm_pd is not None:
                    history_graph.pm_pd = g.pm_pd
                    if use_cuda and hasattr(g.pm_pd, 'to'):
                        history_graph.pm_pd = history_graph.pm_pd.to(self.gpu)
                    break

        # 步骤4：时序编码（Hawkes/Union/CompGCN 由 encoder_name 控制）
        new_ent_emb = current_ent_emb
        new_ent_emb, _ = self.rgcn.forward(history_graph, new_ent_emb, self.emb_rel)
        new_ent_emb = F.normalize(new_ent_emb) if self.layer_norm else new_ent_emb

        # 步骤5：全局重置门融合，抑制单轮历史传播带来的噪声漂移
        weight_vec = self.reset_gate(new_ent_emb).reshape(1, self.num_ents)
        weight = nn.functional.sigmoid(self.reset_gate1(weight_vec))
        new_ent_emb = new_ent_emb * weight + current_ent_emb * (1 - weight)

        # 步骤6：可选 ERD-Net 关系动态更新
        if self.use_relation_dynamics and hasattr(self, 'global_rel_dynamics') and self.global_rel_dynamics is not None:
            try:
                updated_rel_emb = self.global_rel_dynamics(g_list, current_rel_emb, new_ent_emb, time_idx=0)
                if updated_rel_emb is not None and not torch.isnan(updated_rel_emb).any():
                    current_rel_emb = updated_rel_emb
                    rel_embs_history.append(current_rel_emb)
                    # 只在特定条件下打印成功信息，避免干扰进度条
                    if torch.rand(1).item() < 0.01:  # 1%的概率打印
                        from tqdm import tqdm
                        tqdm.write("Relation dynamics updated successfully")
                else:
                    print("Invalid relation dynamics output, using original embeddings")
                    rel_embs_history.append(current_rel_emb)
            except Exception as e:
                print(f"Error in relation dynamics: {e}")
                rel_embs_history.append(current_rel_emb)
        else:
            rel_embs_history.append(current_rel_emb)

        history_embs.append(new_ent_emb)
        graph_h_list.append(self.graph_h)

        return history_embs, rel_embs_history[-1] if rel_embs_history else self.emb_rel, graph_h_list
        
        # #print(len(new_g_list))
        # for i, g in enumerate(new_g_list):
        #     if use_cuda:
        #         g = g.to(self.gpu)
        #     #print(self.h, self.h_0)
        #     current_ent_emb, current_rel_emb = self.rgcn.forward(g, self.emb_ent, self.emb_rel)  #图上信息传播
        #     current_ent_emb = F.normalize(current_ent_emb) if self.layer_norm else current_ent_emb
        #     self.graph_h = (self.node2graph(current_ent_emb)).sum(0, keepdim=True)      #计算图嵌入向量
        #     self.graph_h = F.normalize(self.graph_h, dim=1)
        #     gru_hidden = self.evolve(self.graph_h, gru_hidden)      #进行状态演化
        #     #print(gru_hidden)
            
        #     evolve_matrix = self.gen_matrix(gru_hidden).reshape(self.h_dim, self.h_dim) #生成状态演化矩阵
        #     evolve_matrix = evolve_matrix + 0.5
        #     current_ent_emb = torch.mm(current_ent_emb, evolve_matrix)      #更新节点状态
        #     current_ent_emb = F.normalize(current_ent_emb) if self.layer_norm else current_ent_emb
        #     if self.encoder_name == 'compgcn':
        #         current_rel_emb = torch.mm(current_rel_emb, evolve_matrix)         #更新边状态
        #     else:
        #         current_rel_emb = torch.mm(self.emb_rel, evolve_matrix)
        #     current_rel_emb = F.normalize(current_rel_emb) if self.layer_norm else current_rel_emb

        #     current_ent_emb = current_ent_emb * self.weight + self.emb_ent * (1 - self.weight)
        #     current_rel_emb = current_rel_emb * self.weight + self.emb_rel * (1 - self.weight)
            
        #     history_embs.append(current_ent_emb)
        #     self.h = current_ent_emb
        #     self.h_0 = current_rel_emb
        # return history_embs, self.h_0, gate_list, degree_list

    # ---------------- 基于 FFT 的关系分解与频率正则 ----------------
    def _relation_fft_components(self):
        # 关系嵌入频域分解：低频保留长期趋势，高频反映短期波动
        rel = self.emb_rel  # (num_rels*2, h_dim)
        freq_domain = fft.fft(rel, dim=1)
        freqs = fft.fftfreq(self.h_dim, d=1.0).to(rel.device)
        abs_freqs = freqs.abs()
        # alpha 控制低通强度，显式拆分低/高频成分
        low_mask = torch.exp(-abs_freqs * self.alpha).view(1, -1)
        high_mask = 1.0 - low_mask
        low_freq = freq_domain * low_mask
        high_freq = freq_domain * high_mask
        return low_freq, high_freq

    def relation_freq_reg(self):
        # 频率正则：鼓励低高频分量可分，同时惩罚高频能量过大
        low_freq, high_freq = self._relation_fft_components()
        # 分离项 + 高频惩罚，缓解频谱纠缠与噪声放大
        separation = -torch.norm(low_freq - high_freq, p=2)
        high_intensity = torch.norm(high_freq, p=2)
        norm_factor = 1.0 / (high_freq.shape[0] * high_freq.shape[1])
        return (separation + high_intensity) * norm_factor


    def predict(self, test_graph, num_rels, test_triplets, class_g, use_cuda):
        with torch.no_grad():
            # inverse_test_triplets = test_triplets[:, [2, 1, 0]]
            # inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels  # 将逆关系换成逆关系的id
            # all_triples = torch.cat((test_triplets, inverse_test_triplets))
            all_triples = test_triplets
            
            evolve_embs, r_emb, graph_embs = self.forward(test_graph, class_g, use_cuda)
            embedding = evolve_embs[-1]
            embedding = F.normalize(embedding) if self.layer_norm else embedding

            # 实体预测分支
            score = self.decoder_ob.forward(embedding, r_emb, all_triples, graph_embs[-1],mode="test")
            
            # 关系预测分支：基础解码 + 可选复制生成增强
            try:
                if self.use_copy_generation and hasattr(self, 'copy_gen_decoder') and self.copy_gen_decoder is not None:
                    # 使用 ERD-Net 复制-生成解码器增强关系预测
                    score_rel = self.rdecoder.forward(embedding, r_emb, all_triples, mode="test")
                    # 采用正确参数顺序调用复制-生成分支
                    copy_scores = self.copy_gen_decoder.forward(embedding, r_emb, all_triples, mode="test", use_copy=True)
                    if copy_scores is not None and not torch.isnan(copy_scores).any():
                        # 简单线性融合，并确保维度匹配
                        min_size = min(score_rel.size(0), copy_scores.size(0))
                        score_rel[:min_size] = score_rel[:min_size] + 0.3 * copy_scores[:min_size]
                else:
                    score_rel = self.rdecoder.forward(embedding, r_emb, all_triples, mode="test")
            except Exception as e:
                print(f"Error in copy generation prediction: {e}")
                score_rel = self.rdecoder.forward(embedding, r_emb, all_triples, mode="test")

            # 关系上下文先验以加性偏置方式叠加到关系 logits（结构归纳偏置）
            if self.use_rel_context_prior:
                s_idx = all_triples[:, 0]
                o_idx = all_triples[:, 2]
                x_s = embedding[s_idx]
                x_o = embedding[o_idx]
                pair_feat = torch.cat([x_s, x_o], dim=-1)
                prior_logits = self.rel_prior(pair_feat)
                score_rel = score_rel + self.rel_prior_weight * prior_logits
            return all_triples, score, score_rel


    def get_loss(self, glist, triples, class_g, use_cuda):
        """
        :param glist:
        :param triplets:
        :param use_cuda:
        :return:
        """
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_step = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        # inverse_triples = triples[:, [2, 1, 0]]
        # inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        # all_triples = torch.cat([triples, inverse_triples])
        all_triples = triples
        all_triples = all_triples.to(self.gpu)

        # 前向得到当前时刻实体表示与关系表示
        evolve_embs, r_emb,graph_embs = self.forward(glist, class_g, use_cuda)
        pre_emb = evolve_embs[-1]
        pre_emb = F.normalize(pre_emb) if self.layer_norm else pre_emb

        if self.entity_prediction:
            # 实体预测损失
            scores_ob = self.decoder_ob.forward(pre_emb, r_emb, all_triples, graph_embs[-1]).view(-1, self.num_ents)
            #print(scores_ob)
            loss_ent += self.loss_e(scores_ob, all_triples[:, 2])
     
        if self.relation_prediction:
            try:
                if self.use_copy_generation and hasattr(self, 'copy_gen_decoder') and self.copy_gen_decoder is not None:
                    # 使用 ERD-Net 复制-生成解码器
                    copy_loss = self.copy_gen_decoder.get_loss(pre_emb, r_emb, all_triples, use_copy=True)
                    if hasattr(copy_loss, 'item') and not torch.isnan(copy_loss) and not torch.isinf(copy_loss):
                        loss_rel += 0.1 * copy_loss  # 使用小权重避免影响过强
                        # 只在小概率下打印，避免干扰进度条
                        if torch.rand(1).item() < 0.01:  # 1%的概率打印
                            from tqdm import tqdm
                            tqdm.write(f"Copy generation loss: {copy_loss.item():.6f}")
                    else:
                        print("复制生成损失无效，已跳过")
                else:
                    # 原始关系预测路径（ConvTransR + 可选上下文先验）
                    score_rel = self.rdecoder.forward(pre_emb, r_emb, all_triples, mode="train").view(-1, 2 * self.num_rels)

                    # 关系上下文先验：显式建模实体对的关系偏好
                    if self.use_rel_context_prior:
                        s_idx = all_triples[:, 0]
                        o_idx = all_triples[:, 2]
                        x_s = pre_emb[s_idx]
                        x_o = pre_emb[o_idx]
                        pair_feat = torch.cat([x_s, x_o], dim=-1)
                        prior_logits = self.rel_prior(pair_feat)
                        score_rel = score_rel + self.rel_prior_weight * prior_logits
                    loss_rel += self.loss_r(score_rel, all_triples[:, 1])
            except Exception as e:
                print(f"关系预测出现异常: {e}")
                # 异常时回退到原始关系预测路径
                score_rel = self.rdecoder.forward(pre_emb, r_emb, all_triples, mode="train").view(-1, 2 * self.num_rels)
                loss_rel += self.loss_r(score_rel, all_triples[:, 1])

        # Lie 正则项并入关系损失分支
        if self.use_lie_reg:
            # 传入完整的 pre_emb 用于实体对对比学习（通过 triplets 索引）
            loss_lie = self.lie_regularizer(pre_emb, r_emb, all_triples)
            loss_rel = loss_rel + loss_lie

        return loss_ent, loss_rel
