import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
import random


class MultiFocalLoss(nn.Module):
    """
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor shape=[num_class, ]
        gamma: hyper-parameter
        reduction: reduction type
    """

    def __init__(self, num_class, alpha=None, gamma=2, reduction='mean'):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-6
        self.alpha = alpha
        if alpha is None:
            self.alpha = torch.ones(num_class, ) - 0.5
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        if self.alpha.shape[0] != num_class:
            raise RuntimeError('the length not equal to number of class')

    def forward(self, logit, target):
        # assert isinstance(self.alpha,torch.Tensor)\
        alpha = self.alpha.to(logit.device)
        prob = F.softmax(logit, dim=1)

        if prob.dim() > 2:
            # used for 3d-conv:  N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            N, C = logit.shape[:2]
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]

        ori_shp = target.shape
        target = target.view(-1, 1)

        prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan
        logpt = torch.log(prob)
        # alpha_class = alpha.gather(0, target.squeeze(-1))
        alpha_weight = alpha[target.squeeze().long()]
        loss = -alpha_weight * torch.pow(torch.sub(1.0, prob), self.gamma) * logpt

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)

        return loss


class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=2,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
            self,
            layers_hidden,
            grid_size=3,
            spline_order=3,
            scale_noise=0.05,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.LeakyReLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        print("KAN is being initialized...")
        print("grid_size:", grid_size)
        print("spline_order:", spline_order)
        print("scale_noise:", scale_noise)
        print("scale_base:", scale_base)
        print("scale_spline:", scale_spline)
        print("base_activation:", base_activation)
        print("grid_eps:", grid_eps)
        print("grid_range:", grid_range)

        self.layers = torch.nn.ModuleList()

        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        # 使用 KAN 模型替换原本的单层线性映射
        layers_hidden = [config.feature_size, config.hidden_size]
        self.manual_dense = KAN(layers_hidden)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 输出层保持不变
        self.out_proj_new = nn.Linear(config.hidden_size + config.hidden_size, 2)

    def forward(self, features, manual_features=None, **kwargs):
        x = features[:, 0, :]
        y = manual_features.float()
        # 通过 KAN 模型进行特征映射
        y = self.manual_dense(y)
        y = torch.tanh(y)

        x = torch.cat((x, y), dim=-1)
        x = self.dropout(x)
        x = self.out_proj_new(x)
        return x



class Attention(nn.Module):       #x:[batch, seq_len, hidden_dim*2]
    """
        此注意力的计算步骤：
        1.将输入（包含lstm的所有时刻的状态输出）和w矩阵进行矩阵相乘，然后用tanh压缩到(-1, 1)之间
        2.然后再和矩阵u进行矩阵相乘后，矩阵变为1维，然后进行softmax变化即得到注意力得分。
        3.将输入和此注意力得分线性加权，即相当于将所有时刻的状态进行了一个聚合。
    """
    def __init__(self, hidden_size, need_aggregation=True):
        super().__init__()
        self.need_aggregation = need_aggregation
        # 不双向的话就不用乘2
        self.w = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size * 2))
        self.u = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
        nn.init.uniform_(self.w, -0.1, 0.1)
        nn.init.uniform_(self.u, -0.1, 0.1)

    def forward(self, x):
        device = x.device
        self.w = self.w.to(device)
        self.u = self.u.to(device)

        u = torch.tanh(torch.matmul(x, self.w))         #[batch, seq_len, hidden_size*2]
        score = torch.matmul(u, self.u)                   #[batch, seq_len, 1]
        att = F.softmax(score, dim=1)
        # 下面操作即线性加权
        scored_x = x * att                              #[batch, seq_len, hidden_size*2]

        # 因为词encoder和句encoder后均带有attention机制，而我需要做的是代码行级缺陷检测，
        # 所以句encoder后我不做聚合，相当于将每个代码行看做一个样本来传入全连接分类。
        if self.need_aggregation:
            context = torch.sum(scored_x, dim=1)                  #[batch, hidden_size*2]
            return context
        else:
            return scored_x



class HAN_MODEL(nn.Module):
    def __init__(self, embedding_layer):
        super().__init__()


        self.hidden_size = 256
        self.num_layers = 1
        self.bidirectional = True

        self.embedding = embedding_layer



        self.lstm1 = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bidirectional=self.bidirectional,
                            batch_first=True)
        self.att1 = Attention(self.hidden_size, need_aggregation=True)
        self.lstm2 = nn.LSTM(input_size=self.hidden_size*2,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bidirectional=self.bidirectional,
                            batch_first=True)
        self.att2 = Attention(self.hidden_size, need_aggregation=False)

        # 代码行级分类输出层，代码有多少行，输出就有多少个神经元
        # self.fc1 = nn.Linear(512, 2)
        self.fc1 = nn.Linear(512, 128)
        self.relu = nn.Tanh()
        self.fc2 = nn.Linear(128, 2)
        # self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        # input x : (bs, nus_sentences, nums_words)
        device = x.device
        x = self.embedding(x) # out x : (bs, nus_sentences, nums_words, embedding_dim)
        x = self.dropout(x)
        batch_size, num_sentences, num_words, emb_dim = x.shape

        # 初始化：双向就乘2
        h0_1 = torch.randn(self.num_layers*2, batch_size*num_sentences, self.hidden_size).to(device)
        c0_1 = torch.randn(self.num_layers*2, batch_size*num_sentences, self.hidden_size).to(device)
        h0_2 = torch.randn(self.num_layers*2, batch_size, self.hidden_size).to(device)
        c0_2 = torch.randn(self.num_layers*2, batch_size, self.hidden_size).to(device)

        # 先进入word的处理，将每句话的所有单词的表示通过attention聚合成一个表示
        # 将batch_size, num_sentences两个维度乘起来看成“batch_size”。使用.contiguous()方法首先拷贝了一份张量在内存中的地址，然后将地址按照形状改变后的张量的语义进行排列。
        # touch.view()方法对张量改变“形状”其实并没有改变张量在内存中真正的形状
        x = x.view(batch_size*num_sentences, num_words, emb_dim).contiguous()
        x,(_,_)= self.lstm1(x, (h0_1,c0_1))   # out：batch_size*num_sentences, num_words，hidden_size*2
        x = self.att1(x)   # 线性加权注意力后的输出：batch_size*num_sentences, hidden_size*2

        x = x.view(x.size(0)//num_sentences, num_sentences, self.hidden_size*2).contiguous()
        x,(_,_)= self.lstm2(x, (h0_2,c0_2))   # out：batch_size, num_sentences，hidden_size*2

        x = self.att2(x)   # 线性加权注意力后的输出：batch_size, hidden_size*2
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args

        self.han_word_embedding_layer = self.encoder.embeddings.word_embeddings
        self.han_locator = HAN_MODEL(embedding_layer=self.han_word_embedding_layer)

    def forward(self, inputs_ids, attn_masks, manual_features,
                labels, line_ids, line_label, output_attentions=None):
        outputs = self.encoder(input_ids=inputs_ids, attention_mask=attn_masks, output_attentions=output_attentions)
        last_layer_attn_weights = outputs.attentions[self.config.num_hidden_layers - 1][:, :,
                                  0].detach() if output_attentions else None

        logits = self.classifier(outputs[0], manual_features)
        han_logits = self.han_locator(line_ids)

        logits = (logits + han_logits.mean(dim=1)) / 2

        if labels is not None:
            loss_dp = MultiFocalLoss(alpha=0.25, gamma=2, reduction='mean', num_class=2)
            loss1 = loss_dp(logits, labels)

            loss_dl = MultiFocalLoss(alpha=0.25, gamma=2, reduction='mean', num_class=2)
            loss2 = loss_dl(han_logits.reshape((-1, 2)), line_label.reshape((-1,)))

            loss = loss1 * self.args.dp_loss_weight + loss2 * self.args.dl_loss_weight

            return loss, torch.softmax(logits, dim=1)[:, 1].unsqueeze(1), last_layer_attn_weights, torch.softmax(han_logits, dim=-1)[:, :, 1]
        else:
            return torch.softmax(logits, dim=1)[:, 1].unsqueeze(1)