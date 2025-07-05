import torch.nn as nn

class AutoencoderWithMLP(nn.Module):
    
    def reparameterize(self, mu, log_var): 
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def __init__(self, rna_input_dim, methylation_input_dim, dropout_prob=0.2,
                 enc_meth_hidden_dim=None,  # 甲基化编码器隐藏层维度
                 enc_common_dims=[256, 128],  # 公共编码器各层维度
                 latent_dim=50,  # 隐空间维度
                 dec_rna_dims=[128, 256],  # RNA解码器各层维度
                 dec_meth_hidden_dim=None,  # 甲基化解码器隐藏层维度
                 mlp_dims=[128, 64],  # MLP分类器各层维度
                 num_classes=21):  # 分类类别数
        super(AutoencoderWithMLP, self).__init__()

        # 如果没有特别指定，使用默认值（保持原结构）
        enc_meth_hidden_dim = rna_input_dim if enc_meth_hidden_dim is None else enc_meth_hidden_dim
        dec_meth_hidden_dim = methylation_input_dim if dec_meth_hidden_dim is None else dec_meth_hidden_dim
        
        # ========= 共有模块 ==========

        # 编码器1: 甲基化数据到 RNA 输入维度
        self.encoder_methylation = nn.Sequential(
            nn.Linear(methylation_input_dim, enc_meth_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(enc_meth_hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout_prob)
        )

        # 编码器2: RNA 数据到隐空间
        encoder_common_layers = []
        input_dim = rna_input_dim
        for i, dim in enumerate(enc_common_dims):
            encoder_common_layers.extend([
                nn.Linear(input_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout_prob)
            ])
            input_dim = dim
        
        self.encoder_common = nn.Sequential(*encoder_common_layers)
        
        self.fc_mu = nn.Linear(enc_common_dims[-1], latent_dim)      # 均值层
        self.fc_log_var = nn.Linear(enc_common_dims[-1], latent_dim) # 方差层
        
        
        # ========= 甲基化私有模块 ==========

        # 编码器（与 encoder_methylation + encoder_common 相同结构）
        self.encoder_meth_private = nn.Sequential(
            nn.Linear(methylation_input_dim, enc_meth_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(enc_meth_hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout_prob),
        )

        encoder_meth_private_common_layers = []
        input_dim = enc_meth_hidden_dim
        for dim in enc_common_dims:
            encoder_meth_private_common_layers.extend([
                nn.Linear(input_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout_prob)
            ])
            input_dim = dim

        self.encoder_meth_private_common = nn.Sequential(*encoder_meth_private_common_layers)
        self.fc_mu_meth_private = nn.Linear(enc_common_dims[-1], latent_dim)
        self.fc_log_var_meth_private = nn.Linear(enc_common_dims[-1], latent_dim)

        

        # ========= RNA 私有模块 ==========

        # 编码器（与 encoder_common 相同）
        encoder_rna_private_layers = []
        input_dim = rna_input_dim
        for dim in enc_common_dims:
            encoder_rna_private_layers.extend([
                nn.Linear(input_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout_prob)
            ])
            input_dim = dim
        self.encoder_rna_private = nn.Sequential(*encoder_rna_private_layers)
        self.fc_mu_rna_private = nn.Linear(enc_common_dims[-1], latent_dim)
        self.fc_log_var_rna_private = nn.Linear(enc_common_dims[-1], latent_dim)
        
        
         # ========= 解码器模块 ==========
        
        
         # 解码器1: 隐空间到 RNA 数据
        decoder_rna_layers = []
        input_dim = latent_dim
        for i, dim in enumerate(dec_rna_dims):
            decoder_rna_layers.extend([
                nn.Linear(input_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout_prob)
            ])
            input_dim = dim
        decoder_rna_layers.append(nn.Linear(dec_rna_dims[-1], rna_input_dim))
        decoder_rna_layers.append(nn.Sigmoid())
        
        self.decoder_rna = nn.Sequential(*decoder_rna_layers)

        # 解码器2: RNA 输入维度到甲基化数据
        self.decoder_methylation = nn.Sequential(
            nn.Linear(rna_input_dim, dec_meth_hidden_dim),
            nn.BatchNorm1d(dec_meth_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Sigmoid()
        )
        
        
        # ========= 分类器模块 ==========
    
        # MLP分类器
        mlp_layers = []
        input_dim = latent_dim
        for i, dim in enumerate(mlp_dims):
            mlp_layers.extend([
                nn.Linear(input_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU()
            ])
            input_dim = dim
        mlp_layers.append(nn.Linear(mlp_dims[-1], num_classes))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        
    def forward(self, x, data_type):
     if data_type == 'methylation':
        # === 公有分支 ===
        shared_x = self.encoder_methylation(x)
        common_out = self.encoder_common(shared_x)
        mu = self.fc_mu(common_out)
        log_var = self.fc_log_var(common_out)
        z_shared = self.reparameterize(mu, log_var)

        # === 私有分支 ===
        private_x = self.encoder_meth_private(x)
        private_common_out = self.encoder_meth_private_common(private_x)
        mu_private = self.fc_mu_meth_private(private_common_out)
        log_var_private = self.fc_log_var_meth_private(private_common_out)
        z_private = self.reparameterize(mu_private, log_var_private)

        # === 拼接后的隐空间重构 ===
        z_concat = torch.cat([z_shared, z_private], dim=1)

        
        decoded_rna = self.decoder_rna(z_concat)
        decoded_methylation = self.decoder_methylation(decoded_rna)

        mlp_output = self.mlp(z_concat)

        return {
            'decoded_rna': decoded_rna,
            'decoded_methylation': decoded_methylation,
            'mlp_output': mlp_output,
            'mu': mu,
            'log_var': log_var,
            'mu_private': mu_private,
            'log_var_private': log_var_private
        }

     elif data_type == 'rna':
        # === 公有分支 ===
        common_out = self.encoder_common(x)
        mu = self.fc_mu(common_out)
        log_var = self.fc_log_var(common_out)
        z_shared = self.reparameterize(mu, log_var)

        # === 私有分支 ===
        private_out = self.encoder_rna_private(x)
        mu_private = self.fc_mu_rna_private(private_out)
        log_var_private = self.fc_log_var_rna_private(private_out)
        z_private = self.reparameterize(mu_private, log_var_private)

        # === 拼接后的隐空间重构 ===
        z_concat = torch.cat([z_shared, z_private], dim=1)

        decoded_rna = self.decoder_rna(z_concat)
        mlp_output = self.mlp(z_concat)

        return {
            'decoded_rna': decoded_rna,
            'mlp_output': mlp_output,
            'mu': mu,
            'log_var': log_var,
            'mu_private': mu_private,
            'log_var_private': log_var_private
        }
    def loss_Ldifference(z_shared, z_private):
     """
     Ldifference loss: 保持共享特征与私有特征正交。

     参数:
     - z_shared: Tensor, 大小为 [batch_size, latent_dim]
     - z_private: Tensor, 大小为 [batch_size, latent_dim]

     返回:
     - 正交损失（越小越正交）
     """
     # 保证均值为0，避免量纲影响
     z_shared_norm = z_shared - z_shared.mean(dim=0, keepdim=True)
     z_private_norm = z_private - z_private.mean(dim=0, keepdim=True)

     # 单位归一化：z ⋅ z = 1
     z_shared_norm = F.normalize(z_shared_norm, dim=1)
     z_private_norm = F.normalize(z_private_norm, dim=1)
 
     # 计算 batch 内每个样本的共享与私有向量点积，再取均值
     dot_product = torch.sum(z_shared_norm * z_private_norm, dim=1)  # [batch_size]
     loss = torch.mean(dot_product ** 2)  # 趋近于0表示正交

     return loss

import torch.nn.functional as F

def loss_Lrecon(reconstructed, target):
    """
    Lrecon: 用于重构任务的 MSE 损失（均方误差）

    参数:
    - reconstructed: Tensor，模型重构输出
    - target: Tensor，原始输入数据（RNA 或 methylation）

    返回:
    - MSE 损失（值越小表示重构效果越好）
    """
    return F.mse_loss(reconstructed, target)


def loss_Lkl(mu, log_var):
    """
    Lkl: 变分自编码器中的 KL 散度，用于鼓励潜在分布接近标准正态分布 N(0,1)

    参数:
    - mu: 均值向量（batch_size × latent_dim）
    - log_var: 对数方差向量（batch_size × latent_dim）

    返回:
    - KL 散度损失（平均值）
    """
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    return torch.mean(kld)  # 每个样本一个 KLD，再取 batch 平均


import torch
import torch.nn.functional as F

def gaussian_kernel(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    计算高斯核矩阵
    x: (N, D)
    y: (M, D)
    return: (N+M, N+M)
    """
    n_samples = int(x.size(0)) + int(y.size(0))
    total = torch.cat([x, y], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)  # shape: (N+M, N+M)

def loss_Lsimilarity(h_c_s, h_c_t, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    MMD-based similarity loss between shared features of two modalities.

    参数:
    - h_c_s: source 共享隐变量 (如 methylation 模态) shape: (N^s, D)
    - h_c_t: target 共享隐变量 (如 RNA 模态) shape: (N^t, D)

    返回:
    - L_similarity: MMD 损失
    """
    batch_size = h_c_s.size(0)
    kernels = gaussian_kernel(h_c_s, h_c_t, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]

    loss = torch.mean(XX + YY - XY - YX)
    return loss




import torch.nn.functional as F

def loss_Lclassification(mlp_output, labels, data_type, methylation_has_label=None):
    """
    分类损失函数，根据模态处理方式不同：
    - RNA 模态：标准交叉熵损失
    - Methylation 模态：有 label mask 的情况，只对有标签样本计算损失并归一化

    参数:
    - mlp_output: 模型输出 logits, shape = (B, num_classes)
    - labels: ground truth 标签，shape = (B,)
    - data_type: 'rna' 或 'methylation'
    - methylation_has_label: shape = (B,), bool 或 0/1 张量，仅在 methylation 下有效

    返回:
    - classification_loss: 单个标量损失值
    """
    if data_type == 'rna':
        classification_loss = F.cross_entropy(mlp_output, labels)

    elif data_type == 'methylation':
        if methylation_has_label is None:
            raise ValueError("methylation_has_label must be provided for methylation data.")
        
        # 对有标签的样本计算交叉熵
        classification_losses = F.cross_entropy(mlp_output, labels)
        denominator = methylation_has_label.sum()
        classification_loss = (classification_losses.sum() / (denominator + (denominator == 0))) * (denominator != 0)

    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

    return classification_loss          





model = AutoencoderWithMLP(
    rna_input_dim=rna_input_dim,
    methylation_input_dim=methylation_input_dim,
    dropout_prob=0.2,
    enc_meth_hidden_dim=512,        # 自定义隐藏层维度
    enc_common_dims=[256, 128],
    latent_dim=50,
    dec_rna_dims=[128, 256],
    dec_meth_hidden_dim=512,        # 自定义解码器隐藏层维度
    mlp_dims=[128, 64],
    num_classes=21
)

# ========== 初始化损失记录 ========== #
history = {
    'rna_recon': [],
    'meth_recon': [],
    'rna_kl': [],
    'meth_kl': [],
    'classification': [],
    'difference': [],
    'similarity': [],
    'total': []
}


# 可调节权重超参数
lambda_rna_recon = 1.0
lambda_meth_recon = 1.0
lambda_rna_kl = 0.1
lambda_meth_kl = 0.1
lambda_cls = 1.0
lambda_diff = 0.1
lambda_sim = 1.0

# ========== 训练循环 ========== #
for epoch in range(num_epochs): 
    model.train()
    combined_dataloader = zip(rna_dataloader, methylation_dataloader)

    # 初始化 epoch 累加器
    total_rna_recon = 0.0
    total_meth_recon = 0.0
    total_rna_kl = 0.0
    total_meth_kl = 0.0
    total_classification_loss = 0.0
    total_difference_loss = 0.0
    total_similarity_loss = 0.0
    total_loss = 0.0

    for batch_idx, (rna_batch, meth_batch) in enumerate(combined_dataloader):
        optimizer.zero_grad()

        # === RNA 模态 ===
        rna_data = rna_batch['data'].to(device)
        rna_labels = rna_batch['one_hot'].to(device)

        rna_out = model(rna_data, 'rna')

        rna_recon_loss = loss_Lrecon(rna_out['decoded_rna'], rna_data)
        rna_kl_loss = loss_Lkl(rna_out['mu'], rna_out['log_var']) + \
                      loss_Lkl(rna_out['mu_private'], rna_out['log_var_private'])
        rna_cls_loss = loss_Lclassification(rna_out['mlp_output'], rna_labels, data_type='rna')
        rna_diff_loss = loss_Ldifference(rna_out['mu'], rna_out['mu_private'])

        # === 甲基化模态 ===
        meth_data = meth_batch['data'].to(device)
        meth_labels = meth_batch['one_hot'].to(device)
        meth_has_label = meth_batch['methylation_has_label'].to(device)

        meth_out = model(meth_data, 'methylation')

        meth_recon_loss = loss_Lrecon(meth_out['decoded_methylation'], meth_data)
        meth_kl_loss = loss_Lkl(meth_out['mu'], meth_out['log_var']) + \
                       loss_Lkl(meth_out['mu_private'], meth_out['log_var_private'])
        meth_cls_loss = loss_Lclassification(meth_out['mlp_output'], meth_labels, data_type='methylation', methylation_has_label=meth_has_label)
        meth_diff_loss = loss_Ldifference(meth_out['mu'], meth_out['mu_private'])

        # === MMD 模态对齐损失 ===
        similarity_loss = loss_Lsimilarity(h_c_s=meth_out['mu'], h_c_t=rna_out['mu'])

        # === 加权总损失 ===
        total_batch_loss = (
            lambda_rna_recon * rna_recon_loss +
            lambda_meth_recon * meth_recon_loss +
            lambda_rna_kl * rna_kl_loss +
            lambda_meth_kl * meth_kl_loss +
            lambda_cls * (rna_cls_loss + meth_cls_loss) +
            lambda_diff * (rna_diff_loss + meth_diff_loss) +
            lambda_sim * similarity_loss
        )

        # 反向传播与优化
        total_batch_loss.backward()
        optimizer.step()

        # 累加统计
        total_rna_recon += rna_recon_loss.item()
        total_meth_recon += meth_recon_loss.item()
        total_rna_kl += rna_kl_loss.item()
        total_meth_kl += meth_kl_loss.item()
        total_classification_loss += rna_cls_loss.item() + meth_cls_loss.item()
        total_difference_loss += rna_diff_loss.item() + meth_diff_loss.item()
        total_similarity_loss += similarity_loss.item()
        total_loss += total_batch_loss.item()
        

        # === 每个 batch 输出 ===
        print(f"[Epoch {epoch+1}] Batch {batch_idx+1}: "
              f"RNA Recon: {rna_recon_loss.item():.4f}, Meth Recon: {meth_recon_loss.item():.4f}, "
              f"RNA KL: {rna_kl_loss.item():.4f}, Meth KL: {meth_kl_loss.item():.4f}, "
              f"Cls: {(rna_cls_loss.item() + meth_cls_loss.item()):.4f}, "
              f"Diff: {(rna_diff_loss.item() + meth_diff_loss.item()):.4f}, "
              f"MMD: {similarity_loss.item():.4f}, Total: {total_batch_loss.item():.4f}")
    # ========== 记录 epoch 平均损失 ==========
    num_batches = len(rna_dataloader)

    history['rna_recon'].append(total_rna_recon / num_batches)
    history['meth_recon'].append(total_meth_recon / num_batches)
    history['rna_kl'].append(total_rna_kl / num_batches)
    history['meth_kl'].append(total_meth_kl / num_batches)
    history['classification'].append(total_classification_loss / num_batches)
    history['difference'].append(total_difference_loss / num_batches)
    history['similarity'].append(total_similarity_loss / num_batches)
    history['total'].append(total_loss / num_batches)

    # === 每个 epoch 的平均损失 ===
    print(f"[Epoch {epoch+1} Summary] "
          f"RNA Recon: {total_rna_recon:.4f}, Meth Recon: {total_meth_recon:.4f}, "
          f"RNA KL: {total_rna_kl:.4f}, Meth KL: {total_meth_kl:.4f}, "
          f"Cls: {total_classification_loss:.4f}, "
          f"Diff: {total_difference_loss:.4f}, "
          f"MMD: {total_similarity_loss:.4f}, "
          f"Total: {total_loss:.4f}")


