import argparse
import os
import csv
import random
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from monai.transforms import (
    Compose, RandFlipd, RandRotate90d, RandGaussianNoised,
    RandAffined, RandShiftIntensityd, ToTensord
)
from monai.utils import set_determinism
import torch._dynamo
import torch.nn.functional as F
from typing import Sequence, Tuple, Union

torch._dynamo.config.suppress_errors = True


# =============================================================================
# DATASET
# =============================================================================

class PreprocessedDataset(Dataset):
    """Dataset for preprocessed cropped images with channel selection"""
    def __init__(self, file_paths, pni_labels, pids, transform=None, selected_channels=None):
        self.file_paths = file_paths
        self.pni_labels = pni_labels
        self.pids = pids
        self.transform = transform
        self.selected_channels = selected_channels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = np.load(self.file_paths[idx])
        image = torch.from_numpy(image).float()

        if self.selected_channels is not None:
            image = image[self.selected_channels, :, :, :]

        if self.transform:
            image = self.transform({"input": image})["input"]

        label = torch.tensor(self.pni_labels[idx], dtype=torch.float32)
        pid = self.pids[idx]

        return image, label, pid


def load_preprocessed_data_with_folds(preprocessed_dir, folds_csv_path, val_fold=0):
    """Load preprocessed data with fold-based splitting"""
    folds_df = pd.read_csv(folds_csv_path)
    print(f"Loaded fold information: {len(folds_df)} entries")

    if 'fold' not in folds_df.columns or 'ID' not in folds_df.columns or 'PNI' not in folds_df.columns:
        raise ValueError(f"Required columns not found in {folds_csv_path}")

    print(f"Available folds: {sorted(folds_df['fold'].unique())}")
    print(f"Validation fold {val_fold}: {(folds_df['fold'] == val_fold).sum()} samples")

    preprocessed_files = [f for f in os.listdir(preprocessed_dir) if f.endswith('.npy')]
    print(f"Found {len(preprocessed_files)} preprocessed files")

    fold_dict = dict(zip(folds_df['ID'].astype(str), folds_df['fold']))
    pni_dict = dict(zip(folds_df['ID'].astype(str), folds_df['PNI']))

    val_fold_ids = set(folds_df[folds_df['fold'] == val_fold]['ID'].astype(str))
    training_folds = [f for f in range(6) if f != val_fold]
    train_fold_ids = set(folds_df[folds_df['fold'].isin(training_folds)]['ID'].astype(str))

    train_paths, train_labels, train_pids = [], [], []
    val_paths, val_labels, val_pids = [], [], []

    for patient_id in folds_df['ID'].astype(str):
        file_path_1 = os.path.join(preprocessed_dir, f'patient_{patient_id}.npy')
        file_path_2 = os.path.join(preprocessed_dir, f'patient{patient_id}.npy')

        if os.path.exists(file_path_1):
            file_path = file_path_1
        elif os.path.exists(file_path_2):
            file_path = file_path_2
        else:
            continue

        pni = pni_dict[patient_id]

        if patient_id in val_fold_ids:
            val_paths.append(file_path)
            val_labels.append(pni)
            val_pids.append(patient_id)
        elif patient_id in train_fold_ids:
            train_paths.append(file_path)
            train_labels.append(pni)
            train_pids.append(patient_id)

    print(f"\nTraining: {len(train_paths)} samples (folds {training_folds})")
    print(f"Validation: {len(val_paths)} samples (fold {val_fold})")

    return train_paths, train_labels, train_pids, val_paths, val_labels, val_pids


# =============================================================================
# WINDOW PARTITIONING
# =============================================================================

class WindowPartitioner:
    """Optimized window partitioning with caching"""
    def __init__(self, max_cache_size=10):
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.access_order = []

    def _clean_cache(self):
        if len(self.cache) >= self.max_cache_size:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]

    def calculate_padding(self, D, H, W, window_size):
        Wd, Wh, Ww = window_size
        pad_d = (Wd - D % Wd) % Wd
        pad_h = (Wh - H % Wh) % Wh
        pad_w = (Ww - W % Ww) % Ww
        return (pad_d, pad_h, pad_w)

    def partition(self, x, window_size):
        B, D, H, W, C = x.shape
        Wd, Wh, Ww = window_size

        cache_key = (B, D, H, W, C, Wd, Wh, Ww)
        
        if cache_key not in self.cache:
            self._clean_cache()
            self.cache[cache_key] = {
                'view_shape': (B, D // Wd, Wd, H // Wh, Wh, W // Ww, Ww, C),
                'permute_dims': (0, 1, 3, 5, 2, 4, 6, 7),
                'final_shape': (-1, Wd * Wh * Ww, C),
                'reverse_shape': (B, D // Wd, H // Wh, W // Ww, Wd, Wh, Ww, -1),
                'reverse_permute': (0, 1, 4, 2, 5, 3, 6, 7),
                'reverse_final': (B, D, H, W, -1)
            }
            self.access_order.append(cache_key)
        else:
            self.access_order.remove(cache_key)
            self.access_order.append(cache_key)

        cached_info = self.cache[cache_key]

        x_viewed = x.view(cached_info['view_shape'])
        x_permuted = x_viewed.permute(cached_info['permute_dims']).contiguous()
        windows = x_permuted.view(cached_info['final_shape'])

        return windows, cached_info

    def reverse(self, windows, cache_info, B, D, H, W):
        x = windows.view(cache_info['reverse_shape'])
        x = x.permute(cache_info['reverse_permute']).contiguous()
        x = x.view(cache_info['reverse_final'])
        return x


window_partitioner = WindowPartitioner()


def window_partition_optimized(x, window_size):
    return window_partitioner.partition(x, window_size)


def window_reverse_optimized(windows, cache_info, B, D, H, W):
    return window_partitioner.reverse(windows, cache_info, B, D, H, W)


# =============================================================================
# MOH WINDOW ATTENTION 3D
# =============================================================================

class MoHWindowAttention3D(nn.Module):
    """MoH 3D window attention"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., 
                 proj_drop=0., moh_efficiency=0.75, num_shared_heads=2):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.moh_efficiency = moh_efficiency

        self.num_shared_heads = min(num_shared_heads, num_heads)
        self.num_routed_heads = num_heads - self.num_shared_heads

        if self.num_routed_heads > 0:
            self.top_k = max(1, int(self.num_routed_heads * moh_efficiency))
        else:
            self.top_k = 0

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.num_routed_heads > 0:
            self.router = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(dim // 4, self.num_routed_heads),
            )
            self.alpha_router = nn.Linear(dim, 2)

        self.dim_proj = None

    def forward(self, x, return_load_balance_loss=False):
        B_, N, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(B_, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale

        load_balance_loss = torch.tensor(0.0, device=x.device)

        if self.num_routed_heads > 0:
            window_repr = x.mean(dim=1)
            router_logits = self.router(window_repr)
            router_probs = F.softmax(router_logits, dim=-1)

            top_k_probs, selected_routed_indices = torch.topk(router_probs, self.top_k, dim=-1)
            top_k_weights = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

            alpha_logits = self.alpha_router(window_repr)
            alpha = F.softmax(alpha_logits, dim=-1)
            alpha_shared, alpha_routed = alpha[:, 0:1], alpha[:, 1:2]

            if self.num_shared_heads > 0:
                shared_q = q[:, :self.num_shared_heads]
                shared_k = k[:, :self.num_shared_heads]
                shared_v = v[:, :self.num_shared_heads]

                shared_attn = torch.matmul(shared_q, shared_k.transpose(-2, -1))
                shared_attn = F.softmax(shared_attn, dim=-1)
                shared_attn = self.attn_drop(shared_attn)

                shared_output = torch.matmul(shared_attn, shared_v)
                shared_output = shared_output.sum(dim=1)
                shared_output = shared_output * alpha_shared.unsqueeze(-1)
            else:
                shared_output = torch.zeros(B_, N, self.head_dim, device=x.device)

            if self.top_k > 0:
                batch_indices = torch.arange(B_, device=x.device).unsqueeze(1).expand(-1, self.top_k)
                routed_head_indices = selected_routed_indices + self.num_shared_heads

                routed_q = q[batch_indices, routed_head_indices]
                routed_k = k[batch_indices, routed_head_indices]
                routed_v = v[batch_indices, routed_head_indices]

                routed_attn = torch.matmul(routed_q, routed_k.transpose(-2, -1))
                routed_attn = F.softmax(routed_attn, dim=-1)
                routed_attn = self.attn_drop(routed_attn)

                routed_output = torch.matmul(routed_attn, routed_v)

                combined_weights = (top_k_weights * alpha_routed).unsqueeze(-1).unsqueeze(-1)
                routed_output = routed_output * combined_weights
                routed_output = routed_output.sum(dim=1)
            else:
                routed_output = torch.zeros(B_, N, self.head_dim, device=x.device)

            combined_output = shared_output + routed_output

            if return_load_balance_loss and self.num_routed_heads > 0:
                P_i = router_probs.mean(dim=0)
                f_i = torch.zeros(self.num_routed_heads, device=x.device)
                for i in range(self.num_routed_heads):
                    f_i[i] = (selected_routed_indices == i).float().mean()
                load_balance_loss = torch.sum(P_i * f_i) * self.num_routed_heads

        else:
            attn = torch.matmul(q, k.transpose(-2, -1))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)

            head_outputs = torch.matmul(attn, v)
            combined_output = head_outputs.transpose(1, 2).contiguous().view(B_, N, -1)

        if combined_output.size(-1) != C:
            if self.dim_proj is None:
                self.dim_proj = nn.Linear(combined_output.size(-1), C).to(x.device)
            combined_output = self.dim_proj(combined_output)

        output = self.proj(combined_output)
        output = self.proj_drop(output)

        if return_load_balance_loss:
            return output, load_balance_loss
        return output


# =============================================================================
# MOH LGT BLOCK
# =============================================================================

class MoHLGTBlock(nn.Module):
    """MoH Local-Global Transformer block"""
    def __init__(self, dim, input_resolution, num_heads, window_size_local=(3,3,3), 
                 window_size_global=(6,6,3), shift_size=0, mlp_ratio=2., 
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0., 
                 norm_layer=nn.LayerNorm, moh_efficiency=0.75, num_shared_heads=2):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size_local = window_size_local
        self.window_size_global = window_size_global
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1_local = norm_layer(dim)
        self.attn_local = MoHWindowAttention3D(
            dim, window_size=self.window_size_local, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            moh_efficiency=moh_efficiency, num_shared_heads=num_shared_heads)

        self.norm1_global = norm_layer(dim)
        self.attn_global = MoHWindowAttention3D(
            dim, window_size=self.window_size_global, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            moh_efficiency=moh_efficiency, num_shared_heads=num_shared_heads)

        self.fusion_proj = nn.Linear(dim * 2, dim)

        self.drop_path = nn.Identity() if drop_path == 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

        D, H, W = input_resolution
        self.pad_local = window_partitioner.calculate_padding(D, H, W, self.window_size_local)
        self.pad_global = window_partitioner.calculate_padding(D, H, W, self.window_size_global)

    def forward(self, x, return_load_balance_loss=False):
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)

        shortcut = x
        total_load_balance_loss = 0.0

        x_local = self.norm1_local(x)
        x_global = self.norm1_global(x)

        if any(p > 0 for p in self.pad_local):
            x_local = F.pad(x_local, (0, 0, 0, self.pad_local[2], 0, self.pad_local[1], 0, self.pad_local[0]))

        if any(p > 0 for p in self.pad_global):
            x_global = F.pad(x_global, (0, 0, 0, self.pad_global[2], 0, self.pad_global[1], 0, self.pad_global[0]))

        x_windows_local, local_cache = window_partition_optimized(x_local, self.window_size_local)
        x_windows_global, global_cache = window_partition_optimized(x_global, self.window_size_global)

        if return_load_balance_loss:
            attn_windows_local, local_load_loss = self.attn_local(x_windows_local, return_load_balance_loss=True)
            attn_windows_global, global_load_loss = self.attn_global(x_windows_global, return_load_balance_loss=True)
            total_load_balance_loss = local_load_loss + global_load_loss
        else:
            attn_windows_local = self.attn_local(x_windows_local)
            attn_windows_global = self.attn_global(x_windows_global)

        _, Dp_local, Hp_local, Wp_local, _ = x_local.shape
        _, Dp_global, Hp_global, Wp_global, _ = x_global.shape

        x_local = window_reverse_optimized(attn_windows_local, local_cache, B, Dp_local, Hp_local, Wp_local)
        x_global = window_reverse_optimized(attn_windows_global, global_cache, B, Dp_global, Hp_global, Wp_global)

        if any(p > 0 for p in self.pad_local):
            x_local = x_local[:, :D, :H, :W, :].contiguous()

        if any(p > 0 for p in self.pad_global):
            x_global = x_global[:, :D, :H, :W, :].contiguous()

        x_fused = self.fusion_proj(torch.cat([x_local, x_global], dim=-1))

        x = shortcut + self.drop_path(x_fused)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.permute(0, 4, 1, 2, 3)

        if return_load_balance_loss:
            return x, total_load_balance_loss
        return x


# =============================================================================
# SAF MODULE - FIXED VERSION
# =============================================================================

class SAF(nn.Module):
    """Spatial Attention Fusion - Fixed version with Cross Attention → Spatial Attention"""
    def __init__(self, in_channels_a, in_channels_b, out_channels):
        super().__init__()
        self.in_channels_a = in_channels_a
        self.in_channels_b = in_channels_b
        self.out_channels = out_channels
        
        common_dim = min(in_channels_a, in_channels_b)
        
        # Projections to common dimension
        self.proj_a = nn.Sequential(
            nn.Conv3d(in_channels_a, common_dim, 1),
            nn.BatchNorm3d(common_dim)
        )
        self.proj_b = nn.Sequential(
            nn.Conv3d(in_channels_b, common_dim, 1),
            nn.BatchNorm3d(common_dim)
        )
        
        # Cross-attention (FIRST)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=common_dim, 
            num_heads=4, 
            batch_first=True,
            dropout=0.1
        )
        
        # Spatial Attention (SECOND)
        self.spatial_attn_a = nn.Sequential(
            nn.Conv3d(common_dim, common_dim // 8, 1),
            nn.BatchNorm3d(common_dim // 8),
            nn.ReLU(inplace=True),
            nn.Conv3d(common_dim // 8, 1, 1),
            nn.Sigmoid()
        )
        
        self.spatial_attn_b = nn.Sequential(
            nn.Conv3d(common_dim, common_dim // 8, 1),
            nn.BatchNorm3d(common_dim // 8),
            nn.ReLU(inplace=True),
            nn.Conv3d(common_dim // 8, 1, 1),
            nn.Sigmoid()
        )
        
        # Channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(common_dim * 2, common_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(common_dim // 4, common_dim * 2, 1),
            nn.Sigmoid()
        )
        
        # Gated fusion
        self.gate = nn.Sequential(
            nn.Conv3d(common_dim * 2, common_dim * 2, 1),
            nn.Sigmoid()
        )
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Conv3d(common_dim * 2, out_channels, 1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual projection
        self.residual_proj = None
        if in_channels_a != out_channels:
            self.residual_proj = nn.Sequential(
                nn.Conv3d(in_channels_a, out_channels, 1),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, feat_a, feat_b):
        # Resize feat_b to match feat_a
        feat_b_resized = F.interpolate(
            feat_b, 
            size=feat_a.shape[2:], 
            mode='trilinear', 
            align_corners=False
        )
        
        identity = feat_a
        
        # Project to common dimension
        feat_a_proj = self.proj_a(feat_a)
        feat_b_proj = self.proj_b(feat_b_resized)
        
        B, C, D, H, W = feat_a_proj.shape
        
        # Step 1: Cross-attention (FIRST)
        feat_a_flat = feat_a_proj.flatten(2).permute(0, 2, 1)
        feat_b_flat = feat_b_proj.flatten(2).permute(0, 2, 1)
        
        attn_a, _ = self.cross_attn(feat_a_flat, feat_b_flat, feat_b_flat)
        attn_b, _ = self.cross_attn(feat_b_flat, feat_a_flat, feat_a_flat)
        
        attn_a = attn_a.permute(0, 2, 1).view(B, C, D, H, W)
        attn_b = attn_b.permute(0, 2, 1).view(B, C, D, H, W)
        
        # Residual connection
        feat_a_cross = feat_a_proj + attn_a
        feat_b_cross = feat_b_proj + attn_b
        
        # Step 2: Spatial attention (SECOND)
        spatial_weight_a = self.spatial_attn_a(feat_a_cross)
        spatial_weight_b = self.spatial_attn_b(feat_b_cross)
        
        # Apply with residual
        feat_a_spatial = feat_a_cross * (1 + spatial_weight_a)
        feat_b_spatial = feat_b_cross * (1 + spatial_weight_b)
        
        # Step 3: Fusion
        fused = torch.cat([feat_a_spatial, feat_b_spatial], dim=1)
        
        # Channel attention
        channel_weight = self.channel_attn(fused)
        fused = fused * channel_weight
        
        # Gated fusion
        gate = self.gate(fused)
        fused = fused * gate
        
        # Final projection
        output = self.final_proj(fused)
        
        # Residual connection
        if self.residual_proj is not None:
            identity = self.residual_proj(identity)
        
        output = output + identity
        
        return output


# =============================================================================
# NEONET MODEL
# =============================================================================

class NeoNet(nn.Module):
    """NeoNet with fixed SAF modules"""
    def __init__(self, in_channels=3, moh_efficiency=0.75, num_shared_heads=2):
        super().__init__()
        self.moh_efficiency = moh_efficiency
        self.num_shared_heads = num_shared_heads

        # Initial embedding
        self.patch_embed = nn.Conv3d(in_channels, 48, kernel_size=4, stride=4)

        # MoH LGT blocks
        self.moh_lgt_block1 = MoHLGTBlock(
            dim=48, input_resolution=(24, 24, 12), num_heads=6,
            window_size_local=(3, 3, 3), window_size_global=(6, 6, 6),
            moh_efficiency=moh_efficiency, num_shared_heads=num_shared_heads,
            mlp_ratio=2.0
        )

        self.downsample1 = nn.Conv3d(48, 96, kernel_size=2, stride=2)

        self.moh_lgt_block2 = MoHLGTBlock(
            dim=96, input_resolution=(12, 12, 6), num_heads=8,
            window_size_local=(3, 3, 3), window_size_global=(6, 6, 6),
            moh_efficiency=moh_efficiency, num_shared_heads=num_shared_heads,
            mlp_ratio=2.0
        )

        self.downsample2 = nn.Conv3d(96, 192, kernel_size=2, stride=2)

        self.moh_lgt_block3 = MoHLGTBlock(
            dim=192, input_resolution=(6, 6, 3), num_heads=12,
            window_size_local=(3, 3, 3), window_size_global=(6, 6, 3),
            moh_efficiency=moh_efficiency, num_shared_heads=num_shared_heads,
            mlp_ratio=2.0
        )

        # Fixed SAF modules
        self.saf1 = SAF(96, 48, 96)
        self.saf2 = SAF(192, 96, 192)

        # Classification head
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        hidden_dim = 64
        self.classifier = nn.Sequential(
            nn.Linear(192, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, return_load_balance_loss=False):
        x = self.patch_embed(x)

        total_load_balance_loss = 0.0

        if return_load_balance_loss:
            x1, lb_loss1 = self.moh_lgt_block1(x, return_load_balance_loss=True)
            total_load_balance_loss += lb_loss1
        else:
            x1 = self.moh_lgt_block1(x)

        x_down1 = self.downsample1(x1)

        if return_load_balance_loss:
            x2, lb_loss2 = self.moh_lgt_block2(x_down1, return_load_balance_loss=True)
            total_load_balance_loss += lb_loss2
        else:
            x2 = self.moh_lgt_block2(x_down1)

        x2_fused = self.saf1(x2, x1)

        x_down2 = self.downsample2(x2_fused)

        if return_load_balance_loss:
            x3, lb_loss3 = self.moh_lgt_block3(x_down2, return_load_balance_loss=True)
            total_load_balance_loss += lb_loss3
        else:
            x3 = self.moh_lgt_block3(x_down2)

        x3_fused = self.saf2(x3, x2_fused)

        features = self.global_pool(x3_fused).flatten(1)
        output = self.classifier(features).squeeze(1)

        if return_load_balance_loss:
            return output, total_load_balance_loss
        return output


# =============================================================================
# METRICS & PLOTTING
# =============================================================================

def compute_metrics(preds, trues):
    preds_bin = (preds >= 0.5).astype(int)
    acc = accuracy_score(trues, preds_bin)
    f1 = f1_score(trues, preds_bin, zero_division=0)
    dice = 2 * np.sum(preds_bin * trues) / (np.sum(preds_bin) + np.sum(trues) + 1e-8)
    try:
        auc = roc_auc_score(trues, preds)
    except:
        auc = float('nan')
    return acc, f1, dice, auc


def plot_metrics(metrics_df, save_dir, timestamp, fold, learning_rate, random_seed):
    epoch = len(metrics_df)
    if epoch % 10 == 0 or epoch == 1:
        best_val_loss_idx = metrics_df['val_loss'].idxmin()
        auc_at_min_loss = metrics_df.loc[best_val_loss_idx, 'val_auc']
        epoch_at_min_loss = int(metrics_df.loc[best_val_loss_idx, 'epoch'])

        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle(
            f"NeoNet - Fold: {fold} | LR: {learning_rate:.2e} | Seed: {random_seed}\nEpoch: {epoch} | Time: {timestamp}",
            fontsize=16, fontweight='bold', y=0.98
        )

        # Loss
        axes[0,0].plot(metrics_df["epoch"], metrics_df["train_loss"], label="Train Loss", color='blue', linewidth=2)
        axes[0,0].plot(metrics_df["epoch"], metrics_df["val_loss"], label="Val Loss", color='red', linewidth=2)
        if "load_balance_loss" in metrics_df.columns:
            axes[0,0].plot(metrics_df["epoch"], metrics_df["load_balance_loss"], label="Load Balance Loss (×100)", color='orange', linewidth=1, alpha=0.7)
        axes[0,0].scatter(epoch_at_min_loss, metrics_df.loc[best_val_loss_idx, 'val_loss'], color='red', s=100, zorder=5, marker='*')
        axes[0,0].annotate(f'Lowest: {metrics_df.loc[best_val_loss_idx, "val_loss"]:.4f}\n@Epoch {epoch_at_min_loss}',
            xy=(epoch_at_min_loss, metrics_df.loc[best_val_loss_idx, 'val_loss']),
            xytext=(10, 10), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
        )
        axes[0,0].set_title("Training & Validation Loss", fontweight='bold')
        axes[0,0].set_xlabel("Epoch")
        axes[0,0].set_ylabel("Loss")
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # AUC
        axes[0,1].plot(metrics_df["epoch"], metrics_df["val_auc"], marker="o", color='green', linewidth=2)
        axes[0,1].scatter(epoch_at_min_loss, auc_at_min_loss, color='red', s=120, zorder=5, marker='*')
        axes[0,1].annotate(f'AUC at min loss: {auc_at_min_loss:.4f}\n@Epoch {epoch_at_min_loss}',
            xy=(epoch_at_min_loss, auc_at_min_loss),
            xytext=(10, 10), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
        )
        axes[0,1].set_title("Validation AUC", fontweight='bold')
        axes[0,1].set_xlabel("Epoch")
        axes[0,1].set_ylabel("AUC")
        axes[0,1].grid(True, alpha=0.3)

        # F1
        axes[0,2].plot(metrics_df["epoch"], metrics_df["val_f1"], marker="s", color='purple', linewidth=2)
        axes[0,2].set_title("Validation F1 Score", fontweight='bold')
        axes[0,2].set_xlabel("Epoch")
        axes[0,2].set_ylabel("F1 Score")
        axes[0,2].grid(True, alpha=0.3)

        # Accuracy
        axes[1,0].plot(metrics_df["epoch"], metrics_df["val_acc"], marker="^", color='orange', linewidth=2)
        axes[1,0].set_title("Validation Accuracy", fontweight='bold')
        axes[1,0].set_xlabel("Epoch")
        axes[1,0].set_ylabel("Accuracy")
        axes[1,0].grid(True, alpha=0.3)

        # Dice
        axes[1,1].plot(metrics_df["epoch"], metrics_df["val_dice"], marker="d", color='brown', linewidth=2)
        axes[1,1].set_title("Validation Dice Score", fontweight='bold')
        axes[1,1].set_xlabel("Epoch")
        axes[1,1].set_ylabel("Dice Score")
        axes[1,1].grid(True, alpha=0.3)

        # All metrics
        axes[1,2].plot(metrics_df["epoch"], metrics_df["val_auc"], label="AUC", marker="o", linewidth=2)
        axes[1,2].plot(metrics_df["epoch"], metrics_df["val_f1"], label="F1", marker="s", linewidth=2)
        axes[1,2].plot(metrics_df["epoch"], metrics_df["val_acc"], label="Accuracy", marker="^", linewidth=2)
        axes[1,2].plot(metrics_df["epoch"], metrics_df["val_dice"], label="Dice", marker="d", linewidth=2)
        axes[1,2].set_title("All Validation Metrics", fontweight='bold')
        axes[1,2].set_xlabel("Epoch")
        axes[1,2].set_ylabel("Score")
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        plot_filename = f"neonet_metrics_epoch_{epoch:03d}_{timestamp}.png"
        plt.savefig(os.path.join(save_dir, plot_filename), dpi=300, bbox_inches='tight')
        plt.close()


def save_best_scores_to_csv(metrics_df, fold, learning_rate, random_seed, timestamp, best_scores_dir, args):
    epoch = len(metrics_df)
    if epoch % 10 == 0 or epoch == 1:
        best_val_loss_idx = metrics_df['val_loss'].idxmin()
        best_row = metrics_df.loc[best_val_loss_idx]

        csv_filename = f"neonet_best_scores_fold{fold}_lr{learning_rate:.0e}_seed{random_seed}_{timestamp}.csv"
        csv_path = os.path.join(best_scores_dir, csv_filename)

        best_data = {
            'timestamp': timestamp,
            'model': 'NeoNet',
            'fold': fold,
            'learning_rate': learning_rate,
            'random_seed': random_seed,
            'moh_efficiency': args.moh_efficiency,
            'num_shared_heads': args.num_shared_heads,
            'load_balance_weight': args.load_balance_weight,
            'batch_size': args.batch_size,
            'current_epoch': epoch,
            'best_val_loss_epoch': int(best_row['epoch']),
            'best_val_loss': best_row['val_loss'],
            'corresponding_train_loss': best_row['train_loss'],
            'corresponding_val_auc': best_row['val_auc'],
            'corresponding_val_f1': best_row['val_f1'],
            'corresponding_val_acc': best_row['val_acc'],
            'corresponding_val_dice': best_row['val_dice'],
        }

        if "load_balance_loss" in best_row:
            best_data['corresponding_load_balance_loss'] = best_row['load_balance_loss']

        best_df = pd.DataFrame([best_data])
        if not os.path.exists(csv_path):
            best_df.to_csv(csv_path, index=False)
        else:
            existing_df = pd.read_csv(csv_path)
            updated_df = pd.concat([existing_df, best_df], ignore_index=True)
            updated_df.to_csv(csv_path, index=False)


# =============================================================================
# MAIN TRAINING
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeoNet PNI classification with fixed SAF")

    parser.add_argument("--val_fold", type=int, default=1, help="Validation fold index")
    parser.add_argument("--preprocessed_dir", type=str, 
                       default="/home/rintern07/neonet/training/preprocessed_all_96x96x48_FIXED",
                       help="Directory containing preprocessed .npy files")
    parser.add_argument("--folds_csv", type=str,
                       default="/home/rintern07/final/data/new_168_fold.csv",
                       help="CSV file with fold information")
    parser.add_argument("--learning_rate", type=float, default=8e-5, help="Learning rate")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers")
    parser.add_argument("--use_mixed_precision", action="store_true", default=True, help="Use mixed precision")
    parser.add_argument("--moh_efficiency", type=float, default=0.75, help="MoH efficiency")
    parser.add_argument("--load_balance_weight", type=float, default=0.005, help="Load balance loss weight")
    parser.add_argument("--num_shared_heads", type=int, default=2, help="Number of shared heads")
    parser.add_argument("--postfix", type=str, default="fixed_saf", help="Postfix for save directory")
    parser.add_argument("--selected_channels", type=str, default=None, 
                       help="Comma-separated channel indices (e.g., '0,1,2')")

    args = parser.parse_args()

    # Parse selected channels
    if args.selected_channels:
        args.selected_channels = [int(ch) for ch in args.selected_channels.split(',')]
        in_channels = len(args.selected_channels)
        print(f"Using selected channels: {args.selected_channels}")
    else:
        in_channels = 3
        args.selected_channels = None

    # Set seeds
    VAL_FOLD = args.val_fold
    set_determinism(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    # Directory setup
    training_dir = "/home/rintern07/final/training"
    neonet_dir = os.path.join(training_dir, "neonet")
    os.makedirs(neonet_dir, exist_ok=True)

    base_name = f"fold{VAL_FOLD}_seed{args.random_seed}_lr{args.learning_rate:.0e}_moh{args.moh_efficiency:.0%}_lb{args.load_balance_weight:.0e}_sh{args.num_shared_heads}"
    if args.postfix:
        base_name = f"{base_name}_{args.postfix}"

    save_dir = os.path.join(neonet_dir, base_name)
    pictures_dir = os.path.join(save_dir, "pictures")
    best_scores_dir = os.path.join(neonet_dir, "best_scores")

    for dir_path in [save_dir, pictures_dir, best_scores_dir]:
        os.makedirs(dir_path, exist_ok=True)

    print("=" * 60)
    print(f"NEONET TRAINING CONFIGURATION (Fixed SAF)")
    print("=" * 60)
    print(f"Validation fold: {VAL_FOLD}")
    print(f"Random seed: {args.random_seed}")
    print(f"Batch size: {args.batch_size}")
    print(f"Mixed precision: {args.use_mixed_precision}")
    print(f"MoH efficiency: {args.moh_efficiency}")
    print(f"Load balance weight: {args.load_balance_weight}")
    print(f"Shared heads: {args.num_shared_heads}")
    print(f"Input channels: {in_channels}")
    print(f"Save directory: {save_dir}")
    print("=" * 60)

    # Load data
    train_paths, train_labels, train_pids, val_paths, val_labels, val_pids = load_preprocessed_data_with_folds(
        args.preprocessed_dir, args.folds_csv, args.val_fold
    )

    if len(train_paths) == 0 or len(val_paths) == 0:
        raise ValueError(f"No data found for validation fold {args.val_fold}")

    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")

    # Data augmentation
    train_transform = Compose([
        RandFlipd(keys=["input"], prob=0.5, spatial_axis=0),
        RandRotate90d(keys=["input"], prob=0.5, max_k=3),
        RandGaussianNoised(keys=["input"], prob=0.2, std=0.1),
        RandAffined(keys=["input"], prob=0.2, 
                   rotate_range=[0.1, 0.1, 0.1], translate_range=[5, 5, 2], 
                   scale_range=[0.05, 0.05, 0.05], mode="nearest"),
        RandShiftIntensityd(keys=["input"], prob=0.2, offsets=0.1),
    ])

    # Datasets
    train_dataset = PreprocessedDataset(train_paths, train_labels, train_pids, 
                                       transform=train_transform, 
                                       selected_channels=args.selected_channels)
    val_dataset = PreprocessedDataset(val_paths, val_labels, val_pids, 
                                     transform=None,
                                     selected_channels=args.selected_channels)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True
    )

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = NeoNet(
        in_channels=in_channels,
        moh_efficiency=args.moh_efficiency, 
        num_shared_heads=args.num_shared_heads
    ).to(device)

    print(f"Using NeoNet with FIXED SAF modules")
    print(f"  - SAF: Cross Attention → Spatial Attention")
    print(f"  - Residual connections: 3 levels")
    print(f"  - Channel attention + Gated fusion")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function
    def combined_loss_fn(logits, labels, load_balance_loss, weight=0.005):
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]).to(device))(logits, labels)
        total_loss = bce_loss + weight * load_balance_loss
        return total_loss, bce_loss, load_balance_loss

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    # Mixed precision
    try:
        from torch.amp import GradScaler
        scaler = GradScaler('cuda') if args.use_mixed_precision else None
    except ImportError:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if args.use_mixed_precision else None

    # Model loading
    best_model_path = os.path.join(save_dir, "best_neonet_model.pth")

    if os.path.exists(best_model_path):
        print(f"\n{'='*60}")
        print(f"Existing model found: {best_model_path}")
        
        try:
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
            
            if missing_keys:
                print(f"Missing keys: {len(missing_keys)} layers")
            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)} layers")
            
            print("Continuing training with partial model loading...")
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Starting from scratch")
        
        print(f"{'='*60}\n")

    # CSV setup
    csv_path = os.path.join(save_dir, "neonet_training_metrics.csv")
    pred_csv_path = os.path.join(save_dir, "neonet_val_predictions.csv")

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_f1", "val_dice", "val_auc", "load_balance_loss"])

    with open(pred_csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "pid", "gt", "pred_score", "pred_binary"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Training loop
    max_epochs = 200
    early_stop_patience = 30
    best_val_loss = float("inf")
    best_val_loss_auc = 0.0
    epochs_since_improvement = 0

    print("=" * 80)
    print("STARTING NEONET TRAINING WITH FIXED SAF")
    print("=" * 80)

    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss, train_bce_loss, train_lb_loss, train_batches = 0.0, 0.0, 0.0, 0

        for inputs, labels, _ in tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if args.use_mixed_precision and scaler is not None:
                try:
                    from torch.amp import autocast
                    with autocast('cuda'):
                        logits, load_balance_loss = model(inputs, return_load_balance_loss=True)
                        total_loss, bce_loss, lb_loss = combined_loss_fn(
                            logits, labels, load_balance_loss, args.load_balance_weight)
                except ImportError:
                    from torch.cuda.amp import autocast
                    with autocast():
                        logits, load_balance_loss = model(inputs, return_load_balance_loss=True)
                        total_loss, bce_loss, lb_loss = combined_loss_fn(
                            logits, labels, load_balance_loss, args.load_balance_weight)

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, load_balance_loss = model(inputs, return_load_balance_loss=True)
                total_loss, bce_loss, lb_loss = combined_loss_fn(
                    logits, labels, load_balance_loss, args.load_balance_weight)

                total_loss.backward()
                optimizer.step()

            train_loss += total_loss.item()
            train_bce_loss += bce_loss.item()
            train_lb_loss += lb_loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches
        avg_train_bce_loss = train_bce_loss / train_batches
        avg_train_lb_loss = train_lb_loss / train_batches

        # Validation
        model.eval()
        val_loss, val_bce_loss, val_lb_loss, val_batches = 0.0, 0.0, 0.0, 0
        val_preds, val_trues, val_pids_list = [], [], []

        with torch.no_grad():
            for inputs, labels, pids in tqdm(val_loader, desc=f"[Epoch {epoch+1}] Validation"):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                if args.use_mixed_precision and scaler is not None:
                    try:
                        from torch.amp import autocast
                        with autocast('cuda'):
                            logits, load_balance_loss = model(inputs, return_load_balance_loss=True)
                            total_loss, bce_loss, lb_loss = combined_loss_fn(
                                logits, labels, load_balance_loss, args.load_balance_weight)
                    except ImportError:
                        from torch.cuda.amp import autocast
                        with autocast():
                            logits, load_balance_loss = model(inputs, return_load_balance_loss=True)
                            total_loss, bce_loss, lb_loss = combined_loss_fn(
                                logits, labels, load_balance_loss, args.load_balance_weight)
                else:
                    logits, load_balance_loss = model(inputs, return_load_balance_loss=True)
                    total_loss, bce_loss, lb_loss = combined_loss_fn(
                        logits, labels, load_balance_loss, args.load_balance_weight)

                val_loss += total_loss.item()
                val_bce_loss += bce_loss.item()
                val_lb_loss += lb_loss.item()
                val_batches += 1

                preds = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(preds)
                val_trues.extend(labels.cpu().numpy())
                val_pids_list.extend(pids)

        avg_val_loss = val_loss / val_batches
        avg_val_bce_loss = val_bce_loss / val_batches
        avg_val_lb_loss = val_lb_loss / val_batches

        acc, f1, dice, auc = compute_metrics(np.array(val_preds), np.array(val_trues))

        print(f"Epoch {epoch+1:3d} | "
              f"Train: {avg_train_loss:.4f} (BCE: {avg_train_bce_loss:.4f}, LB: {avg_train_lb_loss:.4f}) | "
              f"Val: {avg_val_loss:.4f} (BCE: {avg_val_bce_loss:.4f}, LB: {avg_val_lb_loss:.4f}) | "
              f"AUC: {auc:.4f} F1: {f1:.4f} Acc: {acc:.4f}")

        # Save metrics
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1, round(avg_train_loss, 4), round(avg_val_loss, 4),
                round(acc, 4), round(f1, 4), round(dice, 4), round(auc, 4), round(avg_val_lb_loss, 4)
            ])

        # Save predictions
        with open(pred_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for pid, gt, pred_score in zip(val_pids_list, val_trues, val_preds):
                pred_binary = int(pred_score >= 0.5)
                writer.writerow([epoch + 1, pid, int(gt), round(pred_score, 4), pred_binary])

        # Model saving and early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_loss_auc = auc
            epochs_since_improvement = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"    ✅ New best model saved! Val Loss: {best_val_loss:.4f}, AUC: {best_val_loss_auc:.4f}")
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= early_stop_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # Plotting
        try:
            metrics_df = pd.read_csv(csv_path)
            plot_metrics(metrics_df, pictures_dir, timestamp, VAL_FOLD, args.learning_rate, args.random_seed)
            save_best_scores_to_csv(metrics_df, VAL_FOLD, args.learning_rate, args.random_seed, timestamp, best_scores_dir, args)
        except Exception as e:
            print(f"Error in plotting/saving: {e}")

    print("=" * 80)
    print("TRAINING COMPLETED")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Corresponding AUC: {best_val_loss_auc:.4f}")
    print(f"Model saved at: {best_model_path}")
    print("=" * 80)
