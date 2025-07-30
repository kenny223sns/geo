import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBlock(nn.Module):
    """
    U-Net 基本卷積塊：Conv2d + BatchNorm + ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DoubleConv(nn.Module):
    """
    雙卷積層（U-Net標準組件）
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class DownSample(nn.Module):
    """
    下採樣層：MaxPool + DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class UpSample(nn.Module):
    """
    上採樣層：ConvTranspose2d + DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 處理尺寸不匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # 跨層連接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    """
    通用 U-Net 模型
    參考 Geo2SigMap 論文的架構設計
    """
    def __init__(self, in_channels, out_channels, base_features=64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 編碼器（下採樣路徑）
        self.inc = DoubleConv(in_channels, base_features)
        self.down1 = DownSample(base_features, base_features * 2)
        self.down2 = DownSample(base_features * 2, base_features * 4)
        self.down3 = DownSample(base_features * 4, base_features * 8)
        
        # 瓶頸層
        self.bottleneck = DownSample(base_features * 8, base_features * 16)
        
        # 解碼器（上採樣路徑）
        self.up1 = UpSample(base_features * 16, base_features * 8)
        self.up2 = UpSample(base_features * 8, base_features * 4)
        self.up3 = UpSample(base_features * 4, base_features * 2)
        self.up4 = UpSample(base_features * 2, base_features)
        
        # 輸出層
        self.outc = nn.Conv2d(base_features, out_channels, kernel_size=1)
    
    def forward(self, x):
        # 編碼器
        x1 = self.inc(x)      # 64
        x2 = self.down1(x1)   # 128
        x3 = self.down2(x2)   # 256  
        x4 = self.down3(x3)   # 512
        x5 = self.bottleneck(x4)  # 1024
        
        # 解碼器（帶跨層連接）
        x = self.up1(x5, x4)  # 512
        x = self.up2(x, x3)   # 256
        x = self.up3(x, x2)   # 128
        x = self.up4(x, x1)   # 64
        
        # 輸出
        output = self.outc(x)  # out_channels
        return output

class UNetIso(nn.Module):
    """
    U-Net-Iso：Stage 1 模型
    輸入：[B, P_UMa] (2通道)
    輸出：Piso (1通道)
    """
    def __init__(self, base_features=64):
        super(UNetIso, self).__init__()
        self.unet = UNet(in_channels=2, out_channels=1, base_features=base_features)
    
    def forward(self, x):
        """
        x: [batch, 2, H, W] - [B, P_UMa]
        return: [batch, 1, H, W] - Piso
        """
        return self.unet(x)

class UNetDir(nn.Module):
    """
    U-Net-Dir：Stage 2 模型  
    輸入：[B, Piso, S_sparse, mask] (4通道)
    輸出：S (1通道)
    """
    def __init__(self, base_features=64):
        super(UNetDir, self).__init__()
        self.unet = UNet(in_channels=4, out_channels=1, base_features=base_features)
    
    def forward(self, x):
        """
        x: [batch, 4, H, W] - [B, Piso, S_sparse, mask]
        return: [batch, 1, H, W] - S
        """
        return self.unet(x)

class Geo2SigMapModel(nn.Module):
    """
    完整的 Geo2SigMap 模型（兩階段串接）
    用於推論時的端到端預測
    """
    def __init__(self, base_features=64):
        super(Geo2SigMapModel, self).__init__()
        self.unet_iso = UNetIso(base_features)
        self.unet_dir = UNetDir(base_features)
    
    def forward(self, B, P_uma, S_sparse, mask):
        """
        完整前向傳播
        
        參數:
            B: [batch, 1, H, W] - 建築高度圖
            P_uma: [batch, 1, H, W] - UMa路徑增益
            S_sparse: [batch, 1, H, W] - 稀疏RSSI
            mask: [batch, 1, H, W] - 0/1 mask
        
        返回:
            Piso_pred: Stage 1輸出
            S_pred: Stage 2最終輸出
        """
        # Stage 1: [B, P_UMa] -> Piso
        stage1_input = torch.cat([B, P_uma], dim=1)  # [batch, 2, H, W]
        Piso_pred = self.unet_iso(stage1_input)      # [batch, 1, H, W]
        
        # Stage 2: [B, Piso, S_sparse, mask] -> S
        stage2_input = torch.cat([B, Piso_pred, S_sparse, mask], dim=1)  # [batch, 4, H, W]
        S_pred = self.unet_dir(stage2_input)         # [batch, 1, H, W]
        
        return Piso_pred, S_pred
    
    def load_stage_weights(self, stage1_path, stage2_path):
        """
        載入兩階段的預訓練權重
        """
        # 載入 Stage 1
        stage1_state = torch.load(stage1_path, map_location='cpu')
        self.unet_iso.load_state_dict(stage1_state)
        print(f"載入 Stage 1 權重: {stage1_path}")
        
        # 載入 Stage 2
        stage2_state = torch.load(stage2_path, map_location='cpu')
        self.unet_dir.load_state_dict(stage2_state)
        print(f"載入 Stage 2 權重: {stage2_path}")

def count_parameters(model):
    """
    計算模型參數數量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"總參數數量: {total_params:,}")
    print(f"可訓練參數: {trainable_params:,}")
    return total_params, trainable_params

def test_models():
    """
    測試模型架構
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    
    # 測試輸入（batch_size=2, 128x128圖像）
    batch_size = 2
    H, W = 128, 128
    
    # Stage 1 測試
    print("\n=== 測試 U-Net-Iso (Stage 1) ===")
    model_iso = UNetIso().to(device)
    count_parameters(model_iso)
    
    # 輸入：[B, P_UMa]
    input_stage1 = torch.randn(batch_size, 2, H, W).to(device)
    with torch.no_grad():
        output_stage1 = model_iso(input_stage1)
    
    print(f"輸入形狀: {input_stage1.shape}")
    print(f"輸出形狀: {output_stage1.shape}")
    
    # Stage 2 測試
    print("\n=== 測試 U-Net-Dir (Stage 2) ===")
    model_dir = UNetDir().to(device)
    count_parameters(model_dir)
    
    # 輸入：[B, Piso, S_sparse, mask]
    input_stage2 = torch.randn(batch_size, 4, H, W).to(device)
    with torch.no_grad():
        output_stage2 = model_dir(input_stage2)
    
    print(f"輸入形狀: {input_stage2.shape}")
    print(f"輸出形狀: {output_stage2.shape}")
    
    # 完整模型測試
    print("\n=== 測試完整 Geo2SigMap 模型 ===")
    full_model = Geo2SigMapModel().to(device)
    count_parameters(full_model)
    
    # 準備輸入
    B = torch.randn(batch_size, 1, H, W).to(device)
    P_uma = torch.randn(batch_size, 1, H, W).to(device)
    S_sparse = torch.randn(batch_size, 1, H, W).to(device)
    mask = torch.randint(0, 2, (batch_size, 1, H, W)).float().to(device)
    
    with torch.no_grad():
        Piso_pred, S_pred = full_model(B, P_uma, S_sparse, mask)
    
    print(f"建築高度圖 B: {B.shape}")
    print(f"UMa路徑增益 P_uma: {P_uma.shape}")
    print(f"稀疏RSSI S_sparse: {S_sparse.shape}")
    print(f"採樣mask: {mask.shape}")
    print(f"Stage 1輸出 Piso: {Piso_pred.shape}")
    print(f"Stage 2輸出 S: {S_pred.shape}")
    
    print("\n模型測試完成！")

class LossFunction:
    """
    損失函數類
    """
    @staticmethod
    def mse_loss(pred, target, mask=None):
        """
        均方誤差損失
        """
        if mask is not None:
            # 只計算mask區域的損失
            loss = F.mse_loss(pred * mask, target * mask, reduction='none')
            return loss.sum() / (mask.sum() + 1e-8)
        else:
            return F.mse_loss(pred, target)
    
    @staticmethod
    def mae_loss(pred, target, mask=None):
        """
        平均絕對誤差損失
        """
        if mask is not None:
            loss = F.l1_loss(pred * mask, target * mask, reduction='none')
            return loss.sum() / (mask.sum() + 1e-8)
        else:
            return F.l1_loss(pred, target)
    
    @staticmethod
    def combined_loss(pred, target, mask=None, mse_weight=1.0, mae_weight=0.1):
        """
        組合損失：MSE + MAE
        """
        mse = LossFunction.mse_loss(pred, target, mask)
        mae = LossFunction.mae_loss(pred, target, mask)
        return mse_weight * mse + mae_weight * mae
    
    @staticmethod
    def consistency_loss(pred, sparse_target, mask):
        """
        觀測點一致性損失（Stage 2專用）
        確保預測在已觀測點與稀疏輸入一致
        """
        return F.mse_loss(pred * mask, sparse_target * mask)

if __name__ == "__main__":
    test_models()