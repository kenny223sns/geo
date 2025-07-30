import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import json
from datetime import datetime

from models import UNetIso, UNetDir, LossFunction
from dataset import create_dataloaders, Geo2SigMapDataset
from channel_models import compute_3gpp_uma_path_loss

class Trainer:
    """
    Geo2SigMap 兩階段訓練器
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用設備: {self.device}")
        
        # 創建輸出目錄
        self.output_dir = config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(os.path.join(self.output_dir, "logs"))
        
        # 初始化模型
        self.setup_models()
        
        # 載入資料
        self.setup_data()
        
        # 訓練狀態
        self.current_stage = config.get("start_stage", 1)
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def setup_models(self):
        """
        初始化模型和優化器
        """
        # Stage 1 模型
        self.model_iso = UNetIso(self.config["base_features"]).to(self.device)
        self.optimizer_iso = optim.Adam(
            self.model_iso.parameters(), 
            lr=self.config["learning_rate"],
            weight_decay=self.config.get("weight_decay", 1e-5)
        )
        self.scheduler_iso = optim.lr_scheduler.StepLR(
            self.optimizer_iso, 
            step_size=self.config.get("scheduler_step", 50),
            gamma=self.config.get("scheduler_gamma", 0.5)
        )
        
        # Stage 2 模型
        self.model_dir = UNetDir(self.config["base_features"]).to(self.device)
        self.optimizer_dir = optim.Adam(
            self.model_dir.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config.get("weight_decay", 1e-5)
        )
        self.scheduler_dir = optim.lr_scheduler.StepLR(
            self.optimizer_dir,
            step_size=self.config.get("scheduler_step", 50),
            gamma=self.config.get("scheduler_gamma", 0.5)
        )
        
        print(f"Stage 1 模型參數: {sum(p.numel() for p in self.model_iso.parameters()):,}")
        print(f"Stage 2 模型參數: {sum(p.numel() for p in self.model_dir.parameters()):,}")
    
    def setup_data(self):
        """
        設置資料載入器
        """
        self.data_dir = self.config["data_dir"]
        self.batch_size = self.config["batch_size"]
        
        # Stage 1 資料
        self.train_loader_s1, self.val_loader_s1, self.test_loader_s1 = create_dataloaders(
            self.data_dir, self.batch_size, stage="stage1", 
            enable_augmentation=self.config.get("augmentation", True)
        )
        
        # Stage 2 資料
        self.train_loader_s2, self.val_loader_s2, self.test_loader_s2 = create_dataloaders(
            self.data_dir, self.batch_size, stage="stage2",
            enable_augmentation=self.config.get("augmentation", True)
        )
        
        print(f"Stage 1 - 訓練: {len(self.train_loader_s1)}, 驗證: {len(self.val_loader_s1)}")
        print(f"Stage 2 - 訓練: {len(self.train_loader_s2)}, 驗證: {len(self.val_loader_s2)}")
    
    def train_stage1(self, num_epochs):
        """
        Stage 1 訓練：[B, P_UMa] -> Piso
        """
        print(f"\n=== 開始 Stage 1 訓練 ({num_epochs} epochs) ===")
        
        self.model_iso.train()
        stage1_step = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            pbar = tqdm(self.train_loader_s1, desc=f"Stage1 Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(pbar):
                inputs = batch['input'].to(self.device)    # [B, P_UMa]
                targets = batch['target'].to(self.device)  # Piso
                
                # 前向傳播
                self.optimizer_iso.zero_grad()
                outputs = self.model_iso(inputs)
                
                # 計算損失
                loss = LossFunction.mse_loss(outputs, targets)
                
                # 反向傳播
                loss.backward()
                self.optimizer_iso.step()
                
                # 記錄
                epoch_loss += loss.item()
                stage1_step += 1
                
                # TensorBoard記錄
                if stage1_step % 100 == 0:
                    self.writer.add_scalar("Stage1/Train_Loss", loss.item(), stage1_step)
                    self.writer.add_scalar("Stage1/Learning_Rate", 
                                         self.optimizer_iso.param_groups[0]['lr'], stage1_step)
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # 驗證
            val_loss = self.validate_stage1()
            
            # 學習率調整
            self.scheduler_iso.step()
            
            # 記錄epoch結果
            avg_train_loss = epoch_loss / len(self.train_loader_s1)
            self.writer.add_scalar("Stage1/Epoch_Train_Loss", avg_train_loss, epoch)
            self.writer.add_scalar("Stage1/Epoch_Val_Loss", val_loss, epoch)
            
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # 儲存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("stage1_best.pth", stage=1, epoch=epoch, loss=val_loss)
            
            # 定期儲存
            if (epoch + 1) % 20 == 0:
                self.save_checkpoint(f"stage1_epoch_{epoch+1}.pth", stage=1, epoch=epoch, loss=val_loss)
        
        print("Stage 1 訓練完成!")
    
    def validate_stage1(self):
        """
        Stage 1 驗證
        """
        self.model_iso.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader_s1:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                outputs = self.model_iso(inputs)
                loss = LossFunction.mse_loss(outputs, targets)
                total_loss += loss.item()
        
        self.model_iso.train()
        return total_loss / len(self.val_loader_s1)
    
    def train_stage2(self, num_epochs, use_predicted_piso=True):
        """
        Stage 2 訓練：[B, Piso, S_sparse, mask] -> S
        
        參數:
            use_predicted_piso: 是否使用Stage1預測的Piso（推薦）
        """
        print(f"\n=== 開始 Stage 2 訓練 ({num_epochs} epochs) ===")
        
        # 載入Stage1最佳模型
        if use_predicted_piso:
            stage1_ckpt = os.path.join(self.output_dir, "checkpoints", "stage1_best.pth")
            if os.path.exists(stage1_ckpt):
                self.load_checkpoint(stage1_ckpt, stage=1)
                print("載入Stage1最佳權重")
            else:
                print("警告：未找到Stage1權重，使用隨機初始化的Piso")
        
        # 凍結Stage1模型
        self.model_iso.eval()
        for param in self.model_iso.parameters():
            param.requires_grad = False
        
        self.model_dir.train()
        stage2_step = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            pbar = tqdm(self.train_loader_s2, desc=f"Stage2 Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(pbar):
                inputs = batch['input'].to(self.device)    # [B, Piso, S_sparse, mask]
                targets = batch['target'].to(self.device)  # S
                
                # 如果使用預測的Piso，需要重新生成輸入
                if use_predicted_piso:
                    B = inputs[:, 0:1]  # 建築高度
                    S_sparse = inputs[:, 2:3]  # 稀疏RSSI
                    mask = inputs[:, 3:4]  # mask
                    
                    # 需要P_UMa來預測Piso（這裡簡化，實際需要從dataset取得）
                    # 暫時使用原本的Piso通道
                    original_piso = inputs[:, 1:2]
                    
                    # 使用Stage1模型預測Piso（可選）
                    # 這裡為了穩定訓練，先使用真實Piso
                    inputs_stage2 = torch.cat([B, original_piso, S_sparse, mask], dim=1)
                else:
                    inputs_stage2 = inputs
                
                # 前向傳播
                self.optimizer_dir.zero_grad()
                outputs = self.model_dir(inputs_stage2)
                
                # 計算損失
                mask_tensor = inputs_stage2[:, 3:4]  # mask通道
                sparse_target = inputs_stage2[:, 2:3]  # 稀疏RSSI
                
                # 組合損失：重建 + 一致性
                reconstruction_loss = LossFunction.mse_loss(outputs, targets)
                consistency_loss = LossFunction.consistency_loss(outputs, sparse_target, mask_tensor)
                
                loss = reconstruction_loss + 0.5 * consistency_loss
                
                # 反向傳播
                loss.backward()
                self.optimizer_dir.step()
                
                # 記錄
                epoch_loss += loss.item()
                stage2_step += 1
                
                # TensorBoard記錄
                if stage2_step % 100 == 0:
                    self.writer.add_scalar("Stage2/Train_Loss", loss.item(), stage2_step)
                    self.writer.add_scalar("Stage2/Reconstruction_Loss", reconstruction_loss.item(), stage2_step)
                    self.writer.add_scalar("Stage2/Consistency_Loss", consistency_loss.item(), stage2_step)
                    self.writer.add_scalar("Stage2/Learning_Rate", 
                                         self.optimizer_dir.param_groups[0]['lr'], stage2_step)
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # 驗證
            val_loss = self.validate_stage2(use_predicted_piso)
            
            # 學習率調整
            self.scheduler_dir.step()
            
            # 記錄epoch結果
            avg_train_loss = epoch_loss / len(self.train_loader_s2)
            self.writer.add_scalar("Stage2/Epoch_Train_Loss", avg_train_loss, epoch)
            self.writer.add_scalar("Stage2/Epoch_Val_Loss", val_loss, epoch)
            
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # 儲存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("stage2_best.pth", stage=2, epoch=epoch, loss=val_loss)
            
            # 定期儲存
            if (epoch + 1) % 20 == 0:
                self.save_checkpoint(f"stage2_epoch_{epoch+1}.pth", stage=2, epoch=epoch, loss=val_loss)
        
        print("Stage 2 訓練完成!")
    
    def validate_stage2(self, use_predicted_piso=True):
        """
        Stage 2 驗證
        """
        self.model_dir.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader_s2:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                outputs = self.model_dir(inputs)
                
                # 計算損失（簡化版）
                loss = LossFunction.mse_loss(outputs, targets)
                total_loss += loss.item()
        
        self.model_dir.train()
        return total_loss / len(self.val_loader_s2)
    
    def save_checkpoint(self, filename, stage, epoch, loss):
        """
        儲存檢查點
        """
        checkpoint_path = os.path.join(self.output_dir, "checkpoints", filename)
        
        if stage == 1:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model_iso.state_dict(),
                'optimizer_state_dict': self.optimizer_iso.state_dict(),
                'scheduler_state_dict': self.scheduler_iso.state_dict(),
                'loss': loss,
                'config': self.config
            }
        else:  # stage == 2
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model_dir.state_dict(),
                'optimizer_state_dict': self.optimizer_dir.state_dict(),
                'scheduler_state_dict': self.scheduler_dir.state_dict(),
                'loss': loss,
                'config': self.config
            }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"儲存檢查點: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path, stage):
        """
        載入檢查點
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if stage == 1:
            self.model_iso.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer_iso.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler_iso.load_state_dict(checkpoint['scheduler_state_dict'])
        else:  # stage == 2
            self.model_dir.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer_dir.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler_dir.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"載入檢查點: {checkpoint_path}")
        return checkpoint['epoch'], checkpoint['loss']
    
    def evaluate(self, stage, num_samples=100):
        """
        評估模型效能
        """
        print(f"\n=== 評估 Stage {stage} 模型 ===")
        
        if stage == 1:
            model = self.model_iso
            test_loader = self.test_loader_s1
        else:
            model = self.model_dir
            test_loader = self.test_loader_s2
        
        model.eval()
        
        # 評估指標
        mse_losses = []
        mae_losses = []
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= num_samples // self.batch_size:
                    break
                    
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                outputs = model(inputs)
                
                # 計算指標
                mse = LossFunction.mse_loss(outputs, targets).item()
                mae = LossFunction.mae_loss(outputs, targets).item()
                
                mse_losses.append(mse)
                mae_losses.append(mae)
        
        # 統計結果
        avg_mse = np.mean(mse_losses)
        avg_mae = np.mean(mae_losses)
        rmse = np.sqrt(avg_mse)
        
        print(f"Stage {stage} 評估結果:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {avg_mae:.4f}")
        print(f"  MSE:  {avg_mse:.4f}")
        
        return rmse, avg_mae, avg_mse

def main():
    parser = argparse.ArgumentParser(description="Geo2SigMap 訓練腳本")
    parser.add_argument("--config", type=str, default="config.json", help="配置檔案路徑")
    parser.add_argument("--stage", type=int, choices=[1, 2], help="訓練階段")
    parser.add_argument("--epochs", type=int, help="訓練epochs")
    parser.add_argument("--eval", action="store_true", help="僅評估模式")
    
    args = parser.parse_args()
    
    # 預設配置
    default_config = {
        "data_dir": "./geo2sigmap_dataset",
        "output_dir": "./outputs",
        "batch_size": 16,
        "learning_rate": 1e-3,
        "base_features": 64,
        "weight_decay": 1e-5,
        "scheduler_step": 50,
        "scheduler_gamma": 0.5,
        "augmentation": True,
        "stage1_epochs": 100,
        "stage2_epochs": 100
    }
    
    # 載入配置檔案
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            user_config = json.load(f)
        default_config.update(user_config)
    else:
        # 儲存預設配置
        with open(args.config, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"創建預設配置檔案: {args.config}")
    
    # 創建訓練器
    trainer = Trainer(default_config)
    
    if args.eval:
        # 評估模式
        if args.stage == 1 or args.stage is None:
            trainer.evaluate(stage=1)
        if args.stage == 2 or args.stage is None:
            trainer.evaluate(stage=2)
    else:
        # 訓練模式
        if args.stage == 1 or args.stage is None:
            epochs = args.epochs or default_config["stage1_epochs"]
            trainer.train_stage1(epochs)
        
        if args.stage == 2 or args.stage is None:
            epochs = args.epochs or default_config["stage2_epochs"]
            trainer.train_stage2(epochs)
    
    trainer.writer.close()
    print("訓練完成!")

if __name__ == "__main__":
    main()