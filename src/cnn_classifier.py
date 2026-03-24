"""
CNN 分类器实现
VGG 风格网络用于 CIFAR-10 图像分类
"""

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
import pickle
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class VGGBlock(nn.Module):
    """VGG 风格的卷积块"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 num_convs: int = 2, dropout: float = 0.0):
        """
        初始化 VGG 块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            num_convs: 卷积层数量
            dropout: Dropout 比率
        """
        super(VGGBlock, self).__init__()
        
        layers = []
        for i in range(num_convs):
            in_ch = in_channels if i == 0 else out_channels
            layers.extend([
                nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VGGForCIFAR10(nn.Module):
    """
    适用于 CIFAR-10 的 VGG 风格网络
    
    网络结构参考 VGG-11/13，但针对 32x32 输入进行了调整
    """
    
    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        """
        初始化 VGG 网络
        
        Args:
            num_classes: 分类类别数
            dropout: 全连接层的 Dropout 比率
        """
        super(VGGForCIFAR10, self).__init__()
        
        # VGG 风格特征提取器
        # 输入：32x32x3
        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            VGGBlock(3, 64, num_convs=2, dropout=0.1),
            # Block 2: 16x16 -> 8x8
            VGGBlock(64, 128, num_convs=2, dropout=0.2),
            # Block 3: 8x8 -> 4x4
            VGGBlock(128, 256, num_convs=3, dropout=0.3),
            # Block 4: 4x4 -> 2x2
            VGGBlock(256, 512, num_convs=2, dropout=0.3),
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class CNNClassifier:
    """CNN 分类器（VGG 风格）"""

    def __init__(self, device: Optional[str] = None,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4,
                 early_stopping: bool = True,
                 patience: int = 2,
                 min_delta: float = 0.001):
        """
        初始化 CNN 分类器

        Args:
            device: 计算设备 ('cuda', 'cpu', 或 None 自动选择)
            learning_rate: 学习率
            weight_decay: 权重衰减（L2 正则化）
            early_stopping: 是否启用早停
            patience: 早停耐心值（验证集准确率多少轮未提升则停止）
            min_delta: 最小改进量（准确率提升小于此值视为未改进）
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = VGGForCIFAR10().to(self.device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )

        self.scaler = GradScaler()  # 混合精度训练

        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']

        self.training_time = 0.0
        self.inference_time = 0.0
        self.train_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        # 早停参数
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.early_stop_epoch = None
    
    def get_transforms(self, augment: bool = True) -> Tuple:
        """
        获取数据变换
        
        Args:
            augment: 是否使用数据增强
            
        Returns:
            (train_transform, test_transform)
        """
        # 基础变换：归一化
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 均值
            std=[0.2470, 0.2435, 0.2616]    # CIFAR-10 标准差
        )
        
        if augment:
            # 训练集变换（带数据增强）
            train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                normalize
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                normalize
            ])
        
        # 测试集变换（无增强）
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            normalize
        ])
        
        return train_transform, test_transform
    
    def prepare_data(self, train_images: np.ndarray, train_labels: np.ndarray,
                     test_images: np.ndarray, test_labels: np.ndarray,
                     batch_size: int = 128,
                     augment: bool = True) -> Tuple:
        """
        准备数据加载器
        
        Args:
            train_images: 训练图像
            train_labels: 训练标签
            test_images: 测试图像
            test_labels: 测试标签
            batch_size: 批大小
            augment: 是否使用数据增强
            
        Returns:
            (train_loader, test_loader)
        """
        train_transform, test_transform = self.get_transforms(augment)
        
        # 创建数据集
        class CSVDataset(torch.utils.data.Dataset):
            def __init__(self, images, labels, transform=None):
                self.images = images
                self.labels = labels
                self.transform = transform
            
            def __len__(self):
                return len(self.images)
            
            def __getitem__(self, idx):
                img = self.images[idx]
                label = self.labels[idx]
                if self.transform:
                    img = self.transform(img)
                return img, label
        
        train_dataset = CSVDataset(train_images, train_labels, train_transform)
        test_dataset = CSVDataset(test_images, test_labels, test_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                  shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                 shuffle=False, num_workers=0, pin_memory=True)
        
        return train_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 混合精度训练
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                             'acc': f'{100.*correct/total:.2f}%'})
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validating', leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_images: np.ndarray, train_labels: np.ndarray,
              test_images: np.ndarray, test_labels: np.ndarray,
              epochs: int = 50, batch_size: int = 128,
              augment: bool = True) -> Dict:
        """
        训练 CNN 模型

        Args:
            train_images: 训练图像
            train_labels: 训练标签
            test_images: 测试图像（用作验证）
            test_labels: 测试标签
            epochs: 训练轮数
            batch_size: 批大小
            augment: 是否使用数据增强

        Returns:
            训练结果字典
        """
        print("=" * 50)
        print("开始训练 CNN (VGG) 分类器...")
        print("=" * 50)
        print(f"设备：{self.device}")
        print(f"训练轮数：{epochs}")
        print(f"批大小：{batch_size}")
        if self.early_stopping:
            print(f"早停：启用 (patience={self.patience}, min_delta={self.min_delta*100:.2f}%)")
        else:
            print(f"早停：禁用")

        # 准备数据
        print("\n准备数据...")
        train_loader, val_loader = self.prepare_data(
            train_images, train_labels,
            test_images, test_labels,
            batch_size=batch_size,
            augment=augment
        )

        # 训练循环
        print("\n开始训练...")
        start_time = time.time()

        best_acc = 0.0
        best_model_state = None
        
        # 早停参数重置
        if self.early_stopping:
            self.best_val_acc = 0.0
            self.patience_counter = 0
            self.early_stop_epoch = None

        for epoch in range(epochs):
            epoch_start = time.time()

            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)

            # 验证
            val_loss, val_acc = self.validate(val_loader)

            # 更新学习率
            self.scheduler.step()

            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)

            epoch_time = time.time() - epoch_start

            # 早停检查
            early_stop_msg = ""
            if self.early_stopping:
                # 检查是否有改进（考虑 min_delta）
                if val_acc > self.best_val_acc + self.min_delta:
                    self.best_val_acc = val_acc
                    best_model_state = self.model.state_dict().copy()
                    self.patience_counter = 0
                    early_stop_msg = " ✓ (最佳)"
                else:
                    self.patience_counter += 1
                    early_stop_msg = f" (耐心值：{self.patience_counter}/{self.patience})"
                    
                    # 触发早停
                    if self.patience_counter >= self.patience:
                        print(f"Epoch [{epoch+1}/{epochs}] - "
                              f"Val Acc: {val_acc:.2f}%, 早停触发！")
                        self.early_stop_epoch = epoch + 1
                        break

            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                  f"Time: {epoch_time:.1f}s{early_stop_msg}")

        self.training_time = time.time() - start_time

        # 恢复最佳模型
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        actual_epochs = self.early_stop_epoch if self.early_stop_epoch else epochs
        print(f"\n训练完成！")
        print(f"总训练时间：{self.training_time:.2f} 秒")
        print(f"实际训练轮数：{actual_epochs}/{epochs}")
        print(f"最佳验证准确率：{self.best_val_acc:.2f}%")

        results = {
            'best_val_accuracy': self.best_val_acc,
            'training_time': self.training_time,
            'actual_epochs': actual_epochs,
            'early_stopped': self.early_stop_epoch is not None,
            'final_train_loss': self.train_history['train_loss'][-1],
            'final_train_acc': self.train_history['train_acc'][-1],
            'final_val_loss': self.train_history['val_loss'][-1],
            'final_val_acc': self.train_history['val_acc'][-1]
        }

        return results
    
    def predict(self, test_images: np.ndarray, test_labels: np.ndarray,
                batch_size: int = 128) -> Tuple[np.ndarray, float]:
        """
        预测测试集
        
        Args:
            test_images: 测试图像
            test_labels: 测试标签
            batch_size: 批大小
            
        Returns:
            (predictions, inference_time)
        """
        self.model.eval()
        
        # 创建数据加载器
        _, test_transform = self.get_transforms(augment=False)
        
        class CSVDataset(torch.utils.data.Dataset):
            def __init__(self, images, labels, transform=None):
                self.images = images
                self.labels = labels
                self.transform = transform
            
            def __len__(self):
                return len(self.images)
            
            def __getitem__(self, idx):
                img = self.images[idx]
                label = self.labels[idx]
                if self.transform:
                    img = self.transform(img)
                return img, label
        
        test_dataset = CSVDataset(test_images, test_labels, test_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                 shuffle=False, num_workers=0)
        
        all_preds = []
        all_labels = []
        
        print("\n进行预测...")
        start_time = time.time()
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Predicting'):
                images = images.to(self.device)
                
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        self.inference_time = time.time() - start_time
        
        return np.array(all_preds), self.inference_time
    
    def evaluate(self, test_images: np.ndarray, test_labels: np.ndarray,
                 batch_size: int = 128) -> Dict:
        """
        评估模型性能
        
        Args:
            test_images: 测试图像
            test_labels: 测试标签
            batch_size: 批大小
            
        Returns:
            评估结果字典
        """
        print("=" * 50)
        print("评估 CNN 分类器...")
        print("=" * 50)
        
        predictions, total_time = self.predict(test_images, test_labels, batch_size)
        
        accuracy = accuracy_score(test_labels, predictions)
        
        print(f"\n测试集准确率：{accuracy:.4f}")
        print(f"总推理时间：{total_time:.2f} 秒")
        print(f"平均每样本推理时间：{total_time/len(test_images)*1000:.2f} ms")
        
        # 分类报告
        print("\n分类报告:")
        print(classification_report(test_labels, predictions, 
                                    target_names=self.class_names))
        
        results = {
            'test_accuracy': accuracy,
            'inference_time': total_time,
            'avg_inference_time_per_sample': total_time / len(test_images),
            'predictions': predictions,
            'true_labels': test_labels
        }
        
        return results
    
    def plot_confusion_matrix(self, true_labels: np.ndarray,
                              predictions: np.ndarray,
                              save_path: Optional[str] = None) -> None:
        """绘制混淆矩阵"""
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens) # type: ignore
        plt.title('CNN (VGG) Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(len(self.class_names))
        plt.xticks(tick_marks, self.class_names, rotation=45, ha='right')
        plt.yticks(tick_marks, self.class_names)
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"混淆矩阵已保存到：{save_path}")
        
        plt.show()
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 损失曲线
        axes[0].plot(self.train_history['train_loss'], label='Train Loss')
        axes[0].plot(self.train_history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 准确率曲线
        axes[1].plot(self.train_history['train_acc'], label='Train Acc')
        axes[1].plot(self.train_history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"训练历史已保存到：{save_path}")
        
        plt.show()
    
    def save_model(self, path: str) -> None:
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.train_history,
            'class_names': self.class_names
        }, path)
        print(f"模型已保存到：{path}")
    
    def load_model(self, path: str) -> None:
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_history = checkpoint['training_history']
        print(f"模型已加载：{path}")


def load_cifar10_torch(data_dir: str = "data") -> Tuple:
    """使用 torchvision 加载 CIFAR-10"""
    print("从 torchvision 加载 CIFAR-10 数据集...")

    # 简单的变换（不归一化，以便保存为 numpy）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.permute(1, 2, 0))  # CHW -> HWC
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                      download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                     download=True, transform=transform)

    # 转换为 numpy 数组
    # CIFAR-10 的 data 属性已经是 numpy 数组 (N, 32, 32, 3) 或 (N, 32, 32)
    if isinstance(train_dataset.data, list):
        train_images = np.array([img.numpy() if hasattr(img, 'numpy') else np.array(img) 
                                  for img in train_dataset.data])
    elif isinstance(train_dataset.data, np.ndarray):
        train_images = train_dataset.data.copy()
    else:
        train_images = np.array(train_dataset.data)
    
    # 确保是 4D 数组
    if len(train_images.shape) == 3:
        train_images = np.expand_dims(train_images, axis=-1)
        train_images = np.repeat(train_images, 3, axis=-1)

    train_labels = np.array(train_dataset.targets)

    # 处理测试集
    if isinstance(test_dataset.data, list):
        test_images = np.array([img.numpy() if hasattr(img, 'numpy') else np.array(img) 
                                 for img in test_dataset.data])
    elif isinstance(test_dataset.data, np.ndarray):
        test_images = test_dataset.data.copy()
    else:
        test_images = np.array(test_dataset.data)
    
    if len(test_images.shape) == 3:
        test_images = np.expand_dims(test_images, axis=-1)
        test_images = np.repeat(test_images, 3, axis=-1)

    test_labels = np.array(test_dataset.targets)

    # 如果数据范围是 0-1，转换到 0-255
    if train_images.max() <= 1.0:
        train_images = (train_images * 255).astype(np.uint8)
        test_images = (test_images * 255).astype(np.uint8)

    print(f"训练集：{len(train_images)} 样本")
    print(f"测试集：{len(test_images)} 样本")

    return train_images, train_labels, test_images, test_labels


def train_and_evaluate_cnn():
    """训练和评估 CNN 分类器的完整流程"""
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 加载数据
    train_images, train_labels, test_images, test_labels = load_cifar10_torch()

    # 创建分类器（启用早停）
    classifier = CNNClassifier(
        learning_rate=0.001,
        weight_decay=1e-4,
        early_stopping=True,   # 启用早停
        patience=10,           # 10 轮未改进则停止
        min_delta=0.001        # 最小改进 0.1%
    )

    # 训练
    train_results = classifier.train(
        train_images, train_labels,
        test_images, test_labels,
        epochs=50,
        batch_size=128,
        augment=True
    )
    
    # 绘制训练历史
    classifier.plot_training_history(save_path="results/cnn_training_history.png")
    
    # 评估
    eval_results = classifier.evaluate(test_images, test_labels, batch_size=128)
    
    # 绘制混淆矩阵
    classifier.plot_confusion_matrix(
        eval_results['true_labels'],
        eval_results['predictions'],
        save_path="results/cnn_confusion_matrix.png"
    )
    
    # 保存模型
    classifier.save_model("models/cnn_vgg_model.pth")
    
    # 保存结果
    results = {
        'train_results': train_results,
        'eval_results': {
            'test_accuracy': eval_results['test_accuracy'],
            'inference_time': eval_results['inference_time'],
            'avg_inference_time_per_sample': eval_results['avg_inference_time_per_sample']
        }
    }
    
    with open("results/cnn_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 50)
    print("CNN 训练和评估完成！")
    print("=" * 50)
    print(f"最佳验证准确率：{train_results['best_val_accuracy']:.2f}%")
    print(f"测试集准确率：{eval_results['test_accuracy']:.4f}")
    print(f"训练时间：{train_results['training_time']:.2f} 秒")
    print(f"平均推理时间：{eval_results['avg_inference_time_per_sample']*1000:.2f} ms/样本")
    
    return results


if __name__ == "__main__":
    train_and_evaluate_cnn()
