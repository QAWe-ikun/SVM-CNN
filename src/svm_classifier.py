"""
SVM 分类器实现
使用 Sobel 特征 + HOG 特征进行 CIFAR-10 图像分类
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, List
import pickle
import json

import cv2
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from sobel_edge_detection import SobelEdgeDetector, extract_sobel_features


class SVMSobelClassifier:
    """基于 Sobel 特征的 SVM 图像分类器"""
    
    def __init__(self, kernel: str = 'rbf', C: float = 10.0, gamma: str = 'scale'):
        """
        初始化 SVM 分类器
        
        Args:
            kernel: 核函数类型 ('linear', 'rbf', 'poly', 'sigmoid')
            C: 正则化参数
            gamma: RBF 核的 gamma 参数
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.svm_model = None
        self.scaler = StandardScaler()
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
        self.training_time = 0.0
        self.inference_time = 0.0
    
    def extract_hog_features(self, image: np.ndarray, 
                             pixels_per_cell: int = 4,
                             cells_per_block: int = 2) -> np.ndarray:
        """
        提取 HOG 特征
        
        Args:
            image: 输入图像（灰度）
            pixels_per_cell: 每个 cell 的像素数
            cells_per_block: 每个 block 的 cell 数
            
        Returns:
            HOG 特征向量
        """
        # 使用 OpenCV 的 HOG 实现
        hog = cv2.HOGDescriptor((image.shape[1], image.shape[0]),
                                (16, 16),  # block size
                                (8, 8),    # block stride
                                (8, 8),    # cell size
                                9)         # number of bins
        
        # 调整图像大小以适配 HOG
        resized = cv2.resize(image, (32, 32))
        features = hog.compute(resized)
        
        return features.flatten() # type: ignore
    
    def extract_combined_features(self, image: np.ndarray) -> np.ndarray:
        """
        提取组合特征（Sobel + HOG）
        
        Args:
            image: 输入图像（RGB 格式，32x32x3）
            
        Returns:
            组合特征向量
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 提取 Sobel 特征
        detector = SobelEdgeDetector(ksize=3)
        gx, gy, magnitude, direction = detector.detect_edges(gray)
        
        # Sobel 特征：梯度幅值的统计特征
        mag_features = [
            np.mean(magnitude),
            np.std(magnitude),
            np.max(magnitude),
            np.min(magnitude),
            np.percentile(magnitude, 75),
            np.percentile(magnitude, 25)
        ]
        
        # 方向特征
        dir_features = [
            np.mean(direction),
            np.std(direction)
        ]
        
        # 分区域的梯度特征（将图像分为 4 个区域）
        h, w = magnitude.shape
        region_features = []
        for i in range(2):
            for j in range(2):
                region = magnitude[i*h//2:(i+1)*h//2, j*w//2:(j+1)*w//2]
                region_features.extend([np.mean(region), np.std(region)])
        
        # HOG 特征
        hog_features = self.extract_hog_features(gray)
        
        # 组合所有特征
        combined = np.concatenate([mag_features, dir_features, region_features, hog_features])
        
        return combined
    
    def prepare_data(self, images: np.ndarray, labels: np.ndarray,
                     max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备训练/测试数据
        
        Args:
            images: 图像数组 (N, 32, 32, 3)
            labels: 标签数组 (N,)
            max_samples: 最大样本数（用于快速测试）
            
        Returns:
            (features, labels)
        """
        if max_samples:
            indices = np.random.choice(len(images), max_samples, replace=False)
            images = images[indices]
            labels = labels[indices]
        
        features = []
        for i, image in enumerate(tqdm(images, desc="  提取特征", unit="样本", ncols=100)):
            feat = self.extract_combined_features(image)
            features.append(feat)
        
        return np.array(features), labels
    
    def train(self, train_images: np.ndarray, train_labels: np.ndarray,
              max_samples: int = 10000) -> Dict:
        """
        训练 SVM 模型
        
        Args:
            train_images: 训练图像 (N, 32, 32, 3)
            train_labels: 训练标签 (N,)
            max_samples: 最大训练样本数
            
        Returns:
            训练结果字典
        """
        print("=" * 50)
        print("开始训练 SVM 分类器...")
        print("=" * 50)
        
        # 准备训练数据
        print("\n提取训练集特征...")
        start_time = time.time()
        X_train, y_train = self.prepare_data(train_images, train_labels, max_samples)
        feature_time = time.time() - start_time
        print(f"特征提取时间：{feature_time:.2f} 秒")
        print(f"特征维度：{X_train.shape[1]}")
        
        # 特征标准化
        print("\n标准化特征...")
        X_train_scaled = self.scaler.fit_transform(X_train)

        # 训练 SVM
        print(f"\n训练 SVM 模型 (kernel={self.kernel}, C={self.C})...")
        start_time = time.time()

        # 使用线性核或 RBF 核
        # verbose 参数可以显示训练进度
        if self.kernel == 'linear':
            self.svm_model = svm.LinearSVC(C=self.C, max_iter=1000, random_state=42)
        else:
            self.svm_model = svm.SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, # type: ignore
                                     random_state=42, probability=True)

        print("开始训练 (sklearn 将显示训练进度)...")
        self.svm_model.fit(X_train_scaled, y_train)
        self.training_time = time.time() - start_time
        
        print(f"训练时间：{self.training_time:.2f} 秒")
        
        # 训练集准确率
        train_pred = self.svm_model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"训练集准确率：{train_acc:.4f}")
        
        results = {
            'train_accuracy': train_acc,
            'training_time': self.training_time,
            'feature_extraction_time': feature_time,
            'n_samples': len(X_train),
            'n_features': X_train.shape[1]
        }
        
        return results
    
    def predict(self, test_images: np.ndarray, 
                batch_size: int = 100) -> Tuple[np.ndarray, float]:
        """
        预测测试集
        
        Args:
            test_images: 测试图像 (N, 32, 32, 3)
            batch_size: 批处理大小
            
        Returns:
            (predictions, inference_time)
        """
        print("\n提取测试集特征...")
        start_time = time.time()
        
        features = []
        for image in tqdm(test_images, desc="  提取特征", unit="样本", ncols=100):
            feat = self.extract_combined_features(image)
            features.append(feat)
        
        X_test = np.array(features)
        feature_time = time.time() - start_time
        
        # 标准化
        X_test_scaled = self.scaler.transform(X_test)
        
        # 预测
        print("进行预测...")
        start_time = time.time()
        predictions = self.svm_model.predict(X_test_scaled) # type: ignore
        self.inference_time = time.time() - start_time + feature_time
        
        return predictions, self.inference_time
    
    def evaluate(self, test_images: np.ndarray, test_labels: np.ndarray,
                 max_samples: Optional[int] = None) -> Dict:
        """
        评估模型性能
        
        Args:
            test_images: 测试图像
            test_labels: 测试标签
            max_samples: 最大测试样本数
            
        Returns:
            评估结果字典
        """
        if max_samples:
            indices = np.random.choice(len(test_images), max_samples, replace=False)
            test_images = test_images[indices]
            test_labels = test_labels[indices]
        
        print("=" * 50)
        print("评估 SVM 分类器...")
        print("=" * 50)
        
        predictions, total_time = self.predict(test_images)
        
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
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues) # type: ignore
        plt.title('SVM + Sobel/HOG Confusion Matrix')
        plt.colorbar()
        
        # 添加标签
        tick_marks = np.arange(len(self.class_names))
        plt.xticks(tick_marks, self.class_names, rotation=45, ha='right')
        plt.yticks(tick_marks, self.class_names)
        
        # 添加数值
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
    
    def save_model(self, path: str) -> None:
        """保存模型"""
        model_data = {
            'svm_model': self.svm_model,
            'scaler': self.scaler,
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"模型已保存到：{path}")
    
    def load_model(self, path: str) -> None:
        """加载模型"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self.svm_model = model_data['svm_model']
        self.scaler = model_data['scaler']
        self.kernel = model_data['kernel']
        self.C = model_data['C']
        self.gamma = model_data['gamma']
        print(f"模型已加载：{path}")


def load_cifar10(data_dir: str = "data\\cifar-10-batches-py") -> Tuple:
    """
    加载 CIFAR-10 数据集
    
    注意：此函数需要 CIFAR-10 的 Python 格式数据
    可以从 https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz 下载
    """
    # 检查是否已有处理好的数据
    npy_path = os.path.join(data_dir, 'cifar10_processed.npz')
    
    if os.path.exists(npy_path):
        print("加载已处理的 CIFAR-10 数据...")
        data = np.load(npy_path)
        return data['train_images'], data['train_labels'], data['test_images'], data['test_labels']
    
    # 尝试从原始文件加载
    train_images = []
    train_labels = []
    
    # 加载 5 个训练批次
    for i in range(1, 6):
        batch_path = os.path.join(data_dir, f'data_batch_{i}')
        if not os.path.exists(batch_path):
            print(f"未找到训练批次：{batch_path}")
            print("请确保 CIFAR-10 数据已下载并解压到 data/ 目录")
            return None, None, None, None
        
        with open(batch_path, 'rb') as f:
            import pickle
            batch = pickle.load(f, encoding='latin1')
        
        images = batch['data'].reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
        labels = np.array(batch['labels'])
        
        train_images.append(images)
        train_labels.append(labels)
    
    train_images = np.concatenate(train_images)
    train_labels = np.concatenate(train_labels)
    
    # 加载测试批次
    test_path = os.path.join(data_dir, 'test_batch')
    with open(test_path, 'rb') as f:
        import pickle
        test_batch = pickle.load(f, encoding='latin1')

    test_images = test_batch['data'].reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = np.array(test_batch['labels'])
    
    # 保存处理后的数据
    np.savez(npy_path, 
             train_images=train_images, 
             train_labels=train_labels,
             test_images=test_images, 
             test_labels=test_labels)
    
    print(f"CIFAR-10 数据已处理并保存到：{npy_path}")
    
    return train_images, train_labels, test_images, test_labels


def train_and_evaluate_svm():
    """训练和评估 SVM 分类器的完整流程"""
    # 创建必要的目录
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # 加载数据
    print("加载 CIFAR-10 数据集...")
    train_images, train_labels, test_images, test_labels = load_cifar10()
    
    if train_images is None:
        print("\n使用 PyTorch 加载 CIFAR-10...")
        try:
            from torchvision import datasets
            train_dataset = datasets.CIFAR10(root='data', train=True, download=True)
            test_dataset = datasets.CIFAR10(root='data', train=False, download=True)

            # CIFAR-10 的 data 属性已经是 numpy 数组
            train_images = train_dataset.data.copy()
            train_labels = np.array(train_dataset.targets)
            test_images = test_dataset.data.copy()
            test_labels = np.array(test_dataset.targets)

            print(f"训练集：{len(train_images)} 样本")
            print(f"测试集：{len(test_images)} 样本")
        except Exception as e:
            print(f"加载数据失败：{e}")
            return
    
    # 创建分类器
    classifier = SVMSobelClassifier(kernel='rbf', C=10.0, gamma='scale')
    
    # 训练
    train_results = classifier.train(train_images, train_labels, max_samples=10000)
    
    # 评估
    eval_results = classifier.evaluate(test_images, test_labels, max_samples=None)
    
    # 绘制混淆矩阵
    classifier.plot_confusion_matrix(
        eval_results['true_labels'],
        eval_results['predictions'],
        save_path="results/svm_confusion_matrix.png"
    )
    
    # 保存模型
    classifier.save_model("models/svm_sobel_model.pkl")
    
    # 保存结果
    results = {
        'train_results': train_results,
        'eval_results': {
            'test_accuracy': eval_results['test_accuracy'],
            'inference_time': eval_results['inference_time'],
            'avg_inference_time_per_sample': eval_results['avg_inference_time_per_sample']
        }
    }
    
    with open("results/svm_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 50)
    print("SVM 训练和评估完成！")
    print("=" * 50)
    print(f"训练集准确率：{train_results['train_accuracy']:.4f}")
    print(f"测试集准确率：{eval_results['test_accuracy']:.4f}")
    print(f"训练时间：{train_results['training_time']:.2f} 秒")
    print(f"平均推理时间：{eval_results['avg_inference_time_per_sample']*1000:.2f} ms/样本")
    
    return results


if __name__ == "__main__":
    train_and_evaluate_svm()
