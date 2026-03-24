"""
Sobel 边缘检测算法实现
使用 OpenCV 实现 Sobel 算子进行边缘检测
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class SobelEdgeDetector:
    """Sobel 边缘检测器类"""
    
    def __init__(self, ksize: int = 3):
        """
        初始化 Sobel 边缘检测器
        
        Args:
            ksize: Sobel 算子核大小，必须为奇数 (1, 3, 5, 7...)
        """
        self.ksize = ksize
    
    def compute_sobel_x(self, image: np.ndarray) -> np.ndarray:
        """
        计算 X 方向的 Sobel 梯度
        
        Args:
            image: 输入图像（灰度图）
            
        Returns:
            X 方向的梯度图像
        """
        return cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=self.ksize)
    
    def compute_sobel_y(self, image: np.ndarray) -> np.ndarray:
        """
        计算 Y 方向的 Sobel 梯度
        
        Args:
            image: 输入图像（灰度图）
            
        Returns:
            Y 方向的梯度图像
        """
        return cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=self.ksize)
    
    def compute_gradient_magnitude(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        """
        计算梯度幅值
        
        Args:
            gx: X 方向梯度
            gy: Y 方向梯度
            
        Returns:
            梯度幅值图像
        """
        return cv2.magnitude(gx, gy)
    
    def compute_gradient_direction(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        """
        计算梯度方向
        
        Args:
            gx: X 方向梯度
            gy: Y 方向梯度
            
        Returns:
            梯度方向图像（弧度）
        """
        return cv2.phase(gx, gy, angleInDegrees=False)
    
    def detect_edges(self, image: np.ndarray, 
                     threshold: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        执行完整的 Sobel 边缘检测
        
        Args:
            image: 输入图像（灰度图）
            threshold: 二值化阈值 (min, max)，用于 Canny 式边缘检测
            
        Returns:
            tuple: (gx, gy, magnitude, direction)
        """
        gx = self.compute_sobel_x(image)
        gy = self.compute_sobel_y(image)
        magnitude = self.compute_gradient_magnitude(gx, gy)
        direction = self.compute_gradient_direction(gx, gy)
        
        return gx, gy, magnitude, direction
    
    def normalize_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """
        归一化梯度图像到 0-255 范围
        
        Args:
            gradient: 梯度图像
            
        Returns:
            归一化后的梯度图像
        """
        normalized = cv2.convertScaleAbs(gradient)
        return normalized
    
    def apply_non_max_suppression(self, magnitude: np.ndarray, 
                                   direction: np.ndarray) -> np.ndarray:
        """
        应用非极大值抑制细化边缘
        
        Args:
            magnitude: 梯度幅值
            direction: 梯度方向
            
        Returns:
            细化后的边缘图像
        """
        M, N = magnitude.shape
        result = np.zeros((M, N), dtype=np.float64)
        
        # 将梯度方向转换为角度（0-180 度）
        angle = direction * 180. / np.pi
        
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                # 确定梯度方向所属的区间
                q = 255.0
                r = 255.0
                
                # 角度 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]
                # 角度 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]
                # 角度 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]
                # 角度 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]
                
                # 非极大值抑制
                if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                    result[i, j] = magnitude[i, j]
                else:
                    result[i, j] = 0
        
        return result
    
    def apply_hysteresis_threshold(self, magnitude: np.ndarray,
                                    low_threshold: float,
                                    high_threshold: float) -> np.ndarray:
        """
        应用滞后阈值处理
        
        Args:
            magnitude: 梯度幅值
            low_threshold: 低阈值
            high_threshold: 高阈值
            
        Returns:
            二值化边缘图像
        """
        strong_edge = 255
        weak_edge = 75
        
        M, N = magnitude.shape
        result = np.zeros((M, N), dtype=np.float64)
        
        # 确定强边缘和弱边缘
        strong_i, strong_j = np.where(magnitude >= high_threshold)
        weak_i, weak_j = np.where((magnitude >= low_threshold) & (magnitude < high_threshold))
        
        result[strong_i, strong_j] = strong_edge
        result[weak_i, weak_j] = weak_edge
        
        # 连接弱边缘到强边缘
        result = self._connect_weak_edges(result)
        
        return result
    
    def _connect_weak_edges(self, image: np.ndarray) -> np.ndarray:
        """连接弱边缘到强边缘"""
        strong_edge = 255
        weak_edge = 75
        M, N = image.shape
        
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if image[i, j] == weak_edge:
                    # 检查 8 邻域是否有强边缘
                    neighborhood = image[i-1:i+2, j-1:j+2]
                    if np.any(neighborhood == strong_edge):
                        image[i, j] = strong_edge
                    else:
                        image[i, j] = 0
        
        return image


def visualize_sobel_results(image: np.ndarray, 
                            gx: np.ndarray, 
                            gy: np.ndarray, 
                            magnitude: np.ndarray,
                            direction: np.ndarray,
                            save_path: Optional[str] = None) -> None:
    """
    可视化 Sobel 检测结果
    
    Args:
        image: 原始图像
        gx: X 方向梯度
        gy: Y 方向梯度
        magnitude: 梯度幅值
        direction: 梯度方向
        save_path: 保存路径（可选）
    """
    detector = SobelEdgeDetector()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原始图像
    if len(image.shape) == 2:
        axes[0, 0].imshow(image, cmap='gray')
    else:
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # X 方向梯度
    axes[0, 1].imshow(detector.normalize_gradient(gx), cmap='gray')
    axes[0, 1].set_title(f'Sobel X Gradient (ksize={detector.ksize})')
    axes[0, 1].axis('off')
    
    # Y 方向梯度
    axes[0, 2].imshow(detector.normalize_gradient(gy), cmap='gray')
    axes[0, 2].set_title(f'Sobel Y Gradient (ksize={detector.ksize})')
    axes[0, 2].axis('off')
    
    # 梯度幅值
    axes[1, 0].imshow(detector.normalize_gradient(magnitude), cmap='gray')
    axes[1, 0].set_title('Gradient Magnitude')
    axes[1, 0].axis('off')
    
    # 梯度方向
    axes[1, 1].imshow(direction, cmap='hsv')
    axes[1, 1].set_title('Gradient Direction')
    axes[1, 1].axis('off')
    
    # 边缘检测结果（非极大值抑制 + 滞后阈值）
    refined = detector.apply_non_max_suppression(magnitude, direction)
    edges = detector.apply_hysteresis_threshold(refined, 50, 150)
    axes[1, 2].imshow(edges, cmap='gray')
    axes[1, 2].set_title('Edge Detection (NMS + Hysteresis)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"结果已保存到：{save_path}")
    
    plt.show()


def extract_sobel_features(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    从图像中提取 Sobel 特征（用于 SVM 分类）
    
    Args:
        image: 输入图像（RGB 或灰度）
        ksize: Sobel 算子核大小
        
    Returns:
        展平的特征向量
    """
    detector = SobelEdgeDetector(ksize=ksize)
    
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # 计算 Sobel 梯度
    gx, gy, magnitude, direction = detector.detect_edges(gray)
    
    # 归一化
    gx_norm = detector.normalize_gradient(gx)
    gy_norm = detector.normalize_gradient(gy)
    mag_norm = detector.normalize_gradient(magnitude)
    
    # 组合特征：原始灰度 + X 梯度 + Y 梯度 + 幅值
    features = np.stack([gray, gx_norm, gy_norm, mag_norm], axis=-1)
    
    # 展平为特征向量
    return features.flatten()


def demo_sobel():
    """Sobel 边缘检测演示"""
    # 创建一个简单的测试图像
    test_image = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(test_image, (50, 50), (150, 150), 255, -1)
    cv2.circle(test_image, (100, 100), 30, 0, -1)
    cv2.line(test_image, (0, 0), (200, 200), 255, 2)
    
    detector = SobelEdgeDetector(ksize=3)
    gx, gy, magnitude, direction = detector.detect_edges(test_image)
    
    visualize_sobel_results(
        test_image, gx, gy, magnitude, direction,
        save_path="results/sobel_demo.png"
    )
    
    print("Sobel 边缘检测演示完成！")


if __name__ == "__main__":
    demo_sobel()
