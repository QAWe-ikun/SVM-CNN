"""
CIFAR-10 图像分类系统
Sobel 边缘检测 + SVM vs CNN 对比分析
"""

from .sobel_edge_detection import SobelEdgeDetector, extract_sobel_features, visualize_sobel_results
from .svm_classifier import SVMSobelClassifier, load_cifar10, train_and_evaluate_svm
from .cnn_classifier import CNNClassifier, VGGForCIFAR10, load_cifar10_torch, train_and_evaluate_cnn
from .comparison_analysis import ComparisonAnalyzer, run_complete_pipeline

__all__ = [
    # Sobel 边缘检测
    'SobelEdgeDetector',
    'extract_sobel_features',
    'visualize_sobel_results',
    
    # SVM 分类器
    'SVMSobelClassifier',
    'load_cifar10',
    'train_and_evaluate_svm',
    
    # CNN 分类器
    'CNNClassifier',
    'VGGForCIFAR10',
    'load_cifar10_torch',
    'train_and_evaluate_cnn',
    
    # 对比分析
    'ComparisonAnalyzer',
    'run_complete_pipeline',
]
