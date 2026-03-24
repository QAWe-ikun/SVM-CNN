"""
对比分析模块
对比 SVM 和 CNN 在 CIFAR-10 上的准确率和计算效率
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
from datetime import datetime


class ComparisonAnalyzer:
    """SVM 和 CNN 对比分析器"""
    
    def __init__(self, results_dir: str = "results"):
        """
        初始化分析器
        
        Args:
            results_dir: 结果目录
        """
        self.results_dir = results_dir
        self.svm_results = None
        self.cnn_results = None
        self.class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
                           'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    
    def load_results(self, svm_path: str = None, cnn_path: str = None) -> bool: # type: ignore
        """
        加载已有的结果文件
        
        Args:
            svm_path: SVM 结果文件路径
            cnn_path: CNN 结果文件路径
            
        Returns:
            是否成功加载
        """
        if svm_path is None:
            svm_path = os.path.join(self.results_dir, "svm_results.json")
        if cnn_path is None:
            cnn_path = os.path.join(self.results_dir, "cnn_results.json")
        
        if os.path.exists(svm_path):
            with open(svm_path, 'r') as f:
                self.svm_results = json.load(f)
            print(f"已加载 SVM 结果：{svm_path}")
        else:
            print(f"未找到 SVM 结果文件：{svm_path}")
        
        if os.path.exists(cnn_path):
            with open(cnn_path, 'r') as f:
                self.cnn_results = json.load(f)
            print(f"已加载 CNN 结果：{cnn_path}")
        else:
            print(f"未找到 CNN 结果文件：{cnn_path}")
        
        return self.svm_results is not None and self.cnn_results is not None
    
    def set_results(self, svm_results: Dict, cnn_results: Dict) -> None:
        """
        直接设置结果
        
        Args:
            svm_results: SVM 结果字典
            cnn_results: CNN 结果字典
        """
        self.svm_results = svm_results
        self.cnn_results = cnn_results
    
    def compare_accuracy(self) -> Dict:
        """
        对比准确率
        
        Returns:
            准确率对比结果
        """
        if not self.svm_results or not self.cnn_results:
            raise ValueError("请先加载结果文件")
        
        svm_train_acc = self.svm_results['train_results']['train_accuracy']
        svm_test_acc = self.svm_results['eval_results']['test_accuracy']
        
        cnn_val_acc = self.cnn_results['train_results']['best_val_accuracy'] / 100.0
        cnn_test_acc = self.cnn_results['eval_results']['test_accuracy']
        
        comparison = {
            'svm_train_accuracy': svm_train_acc,
            'svm_test_accuracy': svm_test_acc,
            'cnn_val_accuracy': cnn_val_acc,
            'cnn_test_accuracy': cnn_test_acc,
            'accuracy_difference': cnn_test_acc - svm_test_acc,
            'relative_improvement': (cnn_test_acc - svm_test_acc) / svm_test_acc * 100
        }
        
        return comparison
    
    def compare_efficiency(self) -> Dict:
        """
        对比计算效率
        
        Returns:
            效率对比结果
        """
        if not self.svm_results or not self.cnn_results:
            raise ValueError("请先加载结果文件")
        
        svm_train_time = self.svm_results['train_results']['training_time']
        svm_feature_time = self.svm_results['train_results']['feature_extraction_time']
        svm_inference_time = self.svm_results['eval_results']['avg_inference_time_per_sample']
        
        cnn_train_time = self.cnn_results['train_results']['training_time']
        cnn_inference_time = self.cnn_results['eval_results']['avg_inference_time_per_sample']
        
        comparison = {
            'svm_training_time': svm_train_time,
            'svm_feature_extraction_time': svm_feature_time,
            'svm_total_train_time': svm_train_time + svm_feature_time,
            'svm_inference_time_per_sample': svm_inference_time,
            'cnn_training_time': cnn_train_time,
            'cnn_inference_time_per_sample': cnn_inference_time,
            'training_speedup': svm_train_time / cnn_train_time if cnn_train_time > 0 else float('inf'),
            'inference_speedup': svm_inference_time / cnn_inference_time if cnn_inference_time > 0 else float('inf')
        }
        
        return comparison
    
    def plot_accuracy_comparison(self, save_path: Optional[str] = None) -> None:
        """绘制准确率对比图"""
        comparison = self.compare_accuracy()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = ['SVM\n+Sobel/HOG', 'CNN\n(VGG)']
        train_accs = [comparison['svm_train_accuracy'], 0]  # CNN 用验证集准确率
        test_accs = [comparison['svm_test_accuracy'], comparison['cnn_test_accuracy']]
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, [t*100 for t in train_accs[:1]] + [comparison['cnn_val_accuracy']*100], 
                       width, label='Training/Val Accuracy', color='#3498db')
        bars2 = ax.bar(x + width/2, [t*100 for t in test_accs], 
                       width, label='Test Accuracy', color='#e74c3c')
        
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('SVM vs CNN: Accuracy Comparison on CIFAR-10')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"准确率对比图已保存到：{save_path}")
        
        plt.show()
    
    def plot_efficiency_comparison(self, save_path: Optional[str] = None) -> None:
        """绘制效率对比图"""
        efficiency = self.compare_efficiency()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 训练时间对比
        ax1 = axes[0]
        methods = ['SVM\n+Sobel/HOG', 'CNN\n(VGG)']
        train_times = [
            efficiency['svm_total_train_time'],
            efficiency['cnn_training_time']
        ]
        colors = ['#3498db', '#e74c3c']
        
        bars1 = ax1.bar(methods, train_times, color=colors, alpha=0.8)
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Training Time Comparison')
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, time_val in zip(bars1, train_times):
            height = bar.get_height()
            ax1.annotate(f'{time_val:.1f}s\n({time_val/60:.1f}min)',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        
        # 推理时间对比
        ax2 = axes[1]
        inference_times = [
            efficiency['svm_inference_time_per_sample'] * 1000,  # 转换为 ms
            efficiency['cnn_inference_time_per_sample'] * 1000
        ]
        
        bars2 = ax2.bar(methods, inference_times, color=colors, alpha=0.8)
        ax2.set_ylabel('Time (ms per sample)')
        ax2.set_title('Inference Time Comparison (per sample)')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, time_val in zip(bars2, inference_times):
            height = bar.get_height()
            ax2.annotate(f'{time_val:.2f}ms',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"效率对比图已保存到：{save_path}")
        
        plt.show()
    
    def plot_radar_comparison(self, save_path: Optional[str] = None) -> None:
        """绘制雷达图对比"""
        comparison_acc = self.compare_accuracy()
        comparison_eff = self.compare_efficiency()
        
        # 归一化各项指标（0-1 范围，越大越好）
        # 准确率：直接用
        # 训练时间：取倒数归一化
        # 推理时间：取倒数归一化
        
        max_train_time = max(comparison_eff['svm_total_train_time'], 
                            comparison_eff['cnn_training_time'])
        max_infer_time = max(comparison_eff['svm_inference_time_per_sample'],
                            comparison_eff['cnn_inference_time_per_sample'])
        
        svm_scores = [
            comparison_acc['svm_test_accuracy'],  # 测试准确率
            comparison_eff['svm_total_train_time'] / max_train_time,  # 训练效率（越小越好，所以用原值）
            comparison_eff['svm_inference_time_per_sample'] / max_infer_time,  # 推理效率
            comparison_acc['svm_train_accuracy']  # 训练准确率
        ]
        
        cnn_scores = [
            comparison_acc['cnn_test_accuracy'],
            comparison_eff['cnn_training_time'] / max_train_time,
            comparison_eff['cnn_inference_time_per_sample'] / max_infer_time,
            comparison_acc['cnn_val_accuracy']
        ]
        
        # 反转时间和效率分数（因为越小越好）
        svm_scores[1] = 1 - svm_scores[1]
        svm_scores[2] = 1 - svm_scores[2]
        cnn_scores[1] = 1 - cnn_scores[1]
        cnn_scores[2] = 1 - cnn_scores[2]
        
        categories = ['Test\nAccuracy', 'Training\nEfficiency', 
                     'Inference\nEfficiency', 'Training\nAccuracy']
        N = len(categories)
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        svm_values = svm_scores + [svm_scores[0]]
        cnn_values = cnn_scores + [cnn_scores[0]]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        ax.plot(angles, svm_values, 'o-', linewidth=2, label='SVM + Sobel/HOG', 
               color='#3498db', markersize=8)
        ax.fill(angles, svm_values, alpha=0.25, color='#3498db')
        
        ax.plot(angles, cnn_values, 'o-', linewidth=2, label='CNN (VGG)', 
               color='#e74c3c', markersize=8)
        ax.fill(angles, cnn_values, alpha=0.25, color='#e74c3c')
        
        ax.set_theta_offset(np.pi / 2) # type: ignore
        ax.set_theta_direction(-1) # type: ignore
        ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=11) # type: ignore
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=9) # type: ignore
        ax.set_ylim(0, 1)
        
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.set_title('SVM vs CNN: Comprehensive Comparison\n', fontsize=14, pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"雷达对比图已保存到：{save_path}")
        
        plt.show()
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        生成详细的对比报告
        
        Args:
            save_path: 保存路径
            
        Returns:
            报告文本
        """
        comparison_acc = self.compare_accuracy()
        comparison_eff = self.compare_efficiency()
        
        report = f"""
# CIFAR-10 图像分类：SVM vs CNN 对比分析报告

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 一、实验设置

### 1.1 数据集
- 数据集：CIFAR-10
- 图像尺寸：32x32 像素
- 类别数：10 类
- 训练样本：50,000 张
- 测试样本：10,000 张

### 1.2 方法对比

| 特性 | SVM + Sobel/HOG | CNN (VGG) |
|------|-----------------|-----------|
| 特征提取 | Sobel 边缘检测 + HOG | 端到端学习 |
| 模型类型 | 支持向量机 | 卷积神经网络 |
| 训练方式 | 批量训练 | 小批量梯度下降 |
| 数据增强 | 无 | 随机翻转、裁剪、色彩抖动 |

## 二、准确率对比

### 2.1 测试结果

| 指标 | SVM + Sobel/HOG | CNN (VGG) |
|------|-----------------|-----------|
| 训练集准确率 | {comparison_acc['svm_train_accuracy']*100:.2f}% | {comparison_acc['cnn_val_accuracy']*100:.2f}% (验证集) |
| 测试集准确率 | {comparison_acc['svm_test_accuracy']*100:.2f}% | {comparison_acc['cnn_test_accuracy']*100:.2f}% |
| 准确率提升 | - | +{comparison_acc['relative_improvement']:.2f}% |

### 2.2 分析

CNN 模型相比 SVM 模型的准确率提升了 {comparison_acc['accuracy_difference']*100:.2f} 个百分点，
相对提升幅度为 {comparison_acc['relative_improvement']:.2f}%。

CNN 的优势在于：
1. 端到端特征学习，无需手工设计特征
2. 卷积操作能有效捕捉空间局部相关性
3. 深层网络能学习更抽象的特征表示
4. 数据增强提高了模型泛化能力

## 三、计算效率对比

### 3.1 训练时间

| 阶段 | SVM + Sobel/HOG | CNN (VGG) |
|------|-----------------|-----------|
| 特征提取 | {comparison_eff['svm_feature_extraction_time']:.2f} 秒 | - (端到端) |
| 模型训练 | {comparison_eff['svm_training_time']:.2f} 秒 | {comparison_eff['cnn_training_time']:.2f} 秒 |
| 总计 | {comparison_eff['svm_total_train_time']:.2f} 秒 | {comparison_eff['cnn_training_time']:.2f} 秒 |

### 3.2 推理时间

| 模型 | 单样本推理时间 |
|------|---------------|
| SVM + Sobel/HOG | {comparison_eff['svm_inference_time_per_sample']*1000:.2f} ms |
| CNN (VGG) | {comparison_eff['cnn_inference_time_per_sample']*1000:.2f} ms |

### 3.3 分析

SVM 的优势：
1. 训练速度较快（特别是对于中小规模数据集）
2. 无需 GPU 即可高效训练
3. 模型简单，易于部署

CNN 的优势：
1. 推理速度更快（在 GPU 上）
2. 批量处理效率更高
3. 可充分利用硬件加速

## 四、总结与建议

### 4.1 方法选择建议

**选择 SVM + Sobel/HOG 的场景：**
- 数据集较小（< 10,000 样本）
- 计算资源有限（无 GPU）
- 需要快速原型开发
- 需要模型可解释性

**选择 CNN 的场景：**
- 数据集较大
- 有 GPU 等加速硬件
- 追求最高准确率
- 有充足的训练时间

### 4.2 性能总结

| 评估维度 | 获胜方法 | 优势幅度 |
|----------|----------|----------|
| 测试准确率 | CNN (VGG) | +{comparison_acc['relative_improvement']:.2f}% |
| 训练效率 | SVM + Sobel/HOG | {comparison_eff['training_speedup']:.2f}x |
| 推理效率 | {'CNN (VGG)' if comparison_eff['inference_speedup'] > 1 else 'SVM + Sobel/HOG'} | {max(comparison_eff['inference_speedup'], 1/comparison_eff['inference_speedup']):.2f}x |

---
报告结束
"""
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"对比报告已保存到：{save_path}")
        
        return report
    
    def run_full_comparison(self, output_dir: str = "results") -> None:
        """
        运行完整的对比分析并保存所有结果
        
        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 60)
        print("开始 SVM vs CNN 对比分析")
        print("=" * 60)
        
        # 绘制所有对比图
        self.plot_accuracy_comparison(
            save_path=os.path.join(output_dir, "accuracy_comparison.png"))
        
        self.plot_efficiency_comparison(
            save_path=os.path.join(output_dir, "efficiency_comparison.png"))
        
        self.plot_radar_comparison(
            save_path=os.path.join(output_dir, "radar_comparison.png"))
        
        # 生成报告
        report = self.generate_report(
            save_path=os.path.join(output_dir, "comparison_report.md"))
        
        # 保存汇总数据
        summary = {
            'accuracy_comparison': self.compare_accuracy(),
            'efficiency_comparison': self.compare_efficiency(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, "comparison_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "=" * 60)
        print("对比分析完成！")
        print("=" * 60)
        print(f"结果已保存到：{output_dir}/")
        
        # 打印关键结果
        acc_comp = self.compare_accuracy()
        eff_comp = self.compare_efficiency()
        
        print(f"\n关键结果:")
        print(f"  SVM 测试准确率：{acc_comp['svm_test_accuracy']*100:.2f}%")
        print(f"  CNN 测试准确率：{acc_comp['cnn_test_accuracy']*100:.2f}%")
        print(f"  CNN 相对提升：{acc_comp['relative_improvement']:.2f}%")
        print(f"\n  SVM 训练时间：{eff_comp['svm_total_train_time']:.2f} 秒")
        print(f"  CNN 训练时间：{eff_comp['cnn_training_time']:.2f} 秒")


def run_complete_pipeline():
    """运行完整的训练和对比流程"""
    print("=" * 60)
    print("CIFAR-10 图像分类：SVM vs CNN 完整对比")
    print("=" * 60)

    # 创建目录
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 导入训练函数
    from svm_classifier import train_and_evaluate_svm
    from cnn_classifier import train_and_evaluate_cnn

    # 训练 SVM
    print("\n" + "=" * 60)
    print("第一步：训练 SVM 分类器")
    print("=" * 60)
    svm_results = train_and_evaluate_svm()

    # 训练 CNN
    print("\n" + "=" * 60)
    print("第二步：训练 CNN 分类器")
    print("=" * 60)
    cnn_results = train_and_evaluate_cnn()

    # 对比分析
    print("\n" + "=" * 60)
    print("第三步：对比分析")
    print("=" * 60)

    analyzer = ComparisonAnalyzer()
    analyzer.set_results(svm_results, cnn_results) # type: ignore
    analyzer.run_full_comparison()

    return svm_results, cnn_results


if __name__ == "__main__":
    # 如果已有结果文件，直接分析
    analyzer = ComparisonAnalyzer()
    
    if analyzer.load_results():
        print("发现已有结果，开始对比分析...")
        analyzer.run_full_comparison()
    else:
        print("未找到结果文件，开始完整训练流程...")
        print("这将需要较长时间，请确保已安装所有依赖。")
        print()
        run_complete_pipeline()
