"""
主程序入口
统一调用 Sobel 边缘检测、SVM 分类、CNN 分类和对比分析
"""

import os
import sys
import time
import argparse
from datetime import datetime

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def print_banner():
    print("=" * 70)
    print(" " * 15 + "CIFAR-10 图像分类系统")
    print(" " * 10 + "Sobel 边缘检测 | SVM 分类 | CNN 分类 | 对比分析")
    print("=" * 70)
    print()


def print_menu():
    """打印主菜单"""
    print("\n请选择要执行的操作:")
    print("-" * 50)
    print("  1. Sobel 边缘检测演示")
    print("  2. 训练 SVM 分类器")
    print("  3. 训练 CNN 分类器")
    print("  4. 运行完整对比分析")
    print("  5. 仅生成对比报告（已有结果时）")
    print("  6. 查看模型信息")
    print("  0. 退出程序")
    print("-" * 50)


def run_sobel_demo():
    """运行 Sobel 边缘检测演示"""
    print("\n" + "=" * 50)
    print("Sobel 边缘检测演示")
    print("=" * 50)
    
    try:
        from sobel_edge_detection import demo_sobel
        demo_sobel()
        print("\n✓ Sobel 边缘检测演示完成！")
    except Exception as e:
        print(f"\n✗ 错误：{e}")
        import traceback
        traceback.print_exc()


def train_svm():
    """训练 SVM 分类器"""
    print("\n" + "=" * 50)
    print("训练 SVM 分类器")
    print("=" * 50)
    
    try:
        from svm_classifier import train_and_evaluate_svm
        results = train_and_evaluate_svm()
        print("\n✓ SVM 训练完成！")
        return results
    except Exception as e:
        print(f"\n✗ 错误：{e}")
        import traceback
        traceback.print_exc()
        return None


def train_cnn():
    """训练 CNN 分类器"""
    print("\n" + "=" * 50)
    print("训练 CNN 分类器")
    print("=" * 50)
    
    try:
        from cnn_classifier import train_and_evaluate_cnn
        results = train_and_evaluate_cnn()
        print("\n✓ CNN 训练完成！")
        return results
    except Exception as e:
        print(f"\n✗ 错误：{e}")
        import traceback
        traceback.print_exc()
        return None


def run_comparison():
    """运行完整对比分析"""
    print("\n" + "=" * 50)
    print("运行完整对比分析")
    print("=" * 50)
    
    try:
        from comparison_analysis import run_complete_pipeline, ComparisonAnalyzer
        
        # 检查结果文件是否存在
        analyzer = ComparisonAnalyzer()
        if analyzer.load_results():
            print("\n发现已有结果文件。")
            choice = input("是否重新训练模型？(y/n): ").strip().lower()
            if choice == 'y':
                run_complete_pipeline()
            else:
                analyzer.run_full_comparison()
        else:
            print("\n未找到结果文件，开始完整训练流程...")
            print("这将需要较长时间，请耐心等待。\n")
            run_complete_pipeline()
        
        print("\n✓ 对比分析完成！")
    except Exception as e:
        print(f"\n✗ 错误：{e}")
        import traceback
        traceback.print_exc()


def generate_report_only():
    """仅生成对比报告"""
    print("\n" + "=" * 50)
    print("生成对比报告")
    print("=" * 50)
    
    try:
        from comparison_analysis import ComparisonAnalyzer
        
        analyzer = ComparisonAnalyzer()
        if analyzer.load_results():
            analyzer.run_full_comparison()
            print("\n✓ 报告生成完成！")
        else:
            print("\n✗ 未找到结果文件，请先训练模型。")
    except Exception as e:
        print(f"\n✗ 错误：{e}")
        import traceback
        traceback.print_exc()


def show_model_info():
    """查看模型信息"""
    print("\n" + "=" * 50)
    print("模型信息")
    print("=" * 50)

    models_dir = "models"
    results_dir = "results"

    # 检查模型文件
    print("\n模型文件:")
    if os.path.exists(models_dir):
        model_files = os.listdir(models_dir)
        if model_files:
            for f in model_files:
                filepath = os.path.join(models_dir, f)
                size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                print(f"  - {f} ({size:.2f} MB, 修改时间：{mtime})")
        else:
            print("  (空目录)")
    else:
        print("  (目录不存在)")

    # 检查结果文件
    print("\n结果文件:")
    if os.path.exists(results_dir):
        result_files = os.listdir(results_dir)
        if result_files:
            for f in result_files:
                filepath = os.path.join(results_dir, f)
                size = os.path.getsize(filepath) / 1024  # KB
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                print(f"  - {f} ({size:.1f} KB, 修改时间：{mtime})")
        else:
            print("  (空目录)")
    else:
        print("  (目录不存在)")
    
    # 尝试加载并显示结果摘要
    print("\n结果摘要:")
    try:
        from comparison_analysis import ComparisonAnalyzer
        import json
        
        analyzer = ComparisonAnalyzer()
        if analyzer.load_results():
            acc_comp = analyzer.compare_accuracy()
            eff_comp = analyzer.compare_efficiency()
            
            print(f"\n  准确率对比:")
            print(f"    SVM 测试准确率：{acc_comp['svm_test_accuracy']*100:.2f}%")
            print(f"    CNN 测试准确率：{acc_comp['cnn_test_accuracy']*100:.2f}%")
            print(f"    CNN 相对提升：+{acc_comp['relative_improvement']:.2f}%")
            
            print(f"\n  效率对比:")
            print(f"    SVM 训练时间：{eff_comp['svm_total_train_time']:.2f} 秒")
            print(f"    CNN 训练时间：{eff_comp['cnn_training_time']:.2f} 秒")
            print(f"    SVM 推理时间：{eff_comp['svm_inference_time_per_sample']*1000:.2f} ms/样本")
            print(f"    CNN 推理时间：{eff_comp['cnn_inference_time_per_sample']*1000:.2f} ms/样本")
        else:
            print("  (无可用结果)")
    except Exception as e:
        print(f"  (无法加载结果：{e})")

def check_dependencies():
    """检查依赖库"""
    print("\n检查依赖库...")
    
    required_packages = {
        'numpy': 'numpy',
        'cv2': 'opencv-python',
        'matplotlib': 'matplotlib',
        'sklearn': 'scikit-learn',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'tqdm': 'tqdm'
    }
    
    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (未安装)")
            missing.append(package)
    
    if missing:
        print(f"\n缺少依赖库，请运行以下命令安装:")
        print(f"  pip install {' '.join(missing)}")
        print(f"  或：pip install -r requirements.txt")
        return False
    
    print("\n✓ 所有依赖库已安装！")
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='CIFAR-10 图像分类系统 - Sobel + SVM vs CNN'
    )
    parser.add_argument('--mode', type=str, default='interactive',
                       choices=['interactive', 'sobel', 'svm', 'cnn', 'compare', 'report'],
                       help='运行模式：interactive(交互), sobel, svm, cnn, compare, report')
    parser.add_argument('--epochs', type=int, default=50,
                       help='CNN 训练轮数（默认：50）')
    parser.add_argument('--check-deps', action='store_true',
                       help='仅检查依赖')
    
    args = parser.parse_args()
    
    # 仅检查依赖
    if args.check_deps:
        check_dependencies()
        return
    
    print_banner()
    
    # 检查依赖
    if not check_dependencies():
        print("\n请先安装所有依赖库后再运行。")
        return
    
    # 创建必要目录
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # 交互模式
    if args.mode == 'interactive':
        while True:
            print_menu()
            choice = input("\n请输入选项 (0-6): ").strip()

            if choice == '1':
                run_sobel_demo()
            elif choice == '2':
                train_svm()
            elif choice == '3':
                train_cnn()
            elif choice == '4':
                run_comparison()
            elif choice == '5':
                generate_report_only()
            elif choice == '6':
                show_model_info()
            elif choice == '0':
                print("\n感谢使用，再见！")
                break
            else:
                print("\n无效选项，请重新输入。")
    
    # 命令行模式
    elif args.mode == 'sobel':
        run_sobel_demo()
    elif args.mode == 'svm':
        train_svm()
    elif args.mode == 'cnn':
        train_cnn()
    elif args.mode == 'compare':
        run_comparison()
    elif args.mode == 'report':
        generate_report_only()


if __name__ == "__main__":
    main()
