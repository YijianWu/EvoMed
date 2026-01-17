"""
Web UI 测试脚本
用于验证 Web UI 所需的依赖和配置是否正确
"""

import sys
import os

def check_dependencies():
    """检查依赖包"""
    print("="*60)
    print("检查依赖包...")
    print("="*60)
    
    required_packages = {
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'json': 'json（内置）',
        'openai': 'openai'
    }
    
    missing = []
    for package, display_name in required_packages.items():
        try:
            if package == 'json':
                import json
            else:
                __import__(package)
            print(f"✅ {display_name}")
        except ImportError:
            print(f"❌ {display_name} - 未安装")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️ 缺少依赖包: {', '.join(missing)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("\n✅ 所有依赖包已安装")
    return True


def check_files():
    """检查必需文件"""
    print("\n" + "="*60)
    print("检查必需文件...")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    required_files = {
        'src/web_ui.py': 'Web UI 主文件',
        'src/diagnosis_api.py': '诊断 API',
        'run_webui.sh': '启动脚本',
        'requirements.txt': '依赖配置'
    }
    
    missing = []
    for file_path, description in required_files.items():
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            print(f"✅ {description}: {file_path}")
        else:
            print(f"❌ {description}: {file_path} - 文件不存在")
            missing.append(file_path)
    
    if missing:
        print(f"\n⚠️ 缺少文件: {', '.join(missing)}")
        return False
    
    print("\n✅ 所有必需文件存在")
    return True


def check_expert_pool():
    """检查专家池文件"""
    print("\n" + "="*60)
    print("检查专家池文件...")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    expert_pool_path = os.path.join(base_dir, 'outputs', 'optimized_expert_pool_28.json')
    
    if os.path.exists(expert_pool_path):
        print(f"✅ 28专家池文件: {expert_pool_path}")
        
        try:
            import json
            with open(expert_pool_path, 'r', encoding='utf-8') as f:
                pool = json.load(f)
            print(f"   专家数量: {len(pool)}")
            print("   ✅ 简化版可以使用")
            return True
        except Exception as e:
            print(f"   ⚠️ 文件格式错误: {e}")
            return False
    else:
        print(f"⚠️ 28专家池文件不存在: {expert_pool_path}")
        print("   简化版功能可能受限")
        return False


def check_api_config():
    """检查 API 配置"""
    print("\n" + "="*60)
    print("检查 API 配置...")
    print("="*60)
    
    try:
        from src.diagnosis_api import API_BASE_URL, API_KEY, MODEL_NAME
        
        print(f"API Base URL: {API_BASE_URL}")
        print(f"Model Name: {MODEL_NAME}")
        
        if API_KEY and len(API_KEY) > 10:
            masked_key = API_KEY[:10] + "..." + API_KEY[-4:]
            print(f"API Key: {masked_key}")
            print("✅ API 配置已设置")
            return True
        else:
            print("⚠️ API Key 未正确配置")
            print("请编辑 src/diagnosis_api.py 配置您的 API 密钥")
            return False
            
    except Exception as e:
        print(f"❌ 读取配置失败: {e}")
        return False


def print_usage_guide():
    """打印使用指南"""
    print("\n" + "="*60)
    print("🚀 启动 Web UI")
    print("="*60)
    print("\n方法 1: 使用启动脚本（推荐）")
    print("  $ ./run_webui.sh")
    print("\n方法 2: 直接使用 streamlit")
    print("  $ streamlit run src/web_ui.py --server.port 8501")
    print("\n访问地址:")
    print("  http://localhost:8501")
    print("\n详细文档:")
    print("  - README.md - 完整使用手册")
    print("  - QUICK_START_WEBUI.md - 快速启动指南")
    print("\n" + "="*60)


def main():
    """主测试函数"""
    print("\n")
    print("="*60)
    print("医疗诊断系统 Web UI - 环境检查")
    print("="*60)
    print("")
    
    all_ok = True
    
    # 1. 检查依赖
    if not check_dependencies():
        all_ok = False
    
    # 2. 检查文件
    if not check_files():
        all_ok = False
    
    # 3. 检查专家池
    check_expert_pool()  # 这个不是必需的
    
    # 4. 检查 API 配置
    if not check_api_config():
        all_ok = False
    
    # 总结
    print("\n" + "="*60)
    if all_ok:
        print("✅ 环境检查通过！可以启动 Web UI")
        print("="*60)
        print_usage_guide()
        return 0
    else:
        print("❌ 环境检查失败，请解决上述问题后重试")
        print("="*60)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


