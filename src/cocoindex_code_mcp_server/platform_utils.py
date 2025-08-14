
import os
import sys

def is_replit_environment():
    """Check if running in Replit environment"""
    return 'REPLIT_ENVIRONMENT' in os.environ or 'REPL_ID' in os.environ

def has_cuda_support():
    """Check if CUDA is available"""
    if is_replit_environment():
        return False
    
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def get_device():
    """Get the appropriate device (cuda/cpu)"""
    if has_cuda_support():
        return 'cuda'
    return 'cpu'

def install_cpu_only_torch():
    """Install CPU-only version of PyTorch if needed"""
    if is_replit_environment():
        try:
            import subprocess
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                'torch', 'torchvision', '--index-url', 
                'https://download.pytorch.org/whl/cpu'
            ])
        except Exception as e:
            print(f"Warning: Could not install CPU-only PyTorch: {e}")
