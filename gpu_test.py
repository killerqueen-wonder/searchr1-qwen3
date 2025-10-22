import torch
import time

def keep_gpu_busy():
    # 检查GPU是否可用
    if not torch.cuda.is_available():
        print("CUDA不可用，请检查您的PyTorch安装")
        return
    
    device = torch.device("cuda")
    print(f"正在使用GPU: {torch.cuda.get_device_name(0)}")
    
    print("GPU压力测试已启动... (按Ctrl+C停止)")
    
    try:
        while True:
            # 创建随机张量并移动到GPU
            a = torch.rand(2048, 2048).to(device)
            b = torch.rand(2048, 2048).to(device)
            
            # 执行矩阵乘法（占用GPU计算资源）
            c = torch.mm(a, b)
            
            # 可选：添加小延迟防止过热
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nGPU压力测试已停止")

if __name__ == "__main__":
    keep_gpu_busy()