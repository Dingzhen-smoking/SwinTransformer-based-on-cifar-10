# check_cuda.py
import torch

def check_cuda():
    print("===== PyTorch CUDA 环境检测 =====")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        print(f"当前设备 ID: {device_id}")
        print(f"GPU 名称: {torch.cuda.get_device_name(device_id)}")
        print(f"显卡数量: {torch.cuda.device_count()}")

        # 测试显存分配
        print("\n>>> 测试显存分配 ...")
        print("分配前 allocated:", torch.cuda.memory_allocated()/1024**2, "MB")
        print("分配前 reserved :", torch.cuda.memory_reserved()/1024**2, "MB")

        # 在 GPU 上创建一个 tensor
        x = torch.randn(1000, 1000, device="cuda")
        print("分配后 allocated:", torch.cuda.memory_allocated()/1024**2, "MB")
        print("分配后 reserved :", torch.cuda.memory_reserved()/1024**2, "MB")

        del x
        torch.cuda.empty_cache()
        print("释放后 allocated:", torch.cuda.memory_allocated()/1024**2, "MB")
        print("释放后 reserved :", torch.cuda.memory_reserved()/1024**2, "MB")
    else:
        print("未检测到可用的 CUDA，当前使用 CPU。")


if __name__ == "__main__":
    check_cuda()
