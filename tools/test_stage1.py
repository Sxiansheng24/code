import subprocess
import os
import sys


CONFIG = "./projects/configs/AEFFSSC/qpn.py"
CHECKPOINT = "/media/ubuntu/新加卷/epoch_36.pth"
GPUS = 2
PORT = 29503  


extra_args = sys.argv[1:]  # 获取Python脚本后面的所有参数

# 设置PYTHONPATH环境变量
pythonpath_dir = os.path.dirname(os.path.abspath(__file__))
pythonpath_value = f"{os.path.join(pythonpath_dir, '..')}:" + os.environ.get('PYTHONPATH', '')
os.environ['PYTHONPATH'] = pythonpath_value


cmd = [
          "python", "-m", "torch.distributed.launch",
          "--nproc_per_node={}".format(GPUS),
          "--master_port={}".format(PORT),
          os.path.join(os.path.dirname(__file__), "test.py"),
          CONFIG, CHECKPOINT,
          "--launcher", "pytorch",
          "--eval", "bbox"
      ] + extra_args  # 添加额外的参数


try:
    subprocess.check_call(cmd)
except subprocess.CalledProcessError as e:
    print(f"Command failed with return code {e.returncode}")
    sys.exit(e.returncode)
