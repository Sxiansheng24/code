import subprocess
import os
import sys


CONFIG = "projects/configs/AEFFSSC/AEFFSSC.py"
CHECKPOINT = "epoch_11.pth"
GPUS = 2
PORT = 29503 


extra_args = sys.argv[1:]  

# 设置PYTHONPATH环境变量
pythonpath_dir = os.path.dirname(os.path.abspath(__file__))
pythonpath_value = f"{os.path.join(pythonpath_dir, '..')}:" + os.environ.get('PYTHONPATH', '')
os.environ['PYTHONPATH'] = pythonpath_value

# 构建要执行的命令
cmd = [
          "python", "-m", "torch.distributed.launch",
          "--nproc_per_node={}".format(GPUS),
          "--master_port={}".format(PORT),
          os.path.join(os.path.dirname(__file__), "test.py"),
          CONFIG, CHECKPOINT,
          "--launcher", "pytorch",
          "--eval", "bbox"
      ] + extra_args  


try:
    subprocess.check_call(cmd)
except subprocess.CalledProcessError as e:
    print(f"Command failed with return code {e.returncode}")
    sys.exit(e.returncode)
