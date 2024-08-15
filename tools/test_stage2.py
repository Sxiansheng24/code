import subprocess
import os
import sys

# 定义从.sh脚本中传递的参数
CONFIG = "./projects/configs/voxformer/voxformer-T.py"
CHECKPOINT = "/home/ubuntu/VoxFormer/权重/stage2/stage2_T/stage2_T_mIou_5208pre_7844/epoch_11.pth"
GPUS = 2
PORT = 29503  # 如果命令行中没有提供PORT，则使用默认值

# 附加传递给test.py的额外参数
extra_args = sys.argv[1:]  # 获取Python脚本后面的所有参数

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
      ] + extra_args  # 添加额外的参数

# 执行命令
try:
    subprocess.check_call(cmd)
except subprocess.CalledProcessError as e:
    print(f"Command failed with return code {e.returncode}")
    sys.exit(e.returncode)