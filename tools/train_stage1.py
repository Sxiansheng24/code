import subprocess
import os
import sys

# 检查命令行参数数量
if len(sys.argv) < 3:
    print("Usage: python dist_train.py <config_file> <gpus> [other_args]")
    sys.exit(1)

# 获取命令行参数
CONFIG = sys.argv[1]
GPUS = int(sys.argv[2])
OTHER_ARGS = sys.argv[3:]

# 获取环境变量PORT，如果不存在则使用默认值
PORT = os.environ.get('PORT', 28509)

# 构造PYTHONPATH，使其包含当前脚本的父目录
pythonpath_dir = os.path.dirname(os.path.abspath(__file__))
pythonpath_parent = os.path.join(pythonpath_dir, '..')
pythonpath = f"{pythonpath_parent}:{os.environ.get('PYTHONPATH', '')}"

# 设置环境变量PYTHONPATH
os.environ['PYTHONPATH'] = pythonpath

# 构造完整的命令行
cmd = [
          'python', '-m', 'torch.distributed.launch',
          '--nproc_per_node', str(GPUS),
          '--master_port', str(PORT),
          os.path.join(pythonpath_dir, 'train.py'),
          CONFIG,
          '--launcher', 'pytorch'
      ] + OTHER_ARGS + ['--deterministic']

# 打印出将要执行的命令
print(f"Executing command: {' '.join(cmd)}")

# 运行命令
try:
    subprocess.check_call(cmd)
except subprocess.CalledProcessError as e:
    print(f"Command failed with return code {e.returncode}")
    print(e.output.decode('utf-8') if e.output else "No output was captured.")
    sys.exit(e.returncode)