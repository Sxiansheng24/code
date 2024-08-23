import subprocess
import os
import sys


if len(sys.argv) < 3:
    print("Usage: python dist_train.py <config_file> <gpus> [other_args]")
    sys.exit(1)


CONFIG = sys.argv[1]
GPUS = int(sys.argv[2])
OTHER_ARGS = sys.argv[3:]

PORT = os.environ.get('PORT', 28509)


pythonpath_dir = os.path.dirname(os.path.abspath(__file__))
pythonpath_parent = os.path.join(pythonpath_dir, '..')
pythonpath = f"{pythonpath_parent}:{os.environ.get('PYTHONPATH', '')}"


os.environ['PYTHONPATH'] = pythonpath


cmd = [
          'python', '-m', 'torch.distributed.launch',
          '--nproc_per_node', str(GPUS),
          '--master_port', str(PORT),
          os.path.join(pythonpath_dir, 'train.py'),
          CONFIG,
          '--launcher', 'pytorch'
      ] + OTHER_ARGS + ['--deterministic']


print(f"Executing command: {' '.join(cmd)}")


try:
    subprocess.check_call(cmd)
except subprocess.CalledProcessError as e:
    print(f"Command failed with return code {e.returncode}")
    print(e.output.decode('utf-8') if e.output else "No output was captured.")
    sys.exit(e.returncode)
