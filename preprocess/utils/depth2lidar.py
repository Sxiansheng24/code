import numpy as np
import os
import argparse
import kitti_util  # 假设这是处理KITTI数据集的自定义工具库
import argparse
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kitti_util



# 修改project_disp_to_depth函数以生成规整的网格点云
def generate_structured_point_cloud(calib, depth_layers, max_high, width=256, height=256):
    # 创建一个256x256的网格
    x_coords, y_coords = np.meshgrid(np.linspace(0, width - 1, width), np.linspace(0, height - 1, height),
                                     indexing='ij')
    points = np.stack([x_coords, y_coords], axis=-1).reshape(-1, 2)

    # 为每个网格点生成32个深度层
    depth_values = np.linspace(0, max_high, depth_layers)
    all_points = []
    for depth in depth_values:
        # 复制网格点到每个深度层
        layer_points = np.hstack((points, depth * np.ones((points.shape[0], 1))))
        # 转换为齐次坐标
        layer_points = np.hstack((layer_points, np.ones((layer_points.shape[0], 1))))
        # 投影到激光雷达坐标系
        cloud_points = calib.project_image_to_velo(layer_points.T).T
        # 添加到总点云中
        all_points.append(cloud_points)

        # 合并所有深度层的点云
    structured_cloud = np.vstack(all_points)

    # 过滤掉无效点（例如，Z坐标小于0或大于等于max_high的点）
    valid = (structured_cloud[:, 2] >= 0) & (structured_cloud[:, 2] < max_high)
    return structured_cloud[valid]


if __name__ == '__main__':
    # ...（省略了部分代码，包括参数解析和目录检查）...
    parser = argparse.ArgumentParser(description='Generate Libar')
    parser.add_argument('--calib_dir', type=str, default='kitti_dir/dataset/sequences/00')
    parser.add_argument('--depth_dir', type=str, default='depth_dir/sequences/00')
    parser.add_argument('--save_dir', type=str, default='lidar_dir/sequences/00')
    parser.add_argument('--max_high', type=int, default=80)
    args = parser.parse_args()

    assert os.path.isdir(args.depth_dir)
    assert os.path.isdir(args.calib_dir)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    depths = [x for x in os.listdir(args.depth_dir) if x[-3:] == 'npy' and 'std' not in x]
    depths = sorted(depths)

    depth_layers = 32  # 设置深度层数为32
    width = 256  # 设置网格宽度为256
    height = 256  # 设置网格高度为256

    for fn in depths:
        for fn in depths:
            predix = fn[:-4]
            # predix = fn[:-8]
            # calib_file = '{}/{}.txt'.format(args.calib_dir, predix)
            calib_file = '{}/{}.txt'.format(args.calib_dir, 'calib')
            calib = kitti_util.Calibration(calib_file)
            depth_map = np.load(args.depth_dir + '/' + fn)



        # 调用generate_structured_point_cloud生成规整的网格点云
            lidar = generate_structured_point_cloud(calib, depth_layers, args.max_high, width, height)

        # 转换为float32以确保每个值占用4字节，并保存到.bin文件
            lidar = lidar.astype(np.float32)
            lidar.tofile('{}/{}.bin'.format(args.save_dir, predix))

            print('Finish Depth {}'.format(predix))