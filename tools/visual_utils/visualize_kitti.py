import os
import numpy as np
import argparse
import time
import os
from pathlib import Path

from ptt.config import cfg, cfg_from_yaml_file
from ptt.datasets import KittiTrackingDataset
from dataset_visualizer import visualize_dataset, show_dataset_visualization


def load_kitti_velodyne(velodyne_path):
    """
    加载KITTI数据集的Velodyne点云数据
    :param velodyne_path: 点云文件路径
    :return: 点云数据 (N, 3) [x, y, z]
    """
    # 加载二进制点云数据
    points = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)
    # 只取前三个坐标 (x, y, z)
    return points[:, :3]


def load_kitti_labels(tracklet_list, frame_id, class_names=None):
    """
    加载KITTI数据集的标签数据
    :param tracklet_list: 标签列表
    :param frame_id: 帧ID，用于过滤特定帧的标签
    :param class_names: 要加载的类别名称列表
    :return: 真实边界框 (M, 7) [x, y, z, l, w, h, ry]
    """
    gt_boxes = []

    for tracklet in tracklet_list:
        # 检查帧ID是否匹配（仅比较数字部分）
        if int(tracklet['frame']) != int(frame_id):
            continue

        # 检查类别是否匹配
        if class_names is not None and tracklet['type'] not in class_names:
            continue

        # 提取边界框参数
        gt_boxes.append([
            tracklet['x'],
            tracklet['y'],
            tracklet['z'],
            tracklet['length'],
            tracklet['width'],
            tracklet['height'],
            tracklet['rotation_y']
        ])

    print(f"共加载 {len(gt_boxes)} 个边界框")
    return np.array(gt_boxes)


def visualize_kitti_data(cfg_file, folder_id, start_frame, end_frame, class_names=None):
    """
    可视化KITTI数据集的指定帧
    :param cfg_file: 配置文件路径
    :param folder_id: 文件夹ID（字符串格式，如'0000'）
    :param start_frame: 起始帧ID
    :param end_frame: 结束帧ID
    :param class_names: 要可视化的类别名称列表
    """
    # 加载配置文件
    cfg_from_yaml_file(cfg_file, cfg)

    # 创建KittiTrackingDataset实例
    dataset = KittiTrackingDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=class_names or cfg.CLASS_NAMES,
        training=True
    )

    # 获取指定场景的tracklet列表
    scene_id = int(folder_id)
    scenes = [scene_id]  # 将folder_id转换为单元素列表
    tracklet_list = dataset.get_tracklet_list(scenes)
    # 过滤出当前场景的所有帧标签
    scene_tracklets = []
    for tracklet in tracklet_list:
        if tracklet and 'scene' in tracklet[0] and str(tracklet[0]['scene']) == folder_id:
            scene_tracklets.extend(tracklet)
    # 将过滤后的标签设置给数据集
    dataset.per_sequence_anno = tracklet_list
    dataset.per_frame_anno = scene_tracklets

    # 循环加载并可视化连续帧
    for frame_num in range(int(start_frame), int(end_frame) + 1):
        # 格式化帧ID为6位数字，前面补零
        frame_id = f'{frame_num:06d}'
        print(f"\n可视化帧: {frame_id}")

        # 构建点云文件路径
        velodyne_path = os.path.join(cfg.DATA_CONFIG.DATA_PATH, 'training', 'velodyne', folder_id, f'{frame_id}.bin')

        # 检查点云文件是否存在
        if not os.path.exists(velodyne_path):
            print(f"点云文件不存在: {velodyne_path}")
            continue

        # 加载数据
        points = load_kitti_velodyne(velodyne_path)
        # 从过滤后的标签中加载当前帧的边界框
        gt_boxes = load_kitti_labels(scene_tracklets, frame_id, class_names or cfg.CLASS_NAMES)

        print(f"加载成功: {len(points)}个点, {len(gt_boxes)}个边界框")

        # 可视化
        show_dataset_visualization(points, gt_boxes, class_names or cfg.CLASS_NAMES)

        # 每帧之间暂停0.5秒，以便观察
        time.sleep(0.5)


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='KITTI数据集连续帧可视化工具')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/ptt.yaml',
                        help='配置文件路径 (默认: cfgs/kitti_models/ptt.yaml)')
    parser.add_argument('--folder_id', type=str, default='0000',
                        help='velodyne下的文件夹ID (默认: 0000)')
    parser.add_argument('--start_frame', type=int, default=0,
                        help='起始帧ID (默认: 0)')
    parser.add_argument('--end_frame', type=int, default=10,
                        help='结束帧ID (默认: 10)')
    parser.add_argument('--class_names', type=str, nargs='+', default=None,
                        help='要可视化的类别名称 (默认: 配置文件中定义的类别)')

    args = parser.parse_args()

    # 可视化连续帧
    visualize_kitti_data(args.cfg_file, args.folder_id, args.start_frame, args.end_frame, args.class_names)