import numpy as np
from mayavi import mlab
import torch


def is_point_in_box(point, box):
    """
    判断点是否在3D边界框内
    :param point: 点坐标 [x, y, z]
    :param box: 3D边界框 [x, y, z, l, w, h, ry]，其中ry是围绕y轴的旋转角
    :return: 是否在框内
    """
    x, y, z = point
    cx, cy, cz, l, w, h, ry = box

    # 边界框的本地坐标系到全局坐标系的转换
    # 1. 平移到原点
    x_local = x - cx
    y_local = y - cy
    z_local = z - cz

    # 2. 绕y轴旋转-ry角度
    cos_ry = np.cos(-ry)
    sin_ry = np.sin(-ry)
    x_rot = x_local * cos_ry - z_local * sin_ry
    z_rot = x_local * sin_ry + z_local * cos_ry

    # 3. 检查是否在边界框内
    half_l = l / 2
    half_w = w / 2
    half_h = h / 2

    if (-half_l <= x_rot <= half_l and
        -half_w <= y_local <= half_w and
        -half_h <= z_rot <= half_h):
        return True
    return False


def boxes_to_corners_3d(boxes3d):
    """
    将3D边界框转换为8个角点
    :param boxes3d: (N, 7)的边界框数组 [x, y, z, l, w, h, ry]
    :return: 角点坐标 (N, 8, 3)
    """
    boxes3d = np.array(boxes3d)
    N = boxes3d.shape[0]

    l = boxes3d[:, 3]
    w = boxes3d[:, 4]
    h = boxes3d[:, 5]
    ry = boxes3d[:, 6]

    # 计算边界框的8个角点（相对于中心）
    x_corners = np.array([-0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5])  # (8,)
    y_corners = np.array([-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5])  # (8,)
    z_corners = np.array([-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5])  # (8,)

    # 缩放角点 - 修正w和h的对应关系
    x_corners = x_corners * l[:, np.newaxis]  # (N, 8) - 长度沿x轴
    y_corners = y_corners * h[:, np.newaxis]  # (N, 8) - 高度沿y轴
    z_corners = z_corners * w[:, np.newaxis]  # (N, 8) - 宽度沿z轴

    # 旋转角点 - 使用与is_point_in_box函数一致的旋转方向
    cos_ry = np.cos(-ry)[:, np.newaxis]  # (N, 1)
    sin_ry = np.sin(-ry)[:, np.newaxis]  # (N, 1)
    x_corners_rot = x_corners * cos_ry - z_corners * sin_ry  # (N, 8)
    z_corners_rot = x_corners * sin_ry + z_corners * cos_ry  # (N, 8)

    # 平移到中心位置 - 调整y轴以匹配KITTI的底面中心定义
    x_corners_rot += boxes3d[:, 0, np.newaxis]  # (N, 8) - x坐标
    y_corners += boxes3d[:, 1, np.newaxis] - h[:, np.newaxis] / 2  # (N, 8) - y坐标（底面中心）
    z_corners_rot += boxes3d[:, 2, np.newaxis]  # (N, 8) - z坐标

    # 堆叠成(N, 8, 3)形状
    corners = np.stack([x_corners_rot, y_corners, z_corners_rot], axis=2)  # (N, 8, 3)
    return corners


def draw_corners3d(corners3d, color=(1, 0, 0), line_width=2, figure=None):
    """
    绘制3D边界框的角点连接线
    :param corners3d: (N, 8, 3)的角点数组
    :param color: 边界框颜色
    :param line_width: 线宽
    :param figure: 图形对象
    :return: 绘制的线条列表
    """
    if figure is None:
        figure = mlab.gcf()

    # 边界框的边连接关系
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]

    lines = []
    for i in range(corners3d.shape[0]):
        corners = corners3d[i]
        for j, k in edges:
            x = [corners[j, 0], corners[k, 0]]
            y = [corners[j, 1], corners[k, 1]]
            z = [corners[j, 2], corners[k, 2]]
            line = mlab.plot3d(x, y, z, color=color, tube_radius=None, line_width=line_width, figure=figure)
            lines.append(line)
    return lines


def visualize_dataset(points, gt_boxes, class_names=None, background_color=(0.0, 0.0, 0.0),
                      outside_points_color=(0.5, 0.5, 0.5), inside_points_color=(0, 0, 1),
                      box_color=(1, 0, 0), box_line_width=3, show_labels=True):
    """
    可视化数据集，显示点云和真实边界框，并区分框内点颜色
    :param points: 点云数据 (N, 3) [x, y, z]
    :param gt_boxes: 真实边界框 (M, 7) [x, y, z, l, w, h, ry]
    :param background_color: 背景颜色
    :param outside_points_color: 框外点颜色
    :param inside_points_color: 框内点颜色
    :param box_color: 边界框颜色
    :param box_line_width: 边界框线宽
    :return: 图形对象
    """
    # 转换为NumPy数组
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()

    # 创建图形
    figure = mlab.figure('dataset_visualization', bgcolor=background_color, size=(1000, 800))

    # 检查每个点是否在任何真实框内
    inside_indices = []
    outside_indices = []
    for i, point in enumerate(points):
        is_inside = False
        for box in gt_boxes:
            if is_point_in_box(point, box):
                is_inside = True
                break
        if is_inside:
            inside_indices.append(i)
        else:
            outside_indices.append(i)

    # 绘制框外点
    if len(outside_indices) > 0:
        mlab.points3d(
            points[outside_indices, 0], points[outside_indices, 1], points[outside_indices, 2],
            color=outside_points_color,  # 框外点颜色
            mode='point',
            scale_factor=0.05,
            figure=figure
        )

    # 绘制框内点
    if len(inside_indices) > 0:
        mlab.points3d(
            points[inside_indices, 0], points[inside_indices, 1], points[inside_indices, 2],
            color=inside_points_color,  # 框内点颜色
            mode='point',
            scale_factor=0.06,  # 框内点稍大一些，更明显
            figure=figure
        )

    # 绘制真实边界框和标签
    if gt_boxes.shape[0] > 0:
        corners3d = boxes_to_corners_3d(gt_boxes)
        draw_corners3d(corners3d, color=box_color, line_width=box_line_width, figure=figure)

        # 绘制标签
        if show_labels and class_names is not None:
            for i, box in enumerate(gt_boxes):
                cx, cy, cz, l, w, h, ry = box
                # 在边界框中心上方添加标签
                label_pos = (cx, cy, cz + h/2 + 0.2)
                if i < len(class_names):
                    label_text = class_names[i]
                else:
                    label_text = f'Object {i+1}'
                mlab.text3d(label_pos[0], label_pos[1], label_pos[2], label_text,
                            color=(1, 1, 1), scale=0.3, figure=figure)

    # 添加坐标系
    mlab.axes(xlabel='X', ylabel='Y', zlabel='Z', figure=figure)
    mlab.points3d(0, 0, 0, color=(1, 1, 1), scale_factor=0.1, figure=figure)  # 原点标记

    # 设置视角
    mlab.view(azimuth=-135, elevation=30, figure=figure)
    mlab.roll(0, figure=figure)

    return figure


def show_dataset_visualization(points, gt_boxes, class_names=None):
    """
    显示数据集可视化
    :param points: 点云数据 (N, 3) [x, y, z]
    :param gt_boxes: 真实边界框 (M, 7) [x, y, z, l, w, h, ry]
    """
    figure = visualize_dataset(points, gt_boxes, class_names=class_names)
    mlab.show()


# 示例使用代码
if __name__ == '__main__':
    # 生成随机点云数据
    np.random.seed(42)
    points = np.random.rand(1000, 3) * 10 - 5  # 在[-5, 5)范围内生成随机点

    # 生成随机真实边界框
    gt_boxes = np.array([
        [0, 0, 0, 2, 1, 1, 0],  # 中心在原点的边界框
        [3, 1, -2, 1.5, 0.8, 0.8, np.pi/4]  # 带旋转的边界框
    ])

    # 显示可视化
    show_dataset_visualization(points, gt_boxes)