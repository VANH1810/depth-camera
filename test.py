#!/usr/bin/env python3
import os
import json
import cv2
import numpy as np
import pandas as pd
import argparse
from libs.dist2camera import Distance2Cam
from libs.robotmetadata import Metadata
from libs.helper import _draw_limbs, _draw_pid


def parse_pose(csv_file):
    data_df = pd.read_csv(csv_file)
    frames = data_df.groupby('FrameID')
    poses_data = {}
    for frame_id in frames.groups:
        pids = []
        boxes = []
        poses = []
        group = frames.get_group(frame_id)
        for _, row in group.iterrows():
            pid = int(row['PID'])
            box = json.loads(row['Bbox'])
            pose = np.array(json.loads(row['Pose'])).reshape(-1,3)
            pids.append(pid)
            boxes.append(box)
            poses.append(pose)
        poses_data[int(frame_id)] = {'pids':pids,'boxes':boxes, 'poses':poses}
    return poses_data

def positioncam2org(x,y,z, meta_data):
    # position of robotdog in original coordination
    x_d = meta_data['metadata']['x']
    y_d = meta_data['metadata']['y']
    z_d = meta_data['metadata']['z']
    z_d = meta_data['metadata']['height'] + meta_data['metadata']['z']
    theta = meta_data['metadata']['yaw']
    
    # position of object in robotdog coordination
    x1 = x
    y1 = z
    z1 = y
    l = np.sqrt(x1*x1 + y1*y1)
    sin_alpha = y1/l
    cos_alpha = x1/l

    # position of object in original coordination
    x_o = x_d + l*(np.cos(theta)*cos_alpha +np.sin(theta)*sin_alpha)
    y_o = y_d + l*(np.sin(theta)*cos_alpha - np.cos(theta)*sin_alpha)
    z_o = z_d + z1
    
    return x_o, y_o, z_o

class PoseWorldPipeline:
    """
    Pipeline:
    1) Đọc poses từ CSV
    2) Lấy depth + pixel -> 3D camera
    3) Chuyển camera -> world
    4) Vẽ bbox, keypoints và toạ độ world lên video
    """
    def __init__(self, record_folder, pose_csv, output_video):
        self.record_folder = record_folder

        # Khởi tạo Distance2Cam và Metadata
        self.distance = Distance2Cam(record_folder)
        data_file = os.path.join(record_folder, 'robot_data.jsonl')
        self.metadata = Metadata(data_file).metadata
        self.poses = parse_pose(pose_csv)

        # Mở video gốc
        color_path = os.path.join(record_folder, 'color.avi')
        self.vid = cv2.VideoCapture(color_path)
        if not self.vid.isOpened():
            raise IOError(f"Cannot open video: {color_path}")

        width  = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = self.vid.get(cv2.CAP_PROP_FPS) or 30.0

        # Thiết lập writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    def run(self):
        frame_id = 1
        while True:
            ret, frame = self.vid.read()
            if not ret:
                break
            img = frame.copy()

            if frame_id in self.poses:
                for pid, box, pose in self.poses[frame_id]:
                    xmin, ymin, w_box, h_box, conf = box
                    x0, y0 = int(xmin), int(ymin)
                    x1, y1 = int(xmin + w_box), int(ymin + h_box)
                    # Vẽ bounding box
                    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    # Vẽ keypoints & limbs
                    pts = np.array(pose).reshape(-1, 3)
                    img = _draw_limbs(pts[:, :2], img)
                    img = _draw_pid(img, [x0, y0, x1, y1], pid)

                    # Tính pixel center lấy depth
                    cx = int(xmin + w_box/2)
                    cy = int(ymin + h_box/2)
                    result = self.distance.get_distance_at_point(cx, cy, frame_id)
                    if result is None:
                        continue
                    _, x3d, y3d, z3d = result
                    # Chuyển sang world
                    meta = self.metadata[frame_id]
                    xw, yw, zw = positioncam2org(x3d, y3d, z3d, meta)
                    # Ghi nhãn toạ độ world
                    label = f"P{pid}: {xw:.2f},{yw:.2f},{zw:.2f}m"
                    cv2.putText(img, label, (x0, y1 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            self.writer.write(img)
            frame_id += 1

        self.vid.release()
        self.writer.release()
        print(f"[INFO] Saved visualization to {self.writer}")


def main():
    parser = argparse.ArgumentParser(description='Pose→World Visualization')
    parser.add_argument('--record_folder', required=True)
    parser.add_argument('--pose_csv', required=True)
    parser.add_argument('--output', default='vis_output.mp4')
    args = parser.parse_args()

    pipeline = PoseWorldPipeline(args.record_folder, args.pose_csv, args.output)
    pipeline.run()


if __name__ == '__main__':
    main()
