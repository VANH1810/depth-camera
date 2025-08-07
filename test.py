#!/usr/bin/env python3
import os
import json
import cv2
import numpy as np
import pandas as pd
import argparse
import matplotlib
matplotlib.use('Agg')  # offscreen
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
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
        poses_data[int(frame_id) + 60] = {'pids':pids,'boxes':boxes, 'poses':poses}
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
    5) Inset real-time 3D trajectory view
    """
    def __init__(self, record_folder, pose_csv, output_video, inset_size=(200,200)):
        self.record_folder = record_folder
        # Khởi tạo Distance2Cam và Metadata
        self.distance = Distance2Cam(record_folder)
        data_file = os.path.join(record_folder, 'robot_data.jsonl')
        self.metadata = Metadata(data_file).metadata
        self.poses = parse_pose(pose_csv)

        # Dữ liệu trajectory: pid -> list of (xw, yw, zw)
        self.trajectory = {}

        # Nâng cao: chuẩn bị figure offscreen để vẽ inset
        self.fig = plt.figure(figsize=(2,2)); self.ax3d = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvas(self.fig)
        self.inset_w, self.inset_h = inset_size

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
        self.frame_width = width
        self.frame_height = height

    def run(self):
        frame_id = 1
        while True:
            ret, frame = self.vid.read()
            if not ret:
                break
            img = frame.copy()

            # Xử lý annotation và trajectory
            if frame_id in self.poses:
                for pid, box, pose in zip(
                    self.poses[frame_id]['pids'],
                    self.poses[frame_id]['boxes'],
                    self.poses[frame_id]['poses']
                ):
                    xmin, ymin, w_box, h_box, conf = box

                    if conf < 0.7: continue

                    x0, y0 = int(xmin), int(ymin)
                    x1, y1 = int(xmin + w_box), int(ymin + h_box)
                    # Vẽ bounding box
                    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    # Vẽ keypoints & limbs
                    pts = np.array(pose).reshape(-1, 3)
                    img = _draw_limbs(pts[:, :2], img)
                    img = _draw_pid(img, [x0, y0, x1, y1], pid)

                    # Tính 3D world
                    cx = int(xmin + w_box/2)
                    cy = int(ymin + h_box/2)
                    result = self.distance.get_distance_at_point(cx, cy, frame_id)
                    if result is None: continue
                    _, x3d, y3d, z3d = result

                    meta = self.metadata.get(frame_id)
                    if meta is None: continue

                    xw, yw, zw = positioncam2org(x3d, y3d, z3d, meta)

                    print("Frame", frame_id, "P", pid, "xw, yw, zw =", xw, yw, zw)
                    
                    self.trajectory.setdefault(pid, []).append((xw, yw, zw))
                    # Ghi nhãn toạ độ world
                    label = f"P{pid}: {xw:.2f},{yw:.2f},{zw:.2f}m"
                    cv2.putText(img, f"P{pid}: {xw:.2f},{yw:.2f},{zw:.2f}m", (x0, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)

            # Vẽ inset 3D realtime
            inset_img = self._render_inset()
            # Chèn vào góc trên bên phải
            x_off = self.frame_width - self.inset_w - 10
            y_off = 10
            img[y_off:y_off+self.inset_h, x_off:x_off+self.inset_w] = inset_img

            self.writer.write(img)
            frame_id += 1

        self.vid.release()
        self.writer.release()
        print(f"[INFO] Saved video with inset to {self.writer}")

    def _render_inset(self):
        self.ax3d.clear()
        # Gather all points
        if self.trajectory:
            all_pts = np.vstack([np.array(v) for v in self.trajectory.values()])
        else:
            all_pts = np.array([[0.,0.,0.]])
        mins = all_pts.min(axis=0); maxs = all_pts.max(axis=0)
        span = maxs - mins; eps = 1e-2
        span[span < eps] = eps
        mins = mins - span*0.1; maxs = maxs + span*0.1
        # Plot and label
        for pid, pts in self.trajectory.items():
            arr = np.array(pts)
            self.ax3d.plot(arr[:,0],arr[:,1],arr[:,2], label=f'P{pid}')
            x_e,y_e,z_e = arr[-1]
            self.ax3d.text(x_e, y_e, z_e, f'P{pid}', fontsize=6)
        # Set proper axes
        self.ax3d.set_xlim(mins[0], maxs[0]); self.ax3d.set_ylim(mins[1], maxs[1]); self.ax3d.set_zlim(mins[2], maxs[2])
        self.ax3d.set_box_aspect([1,1,1])
        self.ax3d.set_xlabel('X'); self.ax3d.set_ylabel('Y'); self.ax3d.set_zlabel('Z')
        self.ax3d.grid(True)
        # Draw using buffer_rgba
        self.canvas.draw()
        raw = self.canvas.buffer_rgba()
        buf = np.frombuffer(raw, dtype=np.uint8)
        h, w = map(int, self.fig.get_size_inches() * self.fig.get_dpi())
        buf = buf.reshape(h, w, 4)
        rgb = buf[..., :3]
        inset = cv2.resize(rgb, (self.inset_w, self.inset_h))
        return cv2.cvtColor(inset, cv2.COLOR_RGB2BGR)

def main():
    base_dir = os.path.dirname(__file__)
    record_folder = os.path.abspath(os.path.join(base_dir, 'recorded_data', 'recorded_data', 'recording_20250725_161358'))
    pose_csv      = os.path.abspath(os.path.join(base_dir, 'color.csv'))
    output        = os.path.abspath(os.path.join(base_dir, 'vis_output.mp4'))
    inset_w       = 200
    inset_h       = 200

    print("[DEBUG] Final record_folder =", record_folder)

    pipeline = PoseWorldPipeline(
        record_folder,
        pose_csv,
        output,
        inset_size=(inset_w, inset_h)
    )
    pipeline.run()


if __name__ == '__main__':
    main()
