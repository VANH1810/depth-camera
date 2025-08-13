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
from kalman_filter import KalmanFilter3D

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
    theta = np.pi - meta_data['metadata']['yaw']
    
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

def _line_intersection(p1, p2, p3, p4, eps=1e-6):
    """
    Giao điểm của 2 đường thẳng (p1-p2) và (p3-p4).
    p* = (x, y). Trả về (x, y) hoặc None nếu gần như song song / suy biến.
    """
    x1, y1 = p1; x2, y2 = p2
    x3, y3 = p3; x4, y4 = p4
    den = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if abs(den) < eps:
        return None
    # dùng công thức định thức
    det1 = (x1*y2 - y1*x2)
    det2 = (x3*y4 - y3*x4)
    px = (det1*(x3 - x4) - (x1 - x2)*det2) / den
    py = (det1*(y3 - y4) - (y1 - y2)*det2) / den
    return (float(px), float(py))

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
        self.writer = cv2.VideoWriter(output_video, fourcc, fps, (width * 2, height))
        self.frame_width = width
        self.frame_height = height

        # Nâng cao: chuẩn bị figure offscreen để vẽ inset
        dpi = 100
        self.fig = plt.figure(figsize=(width / dpi, height / dpi)); 
        self.ax3d = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvas(self.fig)
        self.inset_w, self.inset_h = inset_size

        self._L_SH, self._R_SH, self._L_HIP, self._R_HIP = 5, 6, 11, 12

        # --- Open depth video ---
        depth_path = os.path.join(record_folder, 'depth.avi')
        self.depth_vid = cv2.VideoCapture(depth_path)
        if not self.depth_vid.isOpened():
            print(f"[WARN] Cannot open depth video: {depth_path} (coverage check will be disabled)")
            self.depth_vid = None
            self.depth_width = None
            self.depth_height = None
            self.sx = self.sy = 1.0

        if self.depth_vid is not None:
            self.depth_width  = int(self.depth_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.depth_height = int(self.depth_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # scale từ toạ độ color -> depth
            self.sx = self.depth_width  / float(self.frame_width)
            self.sy = self.depth_height / float(self.frame_height)

    def _bbox_depth_coverage(self, depth_img, bbox_xyxy,
                            white_thr=245,     # >= white_thr coi là "trắng" → NO DEPTH
                            black_floor=5,     # <= floor coi là đen/noise (tuỳ dữ liệu)
                            inner_margin=0.10, # chỉ đo phần “ruột” bbox (cắt viền 10%)
                            min_ratio=0.10):
        x0, y0, x1, y1 = map(int, bbox_xyxy)
        h, w = depth_img.shape[:2]
        x0 = max(0, min(x0, w-1)); x1 = max(0, min(x1, w-1))
        y0 = max(0, min(y0, h-1)); y1 = max(0, min(y1, h-1))
        if x1 <= x0 or y1 <= y0: return 0.0, False

        roi = depth_img[y0:y1, x0:x1]
        if roi.ndim == 3: roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if roi.dtype != np.uint8:
            rmin, rmax = float(roi.min()), float(roi.max())
            roi = ((roi - rmin) / (max(rmax-rmin,1e-6)) * 255.0).astype(np.uint8)

        # chỉ tính phần trong để tránh viền dễ bị saturate trắng
        H, W = roi.shape[:2]
        dx = int(W*inner_margin); dy = int(H*inner_margin)
        roi = roi[dy:H-dy, dx:W-dx] if (W>2*dx and H>2*dy) else roi

        # valid nếu: không trắng, không quá đen
        valid = (roi < white_thr) & (roi > black_floor)

        # lọc nhiễu muối tiêu (điểm đen nhỏ trong biển trắng)
        valid = valid.astype(np.uint8) * 255
        k3 = np.ones((3,3), np.uint8)
        valid = cv2.morphologyEx(valid, cv2.MORPH_OPEN, k3)  # bỏ điểm lẻ
        valid = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, k3) # nối mảng xám liền
        ratio = float(valid.mean() / 255.0) if valid.size else 0.0
        return ratio, (ratio >= min_ratio)
    
    def _torso_point_uv(self, pose, s_thr=0.30):
        """
        Trả về (u, v) là giao của 2 đoạn:
        (RShoulder - LHip) và (LShoulder - RHip).
        KHÔNG fallback. Nếu thiếu KP hoặc không cắt nhau -> (None, None).
        """
        pts = np.array(pose).reshape(-1, 3)
        K = pts.shape[0]

        def ok(idx):
            return (0 <= idx < K and pts[idx, 2] >= s_thr
                    and np.isfinite(pts[idx, 0]) and np.isfinite(pts[idx, 1]))

        if not (ok(self._R_SH) and ok(self._L_HIP) and ok(self._L_SH) and ok(self._R_HIP)):
            return None, None

        p_RS = (float(pts[self._R_SH, 0]), float(pts[self._R_SH, 1]))
        p_LH = (float(pts[self._L_HIP, 0]), float(pts[self._L_HIP, 1]))
        p_LS = (float(pts[self._L_SH, 0]), float(pts[self._L_SH, 1]))
        p_RH = (float(pts[self._R_HIP, 0]), float(pts[self._R_HIP, 1]))

        inter = _line_intersection(p_RS, p_LH, p_LS, p_RH)
        if inter is None or not np.isfinite(inter[0]) or not np.isfinite(inter[1]):
            return None, None

        return int(round(inter[0])), int(round(inter[1]))
    
    def _robust_depth_at(self, u, v, frame_id, half_win=1):
        """
        Lấy depth/3D robust quanh (u,v) bằng median trong ô vuông (2*half_win+1)^2.
        Trả về (x3d,y3d,z3d) hoặc None nếu không có điểm hợp lệ.
        """
        val = []
        for dv in range(-half_win, half_win+1):
            for du in range(-half_win, half_win+1):
                uu = int(np.clip(u+du, 0, self.frame_width-1))
                vv = int(np.clip(v+dv, 0, self.frame_height-1))
                res = self.distance.get_distance_at_point(uu, vv, frame_id)
                if res is None: 
                    continue
                _, x3d, y3d, z3d = res
                if z3d is None or not np.isfinite(z3d) or z3d <= 0:
                    continue
                val.append((x3d, y3d, z3d))
        if not val:
            return None
        arr = np.array(val, dtype=float)
        med = np.median(arr, axis=0)
        return float(med[0]), float(med[1]), float(med[2])
    
    def run(self):
        frame_id = 1
        while True:
            ret, frame = self.vid.read()
            if not ret:
                break
            img = frame.copy()

            # Đọc depth frame tương ứng (nếu có)
            depth_frame = None
            if self.depth_vid is not None:
                dret, dframe = self.depth_vid.read()
                if dret:
                    # dùng grayscale để đo coverage
                    depth_frame = cv2.cvtColor(dframe, cv2.COLOR_BGR2GRAY) if dframe.ndim==3 else dframe

            # Xử lý annotation và trajectory
            if frame_id in self.poses:
                for pid, box, pose in zip(
                    self.poses[frame_id]['pids'],
                    self.poses[frame_id]['boxes'],
                    self.poses[frame_id]['poses']
                ):
                    xmin, ymin, w_box, h_box, conf = box

                    if conf < 0.7: 
                        print(f"[SKIP] frame {frame_id} pid {pid}: low conf={conf:.2f}")
                        continue

                    x0, y0 = int(xmin), int(ymin)
                    x1, y1 = int(xmin + w_box), int(ymin + h_box)

                    # ==== CHECK DEPTH COVERAGE ====
                    depth_ratio, depth_ok = (0.0, True)
                    if depth_frame is not None:
                        # scale bbox (color -> depth)
                        dx0 = int(x0 * self.sx); dy0 = int(y0 * self.sy)
                        dx1 = int(x1 * self.sx); dy1 = int(y1 * self.sy)

                        depth_ratio, depth_ok = self._bbox_depth_coverage(
                            depth_frame, (dx0, dy0, dx1, dy1),
                            white_thr=245,      # trắng = NO DEPTH
                            black_floor=5,      # bỏ nhiễu quá đen nếu có
                            inner_margin=0.10,  # cắt viền 10%
                            min_ratio=0.50      # yêu cầu >=50% vùng xám
                        )
                        print(f"[COVERAGE] frame {frame_id} pid {pid}: {depth_ratio*100:.1f}% (ok={depth_ok})")


                    # Vẽ bounding box
                    # cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)

                    # Vẽ bbox theo coverage + in % coverage
                    col = (0,255,0) if depth_ok else (0,0,255)
                    cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
                    if depth_frame is not None:
                        cv2.putText(img, f"depth={depth_ratio*100:.1f}%", (x0, y0-6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
                    else:
                        cv2.putText(img, "depth:N/A", (x0, y0-6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
                        
                    # Vẽ keypoints & limbs
                    pts = np.array(pose).reshape(-1, 3)
                    img = _draw_limbs(pts[:, :2], img)
                    img = _draw_pid(img, [x0, y0, x1, y1], pid)

                    if depth_frame is not None and not depth_ok:
                        print(f"[WARN] frame {frame_id} pid {pid}: low depth coverage -> skip 3D")
                        continue

                    # Tính 3D world
                    # cx = int(xmin + w_box/2)
                    # cy = int(ymin + h_box/2)

                    # result = self.distance.get_distance_at_point(cx, cy, frame_id)
                    # if result is None: 
                    #     print(f"[SKIP] frame {frame_id} pid {pid}: no depth at ({cx},{cy}) for depth-frame={frame_id}")
                    #     continue
                    # _, x3d, y3d, z3d = result

                    # ==== CHỌN ĐIỂM BỤNG (u,v) — STRICT ====
                    u_vis, v_vis = self._torso_point_uv(pose, s_thr=0.30)
                    if u_vis is None:
                        print(f"[SKIP] frame {frame_id} pid {pid}: no torso intersection")
                        continue

                    # ==== DEPTH/3D tại điểm bụng (median lọc nhiễu) ====
                    depth_3d = self._robust_depth_at(u_vis, v_vis, frame_id, half_win=1)
                    if depth_3d is None:
                        print(f"[SKIP] frame {frame_id} pid {pid}: no depth at torso point")
                        continue
                    x3d, y3d, z3d = depth_3d

                    # Vẽ marker tại điểm bụng
                    cv2.circle(img, (int(u_vis), int(v_vis)), 3, (255, 0, 255), -1)

                    meta = self.metadata.get(frame_id)
                    if meta is None: 
                        print(f"[SKIP] frame {frame_id} pid {pid}: no metadata for frame_id={frame_id}")
                        continue

                    xw, yw, zw = positioncam2org(x3d, y3d, z3d, meta)

                    print("Frame", frame_id, "P", pid, "xw, yw, zw =", xw, yw, zw)
                    
                    self.trajectory.setdefault(pid, []).append((xw, yw, zw))

                    # Ghi nhãn toạ độ world
                    label_world = f"P{pid}: {xw:.2f},{yw:.2f},{zw:.2f}m"
                    y_txt1 = min(y1 + 15, self.frame_height - 10)
                    cv2.putText(img, label_world, (x0, y_txt1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)

                    label_cam = f"Cam: {x3d:.2f},{y3d:.2f},{z3d:.2f}m"
                    y_txt2 = min(y1 + 30, self.frame_height - 5)  # thấp hơn 15px
                    cv2.putText(img, label_cam, (x0, y_txt2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            # Vẽ inset 3D realtime
            traj_img = self._render_full()
            # Combine side by side
            combined = np.hstack([img, traj_img])
            self.writer.write(combined)
            frame_id += 1

        self.vid.release()
        self.writer.release()
        print(f"[INFO] Saved side-by-side video to {self.writer}")

    def _render_full(self):
        self.ax3d.clear()
        pts_list = list(self.trajectory.values())
        if pts_list:
            all_pts = np.vstack([np.array(v) for v in pts_list])
        else:
            all_pts = np.zeros((1,3))
        mins, maxs = all_pts.min(0), all_pts.max(0)
        span = maxs - mins
        span[span==0] = 1e-2
        mins -= span*0.1; maxs += span*0.1
        for pid, pts in self.trajectory.items():
            arr = np.array(pts)
            self.ax3d.plot(arr[:,0],arr[:,1],arr[:,2],label=f'P{pid}')
            x_e,y_e,z_e = arr[-1]
            self.ax3d.text(x_e,y_e,z_e,f'P{pid}',fontsize=6)
        self.ax3d.set_xlim(mins[0],maxs[0]); 
        self.ax3d.set_ylim(mins[1],maxs[1]); 
        self.ax3d.set_zlim(mins[2],maxs[2])
        self.ax3d.set_xlabel('X'); 
        self.ax3d.set_ylabel('Y'); 
        self.ax3d.set_zlabel('Z'); 
        self.ax3d.grid(True)
        
        # use actual canvas size
        width,height = self.canvas.get_width_height()
        self.canvas.draw()
        buf = np.frombuffer(self.canvas.buffer_rgba(),dtype=np.uint8).reshape(height,width,4)
        rgb = buf[...,:3]
        traj = cv2.resize(rgb,(self.frame_width,self.frame_height))
        return cv2.cvtColor(traj, cv2.COLOR_RGB2BGR)

    def export_keypoints_world_csv(self, out_csv, kp_score_thr=0.25, valid_only=False):
        """
        Xuất CSV chứa toạ độ 3D (camera, world) của từng keypoint trên từng frame.
        - out_csv: đường dẫn file CSV đầu ra
        - kp_score_thr: ngưỡng score KP (bỏ qua KP có score < ngưỡng)
        - valid_only: True -> chỉ ghi những dòng có đủ depth/meta (ok=1)
        """
        rows = []
        # Duyệt tất cả frame có pose
        for f_id in sorted(self.poses.keys()):
            meta = self.metadata.get(f_id)
            # Duyệt qua từng person
            for pid, box, pose in zip(
                self.poses[f_id]['pids'],
                self.poses[f_id]['boxes'],
                self.poses[f_id]['poses']
            ):
                pts = np.array(pose).reshape(-1, 3)  # (K, 3) = (u, v, score)
                # Duyệt qua từng keypoint
                for k, (u, v, s) in enumerate(pts):
                    rec = {
                        'FrameID': int(f_id),
                        'PID': int(pid),
                        'KP': int(k),
                        'u': float(u),
                        'v': float(v),
                        'score': float(s),
                        'ok': 0,
                        'x_cam': np.nan, 'y_cam': np.nan, 'z_cam': np.nan,
                        'x_world': np.nan, 'y_world': np.nan, 'z_world': np.nan
                    }

                    # Bỏ qua theo score
                    if s < kp_score_thr or meta is None:
                        if not valid_only:
                            rows.append(rec)
                        continue

                    # Lấy depth -> 3D camera tại (u,v)
                    res = self.distance.get_distance_at_point(int(u), int(v), int(f_id))
                    if res is None:
                        if not valid_only:
                            rows.append(rec)
                        continue

                    _, x3d, y3d, z3d = res
                    # Camera -> World
                    xw, yw, zw = positioncam2org(x3d, y3d, z3d, meta)

                    rec.update({
                        'ok': 1,
                        'x_cam': float(x3d), 'y_cam': float(y3d), 'z_cam': float(z3d),
                        'x_world': float(xw), 'y_world': float(yw), 'z_world': float(zw),
                    })
                    rows.append(rec)

        df = pd.DataFrame(rows)
        if valid_only:
            df = df[df['ok'] == 1].reset_index(drop=True)
        df.to_csv(out_csv, index=False)
        print(f"[INFO] Saved keypoint world coordinates CSV to: {out_csv} (rows={len(df)})")

def main():
    base_dir = os.path.dirname(__file__)
    base_dir = os.path.dirname(__file__)
    record_folder = os.path.abspath(os.path.join(base_dir, 'recorded_data', 'recording_19700105_045503'))
    pose_csv      = os.path.abspath(os.path.join(base_dir, 'recorded_data', 'recording_19700105_045503', 'color.csv'))
    output        = os.path.abspath(os.path.join(base_dir, 'recorded_data', 'recording_19700105_045503', 'recording_19700105_045503.mp4'))
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
    # kp_csv = os.path.join(base_dir, 'recording_19700105_045503')
    # pipeline.export_keypoints_world_csv(kp_csv, kp_score_thr=0.25, valid_only=False)

if __name__ == '__main__':
    main()
