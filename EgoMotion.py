#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import cv2
import os
import time
import json

# ==========================
# Data structures
# ==========================
@dataclass
class Event:
    x: int
    y: int
    ts: float   # seconds (float)
    p: int = 1  # polarity (unused here)


# ==========================
# Helper math
# ==========================

def skew(v: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix of a 3D vector v."""
    x, y, z = v
    return np.array([[0, -z, y],
                     [z, 0, -x],
                     [-y, x, 0]], dtype=np.float32)

def rodrigues_from_rotvec(rvec: np.ndarray) -> np.ndarray:
    """Return 3x3 rotation matrix from a rotation vector using cv2.Rodrigues."""
    R, _ = cv2.Rodrigues(rvec.astype(np.float32))
    return R.astype(np.float32)

def Rz(yaw: float) -> np.ndarray:
    """Rotation around world Z (up) by yaw radians."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float32)

def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4x4 homogeneous transform from R(3x3) and t(3,)."""
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

# ==========================
# Camera params helpers
# ==========================

def K_from_color_intrinsics(fx: float, fy: float, ppx: float, ppy: float) -> np.ndarray:
    """Assemble 3x3 pinhole intrinsic matrix from fx,fy,ppx,ppy."""
    return np.array([[fx, 0.0, ppx],
                     [0.0, fy, ppy],
                     [0.0, 0.0, 1.0]], dtype=np.float32)

def build_from_camera_params(camera_params: dict, image_size: Tuple[int, int]):
    """Extract K and depth scale from your camera_params dictionary."""
    depth_scale = float(camera_params.get('depth_scale', 1.0))
    intr = camera_params.get('color_intrinsics', {})
    fx = float(intr.get('fx'))
    fy = float(intr.get('fy'))
    ppx = float(intr.get('ppx'))
    ppy = float(intr.get('ppy'))
    K = K_from_color_intrinsics(fx, fy, ppx, ppy)
    return K.astype(np.float32), depth_scale

# ==========================
# Motion Compensation (Python port)
# ==========================
class MotCompPy:
    """
    Ego-motion compensation tailored to your metadata layout.

    Assumptions
    -----------
    - Time unit: seconds.
    - Linear velocities v_world = (vx, vy, vz) are in WORLD frame (m/s).
    - Angular velocity omega_body = (wx, wy, wz) is in BODY frame (rad/s).
      If only vyaw (yaw rate) is available, we map it to omega_body=(0,0,vyaw).
    - Depth image: uint16. self.depth_scale converts raw to meters.
    - Intrinsics K are in pixels. Image size = (H, W).

    Notes
    -----
    - Rotational compensation warps events *back to a common reference time*
      (here we use the end of the window t_ref = last event ts).
    - Translational compensation uses per-pixel delta-time (Δt = t_ref - t_event)
      and original depth to perform a plane-induced warp.
    """

    def __init__(self,
                 K: np.ndarray,
                 image_size: Tuple[int, int],  # (H, W)
                 depth_scale: float = 1.0,
                 depth_min_m: float = 0.3,
                 depth_max_m: float = 10.0):
        # Intrinsics and image size
        self.K = K.astype(np.float32)
        self.K_inv = np.linalg.inv(self.K).astype(np.float32)
        self.H, self.W = image_size

        # Depth scaling and valid range
        self.depth_scale = float(depth_scale)  # meters per depth unit
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)

        # Depth convention
        # If the sensor stores Z along the optical axis, set False (default).
        # If it stores range along the ray, set True to convert range->Z.
        self.depth_is_range = False

        # Map BODY gyro to CAMERA axes for rotation warp
        self.map_body_to_cam_rotation = True

        # Optional simple-translation mode (camera-frame velocity already given)
        self.simple_trans_no_frames = False
        self.v_cam = np.zeros(3, dtype=np.float32)

        # Buffers
        self.events: List[Event] = []
        self.depth_img: Optional[np.ndarray] = None

        # State (world pose, velocities)
        self.pos_xy = np.zeros(2, dtype=np.float32)    # world (x, y)
        self.yaw = 0.0                                 # world yaw (rad)
        self.v_world = np.zeros(3, dtype=np.float32)   # (vx, vy, vz) in world
        self.omega_body = np.zeros(3, dtype=np.float32) # (wx, wy, wz) in body

        # Precomputed transforms
        self.cam2body = self._compute_cam2body()       # camera -> body
        self.fc2world = np.eye(4, dtype=np.float32)    # body (FC) -> world, updated each frame

        # Outputs (per-pixel)
        self.time_img = None     # float32 HxW: stores Δt = (t_ref - t_event) mean per pixel
        self.event_counts = None # int32   HxW: counts per pixel

    # ---------- Metadata / IO ----------
    def set_metadata(self,
                     x: float, y: float,
                     yaw: float,
                     vx: float, vy: float,
                     vz: float = 0.0,
                     vyaw: Optional[float] = None,
                     height_static: Optional[float] = None,
                     z_current: Optional[float] = None):
        """Set the current robot/world state. This also updates body->world transform."""
        self.pos_xy[:] = (x, y)
        self.yaw = float(yaw)
        self.v_world[:] = (vx, vy, vz)
        if vyaw is not None:
            self.omega_body[:] = (0.0, 0.0, float(vyaw))
        # body (FC) -> world transform at reference frame
        self.fc2world = make_T(Rz(self.yaw), np.array([x, y, 0.0], np.float32))

    def set_state(self, x, y, yaw, vx, vy, vz, vyaw=None, height_static=None, z_current=None):
        """Alias to be drop-in compatible with your main loop."""
        self.set_metadata(x, y, yaw, vx, vy, vz, vyaw, height_static, z_current)

    def set_imu(self, wx: float, wy: float, wz: float):
        """Optional: set full angular velocity in BODY frame (rad/s)."""
        self.omega_body[:] = (wx, wy, wz)

    def load_events(self, events: List[Event]):
        self.events = events

    def load_depth(self, depth_img_u16: np.ndarray):
        """Load a depth frame (uint16) aligned to color resolution (H, W)."""
        assert depth_img_u16.dtype == np.uint16, "Depth image must be uint16"
        assert depth_img_u16.shape[:2] == (self.H, self.W), "Depth size mismatch"
        self.depth_img = depth_img_u16

    # ---------- Core pipeline ----------
    def run(self, do_rotation: bool = True, do_translation: bool = False):
        """Execute rotational (and optionally translational) compensation."""
        self._clear_outputs()
        if not self.events:
            return

        if do_rotation:
            self._rotational_compensate()
        else:
            self._accumulate_no_compensation()

        if do_translation:
            assert self.depth_img is not None, "Depth image required for translational compensation"
            self._translational_compensate()

    # ---------- Stage 0: setup ----------
    def _clear_outputs(self):
        self.time_img = np.zeros((self.H, self.W), dtype=np.float32)
        self.event_counts = np.zeros((self.H, self.W), dtype=np.int32)

    def _compute_cam2body(self) -> np.ndarray:
        """
        Return camera->body 4x4 transform.

        body: x->front, y->right, z->up
        cam : z->front, x->right, y->down

        This mapping is equivalent to rotating around Y by -90° followed by X by +90°.
        """
        Ry = cv2.Rodrigues(np.array([0, -0.5*np.pi, 0], dtype=np.float32))[0]
        Rx = cv2.Rodrigues(np.array([0.5*np.pi, 0, 0], dtype=np.float32))[0]
        R = Rx @ Ry
        return make_T(R, np.zeros(3, dtype=np.float32))

    # ---------- Stage A: no compensation (baseline accumulation) ----------
    def _accumulate_no_compensation(self):
        """Accumulate timestamps without any warp (for debugging/baseline)."""
        t0 = self.events[0].ts
        for e in self.events:
            dt = float(e.ts - t0)
            x, y = int(e.x), int(e.y)
            if 0 <= x < self.W and 0 <= y < self.H:
                c = self.event_counts[y, x] + 1
                self.time_img[y, x] += (dt - self.time_img[y, x]) / c
                self.event_counts[y, x] = c

    # ---------- Stage B: rotational compensation ----------
    def _rotational_compensate(self):
        """
        Warp events back to a common reference time t_ref (end of window).
        Pixel mapping: x_ref ≈ K · R_cam(ω * Δt) · K^-1 · x_event, with Δt = t_ref - t_event.
        """
        if not self.events:
            return

        t_ref = self.events[-1].ts  # reference: end of window
        prev_ms_mark = None
        rot_K = np.eye(3, dtype=np.float32)
        ev = np.ones((3, 1), np.float32)

        R_cb = self.cam2body[:3, :3]  # camera->body rotation

        for e in self.events:
            dt = float(t_ref - e.ts)          # Δt >= 0
            cur_ms_mark = int(dt * 1000.0)    # update homography every 1 ms

            if cur_ms_mark != prev_ms_mark:
                prev_ms_mark = cur_ms_mark
                rotvec_body = self.omega_body * dt  # body-frame rotation vector * Δt
                if self.map_body_to_cam_rotation:
                    # Map rotation vector from BODY to CAMERA axes
                    rotvec_cam = (R_cb.T @ rotvec_body).astype(np.float32)
                else:
                    rotvec_cam = rotvec_body.astype(np.float32)
                R = rodrigues_from_rotvec(rotvec_cam)      # R_cam(Δt)
                rot_K = (self.K @ R @ self.K_inv).astype(np.float32)  # NO transpose

            ev[0, 0], ev[1, 0] = e.x, e.y
            warped = rot_K @ ev
            if warped[2, 0] <= 0:
                continue
            ix = int(warped[0, 0] / warped[2, 0])
            iy = int(warped[1, 0] / warped[2, 0])

            if 0 <= ix < self.W and 0 <= iy < self.H:
                c = self.event_counts[iy, ix] + 1
                # Store mean Δt at the reference pixel (used by translation stage)
                self.time_img[iy, ix] += (dt - self.time_img[iy, ix]) / c
                self.event_counts[iy, ix] = c

    # ---------- Stage C: translational compensation ----------
    def _translational_compensate(self):
        """
        Compensate translation by shifting 3D points in camera@t_ref frame.

        We construct per-millisecond translations: for Δt = i ms, camera motion in WORLD
        is d_world = v_world * Δt. Convert to camera@t_ref via R_cw, then apply the
        inverse translation (-d_cam) to points before reprojecting with original Z.
        """
        max_dt = float(self.time_img.max())
        max_ms = int(max_dt * 1000.0) + 1

        # Rotation: camera->world at reference (t_ref) is (fc2world @ cam2body).
        # So world->camera is its transpose.
        R_cw = (self.fc2world[:3, :3] @ self.cam2body[:3, :3]).T  # world -> cam@t_ref

        # Precompute per-ms transforms
        trans_vec = [np.eye(4, dtype=np.float32) for _ in range(max_ms)]
        for i in range(max_ms):
            dt = i * 1e-3
            if self.simple_trans_no_frames:
                # Already have camera-frame velocity
                d_cam = (self.v_cam * dt).astype(np.float32)
            else:
                # Translate world motion into camera@t_ref frame
                d_world = (self.v_world * dt).astype(np.float32)  # camera moves +d_world
                d_cam = (R_cw @ d_world).astype(np.float32)       # represent in cam frame

            # Apply inverse translation to compensate ego-motion
            T = make_T(np.eye(3, dtype=np.float32), -d_cam)
            trans_vec[i] = T

        new_counts = np.zeros_like(self.event_counts, dtype=np.int32)
        new_time   = np.zeros_like(self.time_img, dtype=np.float32)

        for y in range(self.H):
            for x in range(self.W):
                c  = self.event_counts[y, x]
                dt = self.time_img[y, x]  # Δt for this pixel
                if dt <= 0.001 or c <= 0:
                    continue
                if not self._depth_in_range(x, y):
                    continue

                Z = self._read_depth_Z(x, y)

                # Unproject to 3D using the original Z
                ray = (self.K_inv @ np.array([[x], [y], [1]], np.float32)).reshape(3)
                P  = ray * Z
                ms = min(int(dt * 1000.0), max_ms - 1)

                # Apply inverse translation (compensation)
                Pp = (trans_vec[ms] @ np.array([P[0], P[1], P[2], 1.0], np.float32))[:3]

                # Plane-induced warp: reproject using original Z as denominator
                pixp = (self.K @ Pp.reshape(3, 1)) * (1.0 / max(Z, 1e-6))
                x2, y2 = int(pixp[0, 0]), int(pixp[1, 0])

                if 0 <= x2 < self.W and 0 <= y2 < self.H:
                    c2 = new_counts[y2, x2] + 1
                    new_time[y2, x2] += (dt - new_time[y2, x2]) / c2
                    new_counts[y2, x2] = c2

        self.time_img = new_time
        self.event_counts = new_counts

    # ---------- Utilities ----------
    def _depth_in_range(self, x: int, y: int) -> bool:
        d_raw = float(self.depth_img[y, x])
        range_m = self.depth_scale * d_raw
        return (self.depth_min_m <= range_m <= self.depth_max_m)

    def _read_depth_Z(self, x: int, y: int) -> float:
        """
        Convert raw depth to optical-axis Z (meters).
        If sensor stores Z along the optical axis, return scaled value directly.
        If sensor stores range along the viewing ray, convert range -> Z.
        """
        d_m = self.depth_scale * float(self.depth_img[y, x])
        if not self.depth_is_range:
            return d_m  # already Z along optical axis

        # range -> Z conversion
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        denom = 1.0 + ((x - cx) ** 2) / (fx ** 2) + ((y - cy) ** 2) / (fy ** 2)
        Z = np.sqrt((d_m * d_m) / max(denom, 1e-8))
        return float(Z)

    def get_visualization(self) -> np.ndarray:
        """
        Visualize time_img as a heatmap (JET). time_img stores Δt (seconds) mean per pixel.
        """
        m = cv2.normalize(self.time_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(m, cv2.COLORMAP_JET)


# ==========================
# Video + Metadata integration runner
# ==========================

def _depth_for_frame(distance, frame_idx: int):
    """
    Try common methods on your Distance2Cam object to fetch per-frame depth (uint16).
    Returns None if unavailable.
    """
    for m in ('get_depth_frame', 'get_depth_at', 'get_depth_frame_by_index', 'depth_at'):
        if hasattr(distance, m):
            arr = getattr(distance, m)(frame_idx)
            if isinstance(arr, np.ndarray) and arr.dtype == np.uint16:
                return arr
    # Fallback: static depth fields if present
    if hasattr(distance, 'depth') and isinstance(distance.depth, np.ndarray):
        return distance.depth
    if hasattr(distance, 'depth_img') and isinstance(distance.depth_img, np.ndarray):
        return distance.depth_img
    return None

def _read_metadata_jsonl(path: str):
    """Optional helper to read JSONL lines (not used if you already have Metadata)."""
    recs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                recs.append(obj)
            except Exception:
                continue
    return recs

def overlay_heatmap(frame_bgr: np.ndarray, heatmap_bgr: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Alpha-blend the heatmap on top of the RGB frame."""
    heatmap_bgr = cv2.resize(heatmap_bgr, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(frame_bgr, 1.0, heatmap_bgr, alpha, 0)

def _extract_status_from_record(rec: dict) -> dict:
    """
    Extract a 'Status-like' dict (x,y,z,yaw,vx,vy,vz,vyaw,height) from various nested schemas.
    """
    if not isinstance(rec, dict):
        return {}
    if 'robot_data' in rec:
        rbd = rec.get('robot_data') or {}
        if isinstance(rbd, dict):
            if 'Robot1' in rbd and isinstance(rbd['Robot1'], dict):
                st = rbd['Robot1'].get('Status', {})
                if isinstance(st, dict):
                    return st
            if 'Status' in rbd and isinstance(rbd['Status'], dict):
                return rbd['Status']
    if 'vx' in rec or 'vyaw' in rec or 'yaw' in rec:
        return rec
    for k, v in rec.items():
        if isinstance(v, dict):
            st = _extract_status_from_record(v)
            if st:
                return st
    return {}

def _state_from_metadata_infos(metadata_infos, frame_idx: int, t_s: float):
    """
    Return a dict with numeric fields {x,y,z,yaw,vx,vy,vz,vyaw,height} for the given frame.
    - Tries exact match by 'metadata.video_frame_number'.
    - Otherwise picks nearest by 'receive_timestamp' relative to the first record.
    """
    if isinstance(metadata_infos, dict):
        if frame_idx in metadata_infos:
            rec = metadata_infos[frame_idx]
        elif str(frame_idx) in metadata_infos:
            rec = metadata_infos[str(frame_idx)]
        else:
            rec = metadata_infos
    elif isinstance(metadata_infos, list) and metadata_infos:
        matched = None
        for r in metadata_infos:
            md = r.get('metadata', {}) if isinstance(r, dict) else {}
            vfn = md.get('video_frame_number', None)
            try:
                if vfn is not None and int(vfn) == int(frame_idx):
                    matched = r
                    break
            except Exception:
                pass
        if matched is None:
            # choose nearest by timestamp
            base = None
            best_err = 1e18
            best = None
            for r in metadata_infos:
                md = r.get('metadata', {}) if isinstance(r, dict) else {}
                rt = md.get('receive_timestamp', None)
                if rt is None:
                    continue
                if base is None:
                    base = float(rt)
                tt = float(rt) - base  # relative seconds
                err = abs(tt - t_s)
                if err < best_err:
                    best_err = err
                    best = r
            rec = best if best is not None else metadata_infos[0]
        else:
            rec = matched
    else:
        rec = {}

    st = _extract_status_from_record(rec)

    def g(k, default=0.0):
        try:
            return float(st.get(k, default))
        except Exception:
            return float(default)

    return {
        'x': g('x', 0.0),
        'y': g('y', 0.0),
        'z': g('z', 0.0),
        'yaw': g('yaw', 0.0),
        'vx': g('vx', 0.0),
        'vy': g('vy', 0.0),
        'vz': g('vz', 0.0),
        'vyaw': g('vyaw', 0.0),
        'height': g('height', 0.0),
    }

def parse_pose(csv_file):
    """Utility to parse your pose CSV (kept from your original code)."""
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
            pose = np.array(json.loads(row['Pose'])).reshape(-1, 3)
            pids.append(pid)
            boxes.append(box)
            poses.append(pose)
        poses_data[int(frame_id)] = {'pids': pids, 'boxes': boxes, 'poses': poses}
    return poses_data

# ============ Metrics (paper-like) ============

def denoise_time(T: np.ndarray, C: np.ndarray, min_counts=3, ksize=5) -> np.ndarray:
    """
    Median filter the time map (stored in seconds) where counts >= min_counts.
    Returns a map in milliseconds (float32).
    """
    mask = (C >= min_counts)
    if not mask.any():
        return np.zeros_like(T, np.float32)
    Td = T.copy()
    Td[~mask] = 0.0
    Td = cv2.medianBlur((Td * 1000.0).astype(np.float32), ksize)  # ms
    Td[~mask] = 0.0
    return Td

def image_mean_var(Td: np.ndarray, C: np.ndarray, min_counts=3):
    """Mean and variance of Td over valid pixels (counts >= min_counts)."""
    m = Td[C >= min_counts].astype(np.float32)
    return (float(m.mean()), float(m.var())) if m.size else (0.0, 0.0)

def relative_contrast(Td: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]):
    """
    Relative Contrast: robustness-focused version using 99th percentile.
    RC = P99(M) / P99(B), where M is the bbox region and B is its complement.
    """
    if bbox is None:
        return None
    x, y, w, h = map(int, bbox)
    H, W = Td.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    M = Td[y0:y1, x0:x1]
    if M.size == 0:
        return 0.0

    # Background as the complement of M
    B_parts = []
    if y0 > 0:        B_parts.append(Td[:y0, :].ravel())
    if y1 < H:        B_parts.append(Td[y1:, :].ravel())
    if x0 > 0:        B_parts.append(Td[y0:y1, :x0].ravel())
    if x1 < W:        B_parts.append(Td[y0:y1, x1:].ravel())
    if not B_parts:
        return 0.0
    B = np.concatenate(B_parts, axis=0)

    Mq = float(np.percentile(M, 99))
    Bq = float(np.percentile(B, 99))
    return float(Mq / (Bq + 1e-6))

def _expand_clip_xywh(x, y, w, h, pad, W, H):
    """Expand bbox by pad pixels and clip to image size."""
    x = int(max(0, x - pad))
    y = int(max(0, y - pad))
    w = int(min(W - x, w + 2 * pad))
    h = int(min(H - y, h + 2 * pad))
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)

def _as_xywh_candidates(box, W, H):
    """
    Build both interpretations (xywh) and (x0y0x1y1)->xywh if possible.
    Returns list of valid (x,y,w,h) candidates clipped to image.
    """
    cands = []

    # Normalize input to list of 4 numbers
    if isinstance(box, dict):
        # try common keys
        if all(k in box for k in ("x","y","w","h")):
            bx = [box["x"], box["y"], box["w"], box["h"]]
        elif all(k in box for k in ("x0","y0","x1","y1")):
            bx = [box["x0"], box["y0"], box["x1"], box["y1"]]
        else:
            bx = list(box.values())[:4]
    else:
        bx = list(box)[:4]

    if len(bx) != 4:
        return cands

    b0, b1, b2, b3 = [float(v) for v in bx]

    # Interpretation A: (x, y, w, h)
    x, y, w, h = b0, b1, b2, b3
    if w > 0 and h > 0:
        x0 = max(0.0, x); y0 = max(0.0, y)
        x1 = min(float(W), x + w); y1 = min(float(H), y + h)
        ww = x1 - x0; hh = y1 - y0
        if ww > 0 and hh > 0:
            cands.append( (int(x0), int(y0), int(ww), int(hh)) )

    # Interpretation B: (x0, y0, x1, y1)
    x0, y0, x1, y1 = b0, b1, b2, b3
    if x1 > x0 and y1 > y0:
        x0c = max(0.0, x0); y0c = max(0.0, y0)
        x1c = min(float(W), x1); y1c = min(float(H), y1)
        ww = x1c - x0c; hh = y1c - y0c
        if ww > 0 and hh > 0:
            cands.append( (int(x0c), int(y0c), int(ww), int(hh)) )

    # Deduplicate candidates
    uniq = []
    seen = set()
    for c in cands:
        if c not in seen:
            uniq.append(c); seen.add(c)
    return uniq

def to_xywh(box, W, H):
    """
    Robustly convert a single bbox (various formats) to (x,y,w,h) clipped to image.
    If both interpretations valid, choose the one with larger area.
    Returns None if neither works.
    """
    cands = _as_xywh_candidates(box, W, H)
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    # choose the one with larger area (usually correct)
    areas = [(w*h, i) for i, (_, _, w, h) in enumerate(cands)]
    _, idx = max(areas)
    return cands[idx]

def pick_bbox_for_frame(poses_by_frame: dict,
                        frame_idx: int,
                        W: int, H: int,
                        pid_prefer: int | None = None,
                        mode: str = "largest",  # "largest" | "union" | "first" | "pid"
                        pad: int = 8) -> tuple[int,int,int,int] | None:
    """
    Return a bbox (x,y,w,h) for this frame.
    - If pid_prefer is not None or mode=="pid", pick box of that PID if available.
    - mode="largest": pick the largest-area box.
    - mode="first":   pick the first valid box.
    - mode="union":   union of all valid boxes in the frame.
    - pad: expand bbox by 'pad' pixels (then clip).
    """
    info = poses_by_frame.get(int(frame_idx))
    if not info:
        return None

    boxes = info.get('boxes') or []
    pids  = info.get('pids')  or list(range(len(boxes)))

    # collect candidate xywh boxes (+ keep pid)
    cand_xywh = []
    for b, p in zip(boxes, pids):
        xywh = to_xywh(b, W, H)
        if xywh is None: 
            continue
        cand_xywh.append( (p, xywh) )

    if not cand_xywh:
        return None

    # If choose by PID
    target_pid = pid_prefer if (pid_prefer is not None or mode == "pid") else None
    if target_pid is not None:
        for p, (x,y,w,h) in cand_xywh:
            if int(p) == int(target_pid):
                return _expand_clip_xywh(x, y, w, h, pad, W, H)
        # if requested PID not found, fall back to largest
        mode = "largest"

    if mode == "first":
        x,y,w,h = cand_xywh[0][1]
        return _expand_clip_xywh(x, y, w, h, pad, W, H)

    if mode == "union":
        x0 = y0 = +10**9
        x1 = y1 = -10**9
        for _, (x,y,w,h) in cand_xywh:
            x0 = min(x0, x); y0 = min(y0, y)
            x1 = max(x1, x+w); y1 = max(y1, y+h)
        if x1 <= x0 or y1 <= y0:
            return None
        x, y, w, h = int(x0), int(y0), int(x1-x0), int(y1-y0)
        return _expand_clip_xywh(x, y, w, h, pad, W, H)

    # default: largest-area
    best = None
    best_area = -1
    for _, (x,y,w,h) in cand_xywh:
        a = w*h
        if a > best_area:
            best_area = a
            best = (x,y,w,h)
    if best is None:
        return None
    return _expand_clip_xywh(*best, pad=pad, W=W, H=H)

# ==========================
# Main
# ==========================
if __name__ == "__main__":
    # Paths
    record_folder = 'recorded_data/recording_19700105_045503'
    data_file     = os.path.join(record_folder, 'robot_data.jsonl')
    vid_path      = os.path.join(record_folder, 'color.avi')

    # Video open
    vid = cv2.VideoCapture(vid_path)
    assert vid.isOpened(), f"Cannot open video: {vid_path}"
    fps = vid.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Time window (seconds)
    START_S, END_S = 9.0, 11.0
    start_frame = int(np.floor(START_S * fps))
    end_frame   = int(np.floor(END_S   * fps)) - 1
    if total > 0:
        start_frame = max(0, min(start_frame, total - 1))
        end_frame   = max(start_frame, min(end_frame, total - 1))
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Sensors / params (your local libs)
    from libs.dist2camera import Distance2Cam
    from libs.robotmetadata import Metadata

    distance = Distance2Cam(record_folder)
    metadata = Metadata(data_file)
    metadata_infos = metadata.metadata

    camera_params = distance.camera_params
    K, ds = build_from_camera_params(camera_params, (H, W))
    WIN = 0.05  # 25 ms window like the paper

    # Writer
    out_path = os.path.join(
        record_folder,
        f'ego_compensated_{int(START_S*1000)}-{int(END_S*1000)}ms.mp4'
    )
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    # Optional bbox M for Relative Contrast (set to None if unknown)
    poses_by_frame = parse_pose(os.path.join(record_folder, 'color.csv')) 

    # Loop
    mc = MotCompPy(K, (H, W), depth_scale=ds)
    prev_gray = None
    events_buf: List[Event] = []
    t_prev = start_frame / fps
    frame_idx = start_frame

    # Metric logs
    times_ms, means, vars_, RCs = [], [], [], []

    while True:
        if frame_idx > end_frame:
            break
        ok, frame = vid.read()
        if not ok:
            break
        t_s = frame_idx / float(fps)

        # State from metadata (frame-indexed)
        st = _state_from_metadata_infos(metadata_infos, frame_idx, t_s)
        mc.set_state(st['x'], st['y'], st['yaw'], st['vx'], st['vy'], st['vz'], st['vyaw'])

        # Depth for this frame (if available)
        depth = _depth_for_frame(distance, frame_idx)
        if isinstance(depth, np.ndarray):
            if depth.dtype != np.uint16:
                depth = depth.astype(np.uint16)
            if depth.shape[:2] != (H, W):
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            mc.load_depth(depth)

        # Generate pseudo-events by frame differencing (for demonstration)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            _, mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
            ys, xs = np.where(mask > 0)
            pts = list(zip(ys[::4], xs[::4]))  # subsample
            n = len(pts)
            dtf = max(t_s - t_prev, 1.0 / fps)
            for k, (yy, xx) in enumerate(pts):
                ts_k = t_prev + (k + 0.5) / max(n, 1) * dtf
                events_buf.append(Event(int(xx), int(yy), ts_k))
        prev_gray = gray
        t_prev = t_s

        # Keep only events in the last WIN seconds: [t_s - WIN, t_s]
        if events_buf and events_buf[0].ts < (t_s - WIN):
            i0 = 0
            cut = t_s - WIN
            while i0 < len(events_buf) and events_buf[i0].ts < cut:
                i0 += 1
            events_buf = events_buf[i0:]

        if events_buf:
            mc.load_events(events_buf)
            t0 = time.perf_counter()
            mc.run(do_rotation=True, do_translation=isinstance(depth, np.ndarray))

            print(
                "counts>=1:", np.count_nonzero(mc.event_counts >= 1),
                "| >=3:", np.count_nonzero(mc.event_counts >= 3),
                "| sumC:", int(mc.event_counts.sum()),
                "| maxC:", int(mc.event_counts.max()),
                "| maxΔt(ms):", mc.time_img.max()*1000.0
            )

            dt_ms = (time.perf_counter() - t0) * 1000.0

            # Denoise + metrics
            Td = denoise_time(mc.time_img, mc.event_counts, min_counts=1, ksize=5)
            mu, var = image_mean_var(Td, mc.event_counts, min_counts=1)

            bbox_for_frame = pick_bbox_for_frame(
                poses_by_frame,
                frame_idx=frame_idx,
                W=W, H=H,
                pid_prefer=None,       # hoặc đặt PID bạn muốn theo dõi
                mode="largest",        # "largest" | "union" | "first" | "pid"
                pad=8
            )

            rc = relative_contrast(Td, bbox_for_frame)

            times_ms.append(dt_ms)
            means.append(mu)
            vars_.append(var)
            if rc is not None:
                RCs.append(rc)

            heat = mc.get_visualization()
            over = overlay_heatmap(frame, heat, alpha=0.45)
        else:
            over = frame

        writer.write(over)
        frame_idx += 1

    vid.release()
    writer.release()
    print(f"Saved: {out_path}")
    if times_ms:
        print(f"Time per frame (ms): mean={np.mean(times_ms):.2f} ± {np.std(times_ms):.2f}")
        print(f"Output mean (ms): {np.mean(means):.5f} | var: {np.mean(vars_):.6f}")
        if RCs:
            print(f"Relative Contrast (if bbox set): mean={np.mean(RCs):.3f} ± {np.std(RCs):.3f}")
