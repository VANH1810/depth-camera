from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import cv2


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
    """Skew-symmetric matrix of a 3D vector."""
    x, y, z = v
    return np.array([[0, -z, y],
                     [z, 0, -x],
                     [-y, x, 0]], dtype=np.float32)


def rodrigues_from_rotvec(rvec: np.ndarray) -> np.ndarray:
    """Return 3x3 rotation matrix from a rotation vector using cv2.Rodrigues."""
    R, _ = cv2.Rodrigues(rvec.astype(np.float32))
    return R.astype(np.float32)


def Rz(yaw: float) -> np.ndarray:
    """Rotation around Z (world up) by yaw radians."""
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
    return np.array([[fx, 0.0, ppx],
                     [0.0, fy, ppy],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def build_from_camera_params(camera_params: dict, image_size: Tuple[int, int]):
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
    Python port of ego-motion compensation tailored to your metadata layout.

    Assumptions
    ----------
    - Time unit: seconds.
    - Velocities vx, vy are in WORLD frame (global), units m/s.
    - Angular velocity vyaw is about WORLD Z (rad/s). If you have full gyro
      (wx, wy, wz) in BODY frame, you can pass that instead (see set_imu()).
    - Depth image: uint16 range measurement (same convention as paper code).
      ReadDepth() converts 'range' to optical-axis Z using intrinsics.
    - Intrinsics K are in pixels. Image size = (H, W).

    Notes
    -----
    - Rotational compensation updates the warp every 1 ms for efficiency, same as paper.
    - Translational compensation precomputes per-millisecond world translations
      and maps them into the camera frame using: cam2body^-1 * fc2world^-1 * T_world * fc2world * cam2body
      because your vx, vy are given in the WORLD frame.
    - Following the author's code, re-projection uses the *original* z_depth to
      normalize (plane-induced warp). If you prefer exact perspective, replace
      1/z_depth with 1/Z_prime.
    """

    def __init__(self,
                 K: np.ndarray,
                 image_size: Tuple[int, int],  # (H, W)
                 depth_scale: float = 1.0,
                 depth_min_m: float = 0.3,
                 depth_max_m: float = 10.0):
        self.K = K.astype(np.float32)
        self.K_inv = np.linalg.inv(self.K).astype(np.float32)
        self.H, self.W = image_size
        self.depth_scale = float(depth_scale)  # meters per depth unit
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)

        # Map BODY gyro to CAMERA axes for rotation warp (recommended)
        self.map_body_to_cam_rotation = False  # pure ego-motion (no frame conversion)

        # Buffers
        self.events: List[Event] = []
        self.depth_img: Optional[np.ndarray] = None

        # Metadata (world pose, velocities)
        self.height_static = None  # optional
        self.pos_xy = np.zeros(2, dtype=np.float32)   # world (x, y)
        self.yaw = 0.0                                 # world yaw (rad)
        self.v_world = np.zeros(3, dtype=np.float32)   # (vx, vy, vz)
        self.omega_body = np.zeros(3, dtype=np.float32) # (wx, wy, wz) in BODY; default yaw-only

        # --- Simple translation (no frame conversions) ---
        # If True, treat provided translational velocity as already in CAMERA frame.
        # Then translation warp uses T = [I | v_cam*dt] directly (no cam2body/fc2world chaining)
        self.simple_trans_no_frames = False
        self.v_cam = np.zeros(3, dtype=np.float32)  # (vx_cam, vy_cam, vz_cam) in CAMERA frame

        # Precomputed transforms
        self.cam2body = self._compute_cam2body()
        self.fc2world = np.eye(4, dtype=np.float32)  # flight-controller/body to world (updated per packet)

        # Outputs
        self.time_img = None          # float32 HxW: mean timestamp per pixel
        self.event_counts = None      # int32   HxW: counts per pixel

    # ---------- Metadata / IO ----------
    def set_metadata(self,
                     x: float, y: float,
                     yaw: float,
                     vx: float, vy: float,
                     vz: float = 0.0,
                     vyaw: Optional[float] = None,
                     height_static: Optional[float] = None,
                     z_current: Optional[float] = None):
        self.pos_xy[:] = (x, y)
        self.yaw = float(yaw)
        self.v_world[:] = (vx, vy, vz)
        # If only vyaw available, treat gyro as [0,0,vyaw] in BODY.
        if vyaw is not None:
            self.omega_body[:] = (0.0, 0.0, float(vyaw))
        if height_static is not None:
            self.height_static = float(height_static)
        # z_current is unused by the core algorithm here, but kept for completeness.

        self._update_fc2world()

    def set_vyaw_only(self, vyaw: float):
        """Set only yaw-rate (rad/s); other angular rates = 0. No frame conversion."""
        self.omega_body[:] = (0.0, 0.0, float(vyaw))

    def set_v_cam_only(self, vx_cam: float, vy_cam: float, vz_cam: float = 0.0):
        """Provide translational velocity already expressed in CAMERA frame (m/s).
        Use together with `self.simple_trans_no_frames = True` if you want translation
        compensation without any frame conversions."""
        self.v_cam[:] = (vx_cam, vy_cam, vz_cam)

    def set_imu(self, wx: float, wy: float, wz: float):
        """Optional: full angular velocity in BODY frame (rad/s)."""
        self.omega_body[:] = (wx, wy, wz)

    def load_events(self, events: List[Event]):
        self.events = events

    def load_depth(self, depth_img_u16: np.ndarray):
        assert depth_img_u16.dtype == np.uint16, "Depth image must be uint16"
        assert depth_img_u16.shape[:2] == (self.H, self.W), "Depth size mismatch"
        self.depth_img = depth_img_u16

    def apply_camera_params(self, camera_params: dict):
        """Set intrinsics K and depth scale from your camera_params dict."""
        K, ds = build_from_camera_params(camera_params, (self.H, self.W))
        self.K = K.astype(np.float32)
        self.K_inv = np.linalg.inv(self.K).astype(np.float32)
        self.depth_scale = float(ds)

    # ---------- Core pipeline ----------
    def run(self, do_rotation: bool = True, do_translation: bool = False):
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

    def _update_fc2world(self):
        # Here we only use yaw (Z-up world). If you also have roll/pitch, extend this.
        R = Rz(self.yaw)
        self.fc2world = make_T(R, np.array([self.pos_xy[0], self.pos_xy[1], 0.0], dtype=np.float32))

    def _compute_cam2body(self) -> np.ndarray:
        # body: x->front, y->right, z->up
        # cam : z->front, x->right, y->down
        # => rotate Y by -90deg, then X by +90deg (same as C++ code)
        Ry = cv2.Rodrigues(np.array([0, -0.5*np.pi, 0], dtype=np.float32))[0]
        Rx = cv2.Rodrigues(np.array([0.5*np.pi, 0, 0], dtype=np.float32))[0]
        R = Rx @ Ry
        return make_T(R, np.zeros(3, dtype=np.float32))

    # ---------- Stage A: no compensation (for debugging) ----------
    def run_ego_rotation_only(self):
        """Fast path: only rotational ego-motion compensation (image-plane warp).
        - uses self.omega_body = (0,0,vyaw)
        - updates warp every 1 ms
        - NO cam/body/world transforms, NO translation, NO depth
        Returns (time_img, event_counts)
        """
        self._clear_outputs()
        self._rotational_compensate()
        return self.time_img, self.event_counts

    def _accumulate_no_compensation(self):
        t0 = self.events[0].ts
        for e in self.events:
            dt = float(e.ts - t0)
            x, y = int(e.x), int(e.y)
            if 0 <= x < self.W and 0 <= y < self.H:
                c = self.event_counts[y, x] + 1
                # running mean
                self.time_img[y, x] += (dt - self.time_img[y, x]) / c
                self.event_counts[y, x] = c

    # ---------- Stage B: rotational compensation ----------
    def _rotational_compensate(self):
        if not self.events:
            return
        t0 = self.events[0].ts
        prev_ms_mark = -1
        rot_K = np.eye(3, dtype=np.float32)

        # Pre-allocate event vector (homogeneous)
        ev = np.ones((3, 1), dtype=np.float32)

        for e in self.events:
            dt = float(e.ts - t0)  # seconds
            cur_ms_mark = int(dt * 1000.0)

            # Update projection warp every 1 ms
            if cur_ms_mark != prev_ms_mark:
                prev_ms_mark = cur_ms_mark
                # rotation vector from BODY gyro; here we assume only wz (yaw rate)
                rotvec_body = self.omega_body * dt  # (wx, wy, wz) * seconds
                # Map BODY rotation into CAMERA frame if requested
                if self.map_body_to_cam_rotation:
                    R_cb = self.cam2body[:3, :3]
                    rotvec_cam = R_cb.T @ rotvec_body
                else:
                    rotvec_cam = rotvec_body
                R = rodrigues_from_rotvec(rotvec_cam.astype(np.float32))
                rot_K = (self.K @ R.T @ self.K_inv).astype(np.float32)

            # Warp event (x, y, 1)
            ev[0, 0] = e.x
            ev[1, 0] = e.y
            warped = rot_K @ ev  # 3x1
            # Homogeneous -> in-place normalization
            if warped[2, 0] != 0:
                xw = warped[0, 0] / warped[2, 0]
                yw = warped[1, 0] / warped[2, 0]
            else:
                xw, yw = -1, -1

            ix, iy = int(xw), int(yw)
            if 0 <= ix < self.W and 0 <= iy < self.H:
                c = self.event_counts[iy, ix] + 1
                self.time_img[iy, ix] += (dt - self.time_img[iy, ix]) / c
                self.event_counts[iy, ix] = c

    # ---------- Stage C: translational compensation ----------
    def _translational_compensate(self):
        # Build per-ms transforms up to max timestamp in time_img
        max_dt = float(self.time_img.max())
        max_ms = int(max_dt * 1000.0 + 1)

        # Precompute per-ms translation transforms
        trans_vec = [np.eye(4, dtype=np.float32) for _ in range(max_ms)]
        for i in range(max_ms):
            dt = i * 1e-3
            if self.simple_trans_no_frames:
                # Directly use camera-frame velocity (no frame conversions)
                t_cam = (self.v_cam * dt).astype(np.float32)
                T = make_T(np.eye(3, dtype=np.float32), t_cam)
            else:
                # Map world translation into camera coordinates
                t_world = (self.v_world * dt).astype(np.float32)
                T_world = make_T(np.eye(3, dtype=np.float32), t_world)
                T = np.linalg.inv(self.cam2body) @ np.linalg.inv(self.fc2world) @ T_world @ self.fc2world @ self.cam2body
            trans_vec[i] = T.astype(np.float32)

        # New accumulators
        new_counts = np.zeros_like(self.event_counts, dtype=np.int32)
        new_time   = np.zeros_like(self.time_img, dtype=np.float32)

        # Iterate over pixels (like the paper code). Can be vectorized later if needed.
        for y in range(self.H):
            for x in range(self.W):
                c = self.event_counts[y, x]
                dt = self.time_img[y, x]
                if dt <= 0.001 or dt >= 1.0 or c <= 0:
                    continue

                # Skip if depth invalid
                if not self._depth_in_range(x, y):
                    continue

                z_depth = self._read_depth_Z(x, y)

                # Unproject to 3D at Z = z_depth
                pix = np.array([[x], [y], [1]], dtype=np.float32)
                ray = (self.K_inv @ pix).reshape(3)
                P_cam = ray * z_depth  # (X, Y, Z)
                P4 = np.array([P_cam[0], P_cam[1], P_cam[2], 1.0], dtype=np.float32)

                ms = int(dt * 1000.0)
                ms = max(0, min(ms, max_ms - 1))
                P4p = trans_vec[ms] @ P4

                # Reproject using original z_depth (plane-induced warp)
                Pp = P4p[:3]
                pixp = (self.K @ Pp.reshape(3, 1)) * (1.0 / max(z_depth, 1e-6))
                x2 = int(pixp[0, 0])
                y2 = int(pixp[1, 0])

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
        Convert range (uint16) to optical-axis Z using intrinsics, as in C++:
        Z = sqrt( range^2 / ( 1 + ((x-cx)^2)/fx^2 + ((y-cy)^2)/fy^2 ) )
        NOTE: If your depth map already stores Z along optical axis, replace this
        with simply `return depth_in_meters` (after scaling).
        """
        depth_range = self.depth_scale * float(self.depth_img[y, x])
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        denom = 1.0 + ((x - cx) ** 2) / (fx ** 2) + ((y - cy) ** 2) / (fy ** 2)
        Z = np.sqrt((depth_range * depth_range) / max(denom, 1e-8))
        return float(Z)

    def get_visualization(self) -> np.ndarray:
        m = cv2.normalize(self.time_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(m, cv2.COLORMAP_JET)


# ==========================
# Video + Metadata integration runner
# ==========================
import os, json

def _safe_get_camera_params_from_distance(distance):
    # Expect distance.camera_params like user's snippet
    cam = getattr(distance, 'camera_params', None)
    if cam is not None:
        return cam
    # Try a method
    for m in ('get_camera_params','get_params','get_config'):
        if hasattr(distance, m):
            return getattr(distance, m)()
    raise ValueError("Distance2Cam has no 'camera_params' attribute or getter.")


def _depth_for_frame(distance, frame_idx: int):
    # Try common ways to fetch per-frame depth aligned to color
    for m in ('get_depth_frame','get_depth_at','get_depth_frame_by_index','depth_at'):
        if hasattr(distance, m):
            arr = getattr(distance, m)(frame_idx)
            if isinstance(arr, np.ndarray) and arr.dtype == np.uint16:
                return arr
    # Fallback: static depth (not ideal, but keeps pipeline running)
    if hasattr(distance, 'depth') and isinstance(distance.depth, np.ndarray):
        return distance.depth
    if hasattr(distance, 'depth_img') and isinstance(distance.depth_img, np.ndarray):
        return distance.depth_img
    return None


def _read_metadata_jsonl(path: str):
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

def _meta_at_time(recs, t_s: float):
    # Very simple nearest-neighbor over a 'timestamp'/'ts' field in seconds or ms
    if not recs:
        return {}
    def _ts(r):
        for k in ('timestamp','ts','time','t'):
            if k in r:
                val = r[k]
                # if looks like ms, convert
                return float(val) / 1000.0 if val and val > 1e6 else float(val)
        return None
    best, best_err = None, 1e18
    for r in recs:
        tt = _ts(r)
        if tt is None:
            continue
        e = abs(tt - t_s)
        if e < best_err:
            best, best_err = r, e
    return best or {}


def _vec_from_meta(meta: dict, keys, default=0.0):
    return tuple(float(meta.get(k, default)) for k in keys)


def overlay_heatmap(frame_bgr: np.ndarray, heatmap_bgr: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap_bgr = cv2.resize(heatmap_bgr, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(frame_bgr, 1.0, heatmap_bgr, alpha, 0)


def _extract_status_from_record(rec: dict) -> dict:
    # Navigate to robot_data -> Robot1 -> Status if present
    if not isinstance(rec, dict):
        return {}
    # If the record is the outer object with keys 'metadata' and 'robot_data'
    if 'robot_data' in rec:
        rbd = rec.get('robot_data') or {}
        # Common schemas: {"Robot1": {"Status": {...}}}
        if isinstance(rbd, dict):
            if 'Robot1' in rbd and isinstance(rbd['Robot1'], dict):
                st = rbd['Robot1'].get('Status', {})
                if isinstance(st, dict):
                    return st
            # Or directly 'Status' under robot_data
            if 'Status' in rbd and isinstance(rbd['Status'], dict):
                return rbd['Status']
    # Already a status-like dict
    if 'vx' in rec or 'vyaw' in rec or 'yaw' in rec:
        return rec
    # Nested fallbacks
    for k, v in rec.items():
        if isinstance(v, dict):
            st = _extract_status_from_record(v)
            if st:
                return st
    return {}


def _state_from_metadata_infos(metadata_infos, frame_idx: int, t_s: float):
    """Return a dict with x,y,z,yaw,vx,vy,vz,vyaw,height for the given frame.
    It follows the user's schema and only *uses* the already loaded metadata_infos.
    """
    # Case 1: dict keyed by frame index
    if isinstance(metadata_infos, dict):
        # exact key
        if frame_idx in metadata_infos:
            rec = metadata_infos[frame_idx]
        elif str(frame_idx) in metadata_infos:
            rec = metadata_infos[str(frame_idx)]
        else:
            # single-record dict
            rec = metadata_infos
    # Case 2: list of records
    elif isinstance(metadata_infos, list) and metadata_infos:
        # Prefer exact match via metadata.video_frame_number
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
            # Fallback: nearest by receive_timestamp (seconds from epoch-like baseline)
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


if __name__ == "__main__":
    # --- User-provided paths ---
    record_folder = 'recorded_data/recording_19700105_045503'
    data_file     = os.path.join(record_folder, 'robot_data.jsonl')
    csv_file      = os.path.join('color.csv')  # optional, for drawing boxes/poses if you want

    # --- IO: video ---
    vid_path = os.path.join(record_folder, 'color.avi')
    vid = cv2.VideoCapture(vid_path)
    assert vid.isOpened(), f"Cannot open video: {vid_path}"

    fps    = vid.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(vid.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # >>> ADD: choose time window [START_S, END_S)
    START_S = 9.0
    END_S   = 10.0
    start_frame = int(np.floor(START_S * fps))
    end_frame   = int(np.floor(END_S   * fps)) - 1  # inclusive
    if total > 0:
        start_frame = max(0, min(start_frame, total - 1))
        end_frame   = max(start_frame, min(end_frame, total - 1))
        
    # seek to start
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(record_folder, 'ego_compensated.mp4')
    writer = cv2.VideoWriter(out_path, writer_fourcc, fps, (width, height))

    # --- Sensors / metadata ---
    try:
        from libs.dist2camera import Distance2Cam
        from libs.robotmetadata import Metadata
    except Exception as e:
        raise RuntimeError("Make sure libs.dist2camera and libs.robotmetadata are importable.")

    distance = Distance2Cam(record_folder)
    metadata = Metadata(data_file)
    
    # Nguồn metadata: ưu tiên dict từ Metadata, fallback đọc JSONL nếu rỗng
    metadata_infos = metadata.metadata if getattr(metadata, 'metadata', None) else {}
    if not metadata_infos:
        metadata_infos = _read_metadata_jsonl(data_file)

    # Kiểm tra cấu trúc nhanh
    import pprint
    print(f"metadata_infos type={type(metadata_infos).__name__}")
    if isinstance(metadata_infos, dict):
        print(f"dict keys count={len(metadata_infos)}")
        sample_key = next(iter(metadata_infos), None)
        if sample_key is not None:
            print("sample key:", sample_key)
            pprint.pprint(metadata_infos[sample_key])
    elif isinstance(metadata_infos, list):
        print(f"list length={len(metadata_infos)}")
        if metadata_infos:
            pprint.pprint(metadata_infos[0])
    
    # Camera params → K & depth_scale
    camera_params = distance.camera_params
    K, ds = build_from_camera_params(camera_params, (height, width))

    mc = MotCompPy(K, (height, width), depth_scale=ds)

    # If you only have angular velocity about yaw:
    mc.map_body_to_cam_rotation = True  # recommended if gyro is in BODY

    # Optional: load poses if your function is available
    # try:
    #     poses_data = parse_pose(csv_file)
    # except Exception:
    #     poses_data = None

    # Pseudo-events builder (frame differencing)
    prev_gray = None
    t_prev = 0.0
    frame_idx = 0

    while True:
        ok, frame = vid.read()
        if not ok:
            break
        t_s = frame_idx / float(fps)  # seconds

        # --- Build/update motion state from user's metadata_infos ---
        state = _state_from_metadata_infos(metadata_infos, frame_idx, t_s)
        mc.set_metadata(
            x=state['x'],
            y=state['y'],
            yaw=state['yaw'],
            vx=state['vx'],
            vy=state['vy'],
            vz=state['vz'],
            vyaw=state['vyaw']
        )

        # --- Độ sâu cho khung hình này (nếu có) ---
        depth_u16 = _depth_for_frame(distance, frame_idx)
        if depth_u16 is not None:
            if depth_u16.shape[:2] != (height, width):
                depth_u16 = cv2.resize(depth_u16, (width, height), interpolation=cv2.INTER_NEAREST)
            mc.load_depth(depth_u16)

        # --- Pseudo events from frame difference (fallback when real event data not present) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        events = []
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            # Threshold and stride to limit event count
            _, mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
            ys, xs = np.where(mask > 0)
            # Subsample to reduce load
            for y, x in zip(ys[::4], xs[::4]):
                events.append(Event(int(x), int(y), t_s))
        prev_gray = gray

        # --- Run ego-motion compensation on this slice ---
        if events:
            mc.load_events(events)
            mc.run(do_rotation=True, do_translation=depth_u16 is not None)
            heat = mc.get_visualization()
            over = overlay_heatmap(frame, heat, alpha=0.45)
        else:
            over = frame

        writer.write(over)
        frame_idx += 1

    vid.release()
    writer.release()
    print(f"Saved: {out_path}")
