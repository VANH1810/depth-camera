### Depth Camera 
Dự án xử lý cảm biến và video cho robot: trích xuất độ sâu, quy đổi pixel→3D, bù chuyển động (ego-motion), vẽ skeleton/trajectory, và phân tích IMU.

## Tệp chính ở thư mục gốc

- **EgoMotion.py**: Bộ bù chuyển động ego-motion bằng Python.
  - Liên kết: `./EgoMotion.py`
  - **Lớp** `MotCompPy`:
    - Khởi tạo với ma trận nội tại camera `K`, kích thước ảnh, `depth_scale`.
    - `set_state/set_metadata`: cập nhật trạng thái robot theo thời gian (x, y, yaw, vx, vy, vz, vyaw).
    - `load_events`, `load_depth`: nạp sự kiện và khung độ sâu `uint16`.
    - `run(do_rotation, do_translation)`: bù quay (homography từ gyro) và tùy chọn bù tịnh tiến (dựa vào depth + velocity).
    - `get_visualization()`: xuất heatmap thời gian tích lũy sau bù.

- **analytics.py**: Trình phát và đo khoảng cách theo thời gian từ dữ liệu đã ghi.
  - Liên kết: `./analytics.py`
  - **Lớp** `DistanceTester`:
    - Tải `depth_metadata.json`, các batch `depth_data/*.npz`, `camera_parameters.json`, và mở `color.avi`/`depth.avi` nếu có.
    - Giao diện: nhấp chuột để lấy khoảng cách và tọa độ 3D (theo camera) tại pixel; vẽ nhãn trực tiếp trên khung hình; phím tắt điều khiển phát/tạm dừng/nhảy khung.
    - API: `get_depth_frame`, `get_timestamp`, `get_distance_at_point`, `pixel_to_distance_and_3d`.
  - CLI: liệt kê/chọn bản ghi, chọn bản ghi mới nhất, hoặc chỉ định đường dẫn.

- **demo.py**: Pipeline xuất video minh họa với skeleton + tọa độ 3D và inset quỹ đạo 3D.
  - Liên kết: `./demo.py`
  - **Lớp** `PoseWorldPipeline`:
    - Đọc pose từ CSV (khung→pids, bbox, keypoints), tính điểm “torso” bền vững (giao 2 đoạn vai–hông).
    - Lấy độ sâu bền vững quanh điểm (median trong ô nhỏ) → 3D camera; quy đổi 3D camera → 3D world bằng trạng thái robot từ `Metadata` (x, y, z, yaw, height).
    - Kiểm tra độ phủ depth trong bbox (coverage) từ `depth.avi`, chú màu bbox theo mức có dữ liệu depth.
    - Vẽ keypoints/ID, in nhãn tọa độ camera & world, ghép khung với hình quỹ đạo 3D (Matplotlib), ghi video MP4.
  - Có hàm xuất CSV các keypoint 3D (camera/world) nếu cần.

- **test.py**: Ví dụ đơn giản đọc pose CSV, vẽ bbox + skeleton, và thử truy vấn 3D tại tâm bbox.
  - Liên kết: `./test.py`

- **kalman_filter.py**: Bộ lọc Kalman 3D vị trí–vận tốc.
  - Liên kết: `./kalman_filter.py`
  - **Lớp** `KalmanFilter3D`: trạng thái 6D [X, Y, Z, VX, VY, VZ], ma trận F/H/Q/R tiêu chuẩn, Mahalanobis gating để loại outlier; `predict()` và `update()` (hỗ trợ R thích nghi).

## Thư mục `libs/`

- **libs/dist2camera.py**: Trình đọc dữ liệu độ sâu/param camera cho một bản ghi.
  - Liên kết: `./libs/dist2camera.py`
  - Tải `depth_metadata.json` và toàn bộ batch `npz` vào bộ nhớ để truy vấn nhanh theo frame.
  - Đọc tham số camera từ `camera_parameters.json` (fx, fy, ppx, ppy, depth_scale).
  - API chính: `get_depth_frame(frame_index)`, `get_distance_at_point(x, y, frame_index)` → trả về `(distance_m, x_3d, y_3d, z_3d)` theo hệ camera.

- **libs/robotmetadata.py**: Trình nạp metadata robot từ `robot_data.jsonl`.
  - Liên kết: `./libs/robotmetadata.py`
  - Ánh xạ `video_frame_number` → `{'time': ..., 'metadata': Status}` để tra cứu nhanh trạng thái (x, y, z, yaw, vx, vy, vyaw, height) theo frame.

- **libs/helper.py**: Hàm vẽ hỗ trợ hiển thị.
  - Liên kết: `./libs/helper.py`
  - `_draw_limbs(current_pose, img)`: nối keypoints theo cặp chuẩn (vai, khuỷu, hông, gối,...).
  - `_draw_pid(img, box, identity)`: hiển thị ID/người trên bbox; có chế độ font PIL.
  - `_draw_action(...)`, `_draw_label(...)`: vẽ nhãn hành động/văn bản trên ảnh.

## Thư mục `metadata/`

- **metadata/plot_imu_data.py**: Tạo biểu đồ/ảnh động từ log IMU dạng text JSON.
  - Liên kết: `./metadata/plot_imu_data.py`
  - Đọc file trong `metadata/Metadata/*.txt`, vẽ và lưu `height_vs_time.png`, `velocity_vs_time.png`, `angular_velocity_vs_time.png`, `position_vs_time.png`, và `trajectory.gif`.

## Thư mục dữ liệu (tham chiếu, không mô tả chi tiết)

- `recorded_data/`, `Depth_Test/`, `csv/`: chứa dữ liệu mẫu (video, depth batch, tham số camera, csv pose, ...). 

## Phụ thuộc môi trường

- Python 3.9+
- numpy, opencv-python, pandas, matplotlib, Pillow
