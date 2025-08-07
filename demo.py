from libs.dist2camera import Distance2Cam
from libs.robotmetadata import Metadata
import os
import pandas as pd
import numpy as np
import cv2
import json
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

if __name__ == '__main__':
    record_folder = 'recorded_data/recorded_data/recording_20250725_161358'
    data_file = os.path.join(record_folder, 'robot_data.jsonl')
    csv_file = os.path.join('color.csv')
    distance = Distance2Cam(record_folder)

    metadata = Metadata(data_file)
    metadata_infos = metadata.metadata
    poses_data = parse_pose(csv_file)
    vid_path = os.path.join(record_folder, 'color.avi')
    vid = cv2.VideoCapture(vid_path)

    writer_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vid.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter('test8.mp4',
                                writer_fourcc,
                                fps, (width, height))
    frame_id = 1
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        
        if frame_id in poses_data:
            pids = poses_data[frame_id]['pids']
            poses = poses_data[frame_id]['poses']
            boxes = poses_data[frame_id]['boxes'] 
            for pid, box, pose in zip(pids, boxes, poses):
                xmin, ymin, w,h, conf = box
                if conf < 0.7:
                    continue
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmin + w)
                ymax = int(ymin + h)
                cx = int(xmin + w/2)
                cy = int(ymin + h/2)
                # x0, y0 = int(xmin), int(ymin)
                # x1, y1 = int(xmin + w_box), int(ymin + h_box)
                # Vẽ bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                # Vẽ keypoints & limbs
                pts = np.array(pose).reshape(-1, 3)
                frame = _draw_limbs(pts[:, :2], frame)

                result = distance.get_distance_at_point(cx, cy, frame_id)
                # result2 = distance.get_distance_at_point(363, 150, frame_id)
                if result is not None:
                    # print(result)
                    robotdata = metadata_infos[frame_id]
                    # print(robotdata)
                    depth, x,y,z = result

                    xo, yo,zo = positioncam2org(x,y,z,robotdata)
                    print(robotdata)
                    print(xo, yo, zo)
                # print(result, result2)
        cv2.namedWindow('img',cv2.WINDOW_NORMAL)
        cv2.imshow('img',frame)
        cv2.waitKey(5)

        frame_id +=1