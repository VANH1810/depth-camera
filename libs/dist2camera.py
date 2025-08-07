import cv2
import numpy as np
import json
import os
import argparse
import time
import glob
from datetime import datetime


class Distance2Cam:
	def __init__(self, recording_folder):
		"""Initialize distance tester with recording data"""
		self.recording_folder = recording_folder
		print("[DEBUG] Received record_folder:", recording_folder)  # 👈 Thêm dòng này
		self.depth_batches = []
		self.depth_metadata = None
		self.camera_params = None
		self.color_video = None
		self.depth_video = None
		self.current_frame = 0
		self.total_frames = 0
		self.fps = 30
		self.paused = False
		self.click_points = []
		self.show_distances = True
		
		# Load data
		self.load_recording_data()


	def load_recording_data(self):
		"""Load recording data from folder"""
		print(f"[INFO] Loading recording data from: {self.recording_folder}")
		
		# Load depth metadata
		metadata_path = os.path.join(self.recording_folder, 'depth_metadata.json')
		if not os.path.exists(metadata_path):
			raise FileNotFoundError(f"Depth metadata not found: {metadata_path}")
		
		with open(metadata_path, 'r') as f:
			self.depth_metadata = json.load(f)
		
		self.total_frames = self.depth_metadata['total_frames']
		print(f"[INFO] Total depth frames: {self.total_frames}")
		
		# Pre-load all depth batches for faster access
		depth_data_dir = os.path.join(self.recording_folder, 'depth_data')
		if not os.path.exists(depth_data_dir):
			raise FileNotFoundError(f"Depth data directory not found: {depth_data_dir}")
		
		print("[INFO] Loading depth batches...")
		self.depth_batches = []
		batch_info_list = []
		
		for batch_info in self.depth_metadata['batches']:
			batch_file = os.path.join(depth_data_dir, batch_info['filename'])
			if os.path.exists(batch_file):
				batch_data = np.load(batch_file)
				self.depth_batches.append({
					'frames': batch_data['depth_frames'],
					'timestamps': batch_data['timestamps'],
					'frame_count': batch_info['frame_count']
				})
				batch_info_list.append(batch_info)
				print(f"  Loaded batch {batch_info['batch_index']}: {batch_info['frame_count']} frames")
			else:
				print(f"  [WARN] Batch file not found: {batch_file}")
		
		print(f"[INFO] Loaded {len(self.depth_batches)} depth batches")
		
		# Load camera parameters
		# camera_params_path = os.path.join(self.recording_folder, 'camera_params.npz')
		camera_params_path = os.path.join(self.recording_folder, 'camera_parameters.json')
		if not os.path.exists(camera_params_path):
			raise FileNotFoundError(f"Camera parameters not found: {camera_params_path}")
		with open(camera_params_path, 'r') as read:
			self.camera_params = json.load(read)
		print(self.camera_params)
		# self.camera_params = np.load(camera_params_path)
		print(f"[INFO] Loaded camera parameters")
		
		# Load recording info
		recording_info_path = os.path.join(self.recording_folder, 'camera_parameters.json')
		if os.path.exists(recording_info_path):
			with open(recording_info_path, 'r') as f:
				recording_info = json.load(f)
				self.fps = recording_info.get('recording_info', {}).get('fps', 30)
				print(f"[INFO] Recording FPS: {self.fps}")
		
		# Open color video
		color_video_path = os.path.join(self.recording_folder, 'color.avi')
		if os.path.exists(color_video_path):
			self.color_video = cv2.VideoCapture(color_video_path)
			print(f"[INFO] Opened color video")
		
		# Open depth video for visualization
		depth_video_path = os.path.join(self.recording_folder, 'depth.avi')
		if os.path.exists(depth_video_path):
			self.depth_video = cv2.VideoCapture(depth_video_path)
			print(f"[INFO] Opened depth video")

	def get_depth_frame(self, frame_index):
		"""Get depth frame by index from batches"""
		if frame_index >= self.total_frames or frame_index < 0:
			return None
		
		# Find which batch contains this frame
		current_frame = 0
		for batch in self.depth_batches:
			batch_start = current_frame
			batch_end = current_frame + batch['frame_count']
			
			if batch_start <= frame_index < batch_end:
				# Frame is in this batch
				frame_in_batch = frame_index - batch_start
				return batch['frames'][frame_in_batch]
			
			current_frame = batch_end
		
		return None
	
	def pixel_to_distance_and_3d(self, x, y, depth_value):
		"""Convert pixel coordinates and depth value to distance and 3D coordinates"""
		if depth_value == 0:
			return None
		
		depth_scale = self.camera_params['depth_scale']
		fx = self.camera_params['color_intrinsics']['fx']
		fy = self.camera_params['color_intrinsics']['fy']
		ppx = self.camera_params['color_intrinsics']['ppx']
		ppy = self.camera_params['color_intrinsics']['ppy']
		
		# Distance = depth value * depth scale
		distance_meters = depth_value * depth_scale
		
		# 3D coordinates in camera space
		x_3d = (x - ppx) * distance_meters / fx
		y_3d = (y - ppy) * distance_meters / fy
		z_3d = np.sqrt(distance_meters*distance_meters - x_3d*x_3d - y_3d*y_3d)
		
		return distance_meters, x_3d, y_3d, z_3d
	
	def get_distance_at_point(self, x, y, frame_index=None):
		"""Get distance at specific point for current or specified frame"""
		if frame_index is None:
			frame_index = self.current_frame
		
		depth_frame = self.get_depth_frame(frame_index)
		if depth_frame is None:
			return None
		
		if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
			depth_value = depth_frame[y, x]  # Note: y first, then x
			return self.pixel_to_distance_and_3d(x, y, depth_value)
		
		return None


	def get_current_frames(self):
		"""Get current color and depth frames"""
		color_frame = None
		depth_frame = None
		
		# Get color frame
		if self.color_video is not None:
			self.color_video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
			ret, color_frame = self.color_video.read()
			if not ret:
				color_frame = None
		
		# Get depth frame for visualization
		if self.depth_video is not None:
			self.depth_video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
			ret, depth_frame = self.depth_video.read()
			if not ret:
				depth_frame = None
		
		return color_frame, depth_frame


if __name__ == '__main__':
	dist = Distance2Cam('recorded_data/recording_20250725_161358')