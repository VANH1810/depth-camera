import cv2
import numpy as np
import json
import os
import argparse
import time
import glob
from datetime import datetime

class DistanceTester:
	def __init__(self, recording_folder):
		"""Initialize distance tester with recording data"""
		self.recording_folder = recording_folder
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
		
		# self.camera_params = np.load(camera_params_path)
		with open(camera_params_path, 'r') as read:
			self.camera_params = json.load(read)

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
	
	def get_timestamp(self, frame_index):
		"""Get timestamp for frame by index"""
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
				return batch['timestamps'][frame_in_batch]
			
			current_frame = batch_end
		
		return None
	
	def pixel_to_distance_and_3d(self, x, y, depth_value):
		"""Convert pixel coordinates and depth value to distance and 3D coordinates"""
		if depth_value == 0:
			return None
		
		# depth_scale = self.camera_params['depth_scale']
		# fx = self.camera_params['depth_fx']
		# fy = self.camera_params['depth_fy']
		# ppx = self.camera_params['depth_ppx']
		# ppy = self.camera_params['depth_ppy']
		
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
		# depth_frame = depth_frame.T
		if depth_frame is None:
			return None
		if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
			depth_value = depth_frame[y, x]  # Note: y first, then x
			return self.pixel_to_distance_and_3d(x, y, depth_value)
		
		return None
	
	def mouse_callback(self, event, x, y, flags, param):
		"""Handle mouse clicks to add/remove measurement points"""
		if event == cv2.EVENT_LBUTTONDOWN:
			# Add point
			self.click_points.append((x, y))
			print(f"[INFO] Added point {len(self.click_points)}: ({x}, {y})")
		
		elif event == cv2.EVENT_RBUTTONDOWN:
			# Remove last point
			if self.click_points:
				removed = self.click_points.pop()
				print(f"[INFO] Removed point: {removed}")
	
	def draw_measurements(self, image):
		"""Draw measurement points and distances on image"""
		if not self.show_distances:
			return image
		
		overlay = image.copy()
		
		for i, (x, y) in enumerate(self.click_points):
			# Draw point
			cv2.circle(overlay, (x, y), 5, (0, 255, 0), -1)
			cv2.circle(overlay, (x, y), 8, (255, 255, 255), 2)
			
			# Get distance
			result = self.get_distance_at_point(x, y)
			
			if result:
				distance, x_3d, y_3d, z_3d = result
				
				# Draw distance text
				text = f"P{i+1}: {distance:.3f}m"
				text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
				
				# Position text above point
				text_x = max(0, min(x - text_size[0]//2, image.shape[1] - text_size[0]))
				text_y = max(text_size[1] + 5, y - 10)
				
				# Draw text background
				cv2.rectangle(overlay, 
							(text_x - 5, text_y - text_size[1] - 5),
							(text_x + text_size[0] + 5, text_y + 5),
							(0, 0, 0), -1)
				
				# Draw text
				cv2.putText(overlay, text, (text_x, text_y),
						   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
				
				# Draw 3D coordinates (smaller text)
				coord_text = f"3D: ({x_3d:.2f}, {y_3d:.2f}, {z_3d:.2f})"
				cv2.putText(overlay, coord_text, (text_x, text_y + 20),
						   cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 0), 1)
				print(distance,x_3d, y_3d, z_3d)
			else:
				# No depth data
				text = f"P{i+1}: No depth"
				cv2.putText(overlay, text, (x - 50, y - 10),
						   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
		
		# Blend overlay with original image
		return cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
	
	def draw_ui_info(self, image):
		"""Draw UI information on image"""
		height, width = image.shape[:2]
		
		# Frame info
		timestamp = self.get_timestamp(self.current_frame)
		timestamp_str = f" ({timestamp:.3f}s)" if timestamp is not None else ""
		frame_text = f"Frame: {self.current_frame + 1}/{self.total_frames}{timestamp_str}"
		cv2.putText(image, frame_text, (10, 30),
				   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
		
		# Status
		status = "PAUSED" if self.paused else "PLAYING"
		cv2.putText(image, status, (10, 60),
				   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
		
		# Controls
		controls = [
			"Controls:",
			"SPACE: Play/Pause",
			"A/D: Frame -/+",
			"Q/E: Frame -10/+10",
			"R: Reset frame to 0",
			"C: Clear points",
			"T: Toggle distances",
			"S: Screenshot",
			"P: Print measurements",
			"ESC: Exit",
			"",
			"Mouse:",
			"Left click: Add point",
			"Right click: Remove point"
		]
		
		y_offset = height - len(controls) * 20 - 10
		for i, control in enumerate(controls):
			if control == "Controls:" or control == "Mouse:":
				color = (0, 255, 255)
				thickness = 2
			else:
				color = (255, 255, 255)
				thickness = 1
			
			cv2.putText(image, control, (10, y_offset + i * 20),
					   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
		
		# Points info
		if self.click_points:
			points_text = f"Points: {len(self.click_points)}"
			cv2.putText(image, points_text, (10, 90),
					   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
		
		return image
	
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
	
	def print_current_measurements(self):
		"""Print all current measurements to console"""
		if not self.click_points:
			return
		
		timestamp = self.get_timestamp(self.current_frame)
		timestamp_str = f" (timestamp: {timestamp:.3f}s)" if timestamp is not None else ""
		
		print(f"\n=== MEASUREMENTS FOR FRAME {self.current_frame + 1}{timestamp_str} ===")
		for i, (x, y) in enumerate(self.click_points):
			result = self.get_distance_at_point(x, y)
			if result:
				distance, x_3d, y_3d, z_3d = result
				print(f"Point {i+1} ({x:3d}, {y:3d}): "
					  f"Distance = {distance:.3f}m, "
					  f"3D = ({x_3d:6.3f}, {y_3d:6.3f}, {z_3d:6.3f})")
			else:
				print(f"Point {i+1} ({x:3d}, {y:3d}): No depth data")
	
	def run(self):
		"""Main loop for real-time distance testing"""
		print("\n" + "="*60)
		print("DISTANCE TESTER - REAL-TIME PLAYBACK")
		print("="*60)
		print(f"Recording: {os.path.basename(self.recording_folder)}")
		print(f"Total frames: {self.total_frames}")
		print(f"FPS: {self.fps}")
		print("="*60)
		
		# Create window and set mouse callback
		cv2.namedWindow('Distance Tester', cv2.WINDOW_NORMAL)
		cv2.setMouseCallback('Distance Tester', self.mouse_callback)
		
		frame_delay = 1.0 / self.fps
		last_frame_time = time.time()
		
		while True:
			current_time = time.time()
			
			# Frame timing
			if not self.paused and (current_time - last_frame_time) >= frame_delay:
				self.current_frame = (self.current_frame + 1) % self.total_frames
				last_frame_time = current_time
			
			# Get current frames
			color_frame, depth_frame = self.get_current_frames()
			
			# Use color frame if available, otherwise create from depth
			if color_frame is not None:
				display_frame = color_frame.copy()
			elif depth_frame is not None:
				# Convert depth to color
				display_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
			else:
				# Create blank frame
				display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
				cv2.putText(display_frame, "No video data available", 
						   (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
			
			# Draw measurements
			display_frame = self.draw_measurements(display_frame)
			
			# Draw UI info
			display_frame = self.draw_ui_info(display_frame)
			
			# Show frame
			cv2.imshow('Distance Tester', display_frame)
			
			# Handle keyboard input
			key = cv2.waitKey(1) & 0xFF
			
			if key == 27:  # ESC
				break
			elif key == ord(' '):  # SPACE - Play/Pause
				self.paused = not self.paused
				status = "PAUSED" if self.paused else "PLAYING"
				print(f"[INFO] {status}")
			elif key == ord('a'):  # A - Previous frame
				self.current_frame = max(0, self.current_frame - 1)
				self.print_current_measurements()
			elif key == ord('d'):  # D - Next frame
				self.current_frame = min(self.total_frames - 1, self.current_frame + 1)
				self.print_current_measurements()
			elif key == ord('q'):  # Q - Jump back 10 frames
				self.current_frame = max(0, self.current_frame - 10)
				self.print_current_measurements()
			elif key == ord('e'):  # E - Jump forward 10 frames
				self.current_frame = min(self.total_frames - 1, self.current_frame + 10)
				self.print_current_measurements()
			elif key == ord('r'):  # R - Reset to frame 0
				self.current_frame = 0
				print("[INFO] Reset to frame 0")
				self.print_current_measurements()
			elif key == ord('c'):  # C - Clear points
				self.click_points.clear()
				print("[INFO] Cleared all points")
			elif key == ord('t'):  # T - Toggle distance display
				self.show_distances = not self.show_distances
				status = "ON" if self.show_distances else "OFF"
				print(f"[INFO] Distance display: {status}")
			elif key == ord('s'):  # S - Save screenshot
				timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
				screenshot_path = os.path.join(self.recording_folder, f"screenshot_{timestamp}.png")
				cv2.imwrite(screenshot_path, display_frame)
				print(f"[INFO] Screenshot saved: {screenshot_path}")
			elif key == ord('p'):  # P - Print measurements
				self.print_current_measurements()
			cv2.waitKey()
		
		# Cleanup
		cv2.destroyAllWindows()
		if self.color_video:
			self.color_video.release()
		if self.depth_video:
			self.depth_video.release()

def find_latest_recording(base_dir="Depth_Test"):
	"""Find the latest recording folder"""
	if not os.path.exists(base_dir):
		return None
	
	recording_folders = glob.glob(os.path.join(base_dir, "recording_*"))
	if not recording_folders:
		return None
	
	# Sort by modification time (newest first)
	recording_folders.sort(key=lambda x: os.path.getmtime(x), reverse=True)
	return recording_folders[0]

def list_available_recordings(base_dir="Depth_Test"):
	"""List all available recording folders"""
	if not os.path.exists(base_dir):
		print(f"[ERROR] Base directory '{base_dir}' does not exist")
		return []
	
	recording_folders = glob.glob(os.path.join(base_dir, "recording_*"))
	recording_folders.sort(reverse=True)  # Newest first
	
	print(f"\nAvailable recordings in '{base_dir}':")
	print("-" * 50)
	for i, folder in enumerate(recording_folders, 1):
		folder_name = os.path.basename(folder)
		# Try to get recording info
		info_path = os.path.join(folder, "recording_summary.json")
		info_str = ""
		if os.path.exists(info_path):
			try:
				with open(info_path, 'r') as f:
					summary = json.load(f)
					duration = summary.get('recording_info', {}).get('duration_seconds', 0)
					frames = summary.get('recording_info', {}).get('total_frames', 0)
					info_str = f" ({duration:.1f}s, {frames} frames)"
			except:
				pass
		
		# Check if depth metadata exists
		metadata_path = os.path.join(folder, "depth_metadata.json")
		if os.path.exists(metadata_path):
			try:
				with open(metadata_path, 'r') as f:
					metadata = json.load(f)
					total_frames = metadata.get('total_frames', 0)
					if not info_str:
						info_str = f" ({total_frames} depth frames)"
			except:
				pass
		
		print(f"{i:2d}. {folder_name}{info_str}")
	
	return recording_folders

def parse_arguments():
	"""Parse command line arguments"""
	parser = argparse.ArgumentParser(description='Test distance measurements from recorded RealSense data')
	
	parser.add_argument('--recording-folder', type=str,
					   help='Path to specific recording folder')
	parser.add_argument('--base-dir', type=str, default='Depth_Test',
					   help='Base directory containing recordings (default: Depth_Test)')
	parser.add_argument('--list', action='store_true',
					   help='List available recordings and exit')
	parser.add_argument('--latest', action='store_true',
					   help='Use latest recording automatically')
	
	return parser.parse_args()

def main():
	"""Main function"""
	args = parse_arguments()
	
	if args.list:
		list_available_recordings(args.base_dir)
		return
	
	recording_folder = "Depth_Test"
	
	if args.recording_folder:
		# Use specified folder
		recording_folder = args.recording_folder
	elif args.latest:
		# Use latest recording
		recording_folder = find_latest_recording(args.base_dir)
		if recording_folder:
			print(f"[INFO] Using latest recording: {os.path.basename(recording_folder)}")
	else:
		# Interactive selection
		recordings = list_available_recordings(args.base_dir)
		if not recordings:
			print("[ERROR] No recordings found")
			return
		
		try:
			choice = input(f"\nSelect recording (1-{len(recordings)}) or 'latest': ").strip()
			if choice.lower() == 'latest':
				recording_folder = recordings[0]
			else:
				idx = int(choice) - 1
				if 0 <= idx < len(recordings):
					recording_folder = recordings[idx]
				else:
					print("[ERROR] Invalid selection")
					return
		except (ValueError, KeyboardInterrupt):
			print("[ERROR] Invalid input or cancelled")
			return
	
	if not recording_folder or not os.path.exists(recording_folder):
		print(f"[ERROR] Recording folder not found: {recording_folder}")
		return
	
	try:
		# Create and run distance tester
		tester = DistanceTester(recording_folder)
		tester.run()
	except Exception as e:
		print(f"[ERROR] Failed to run distance tester: {e}")
		import traceback
		traceback.print_exc()

if __name__ == "__main__":
	main()