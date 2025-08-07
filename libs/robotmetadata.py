import json
import os
from datetime import datetime

class Metadata:
    def __init__(self, data_file):
        self.metadata = {}
        self.data_file = data_file
        self._load_metadata()
    
    def _load_metadata(self):
        with open(self.data_file, 'r') as read:
            for line in read:
                data = json.loads(line)
                meta_info = data['metadata']
                robot_info = data['robot_data']
                video_frame_number = meta_info['video_frame_number']
                receive_timestamp = float(meta_info['receive_timestamp'])
                receive_time = datetime.fromtimestamp(receive_timestamp)
                if not video_frame_number in self.metadata:
                    self.metadata[video_frame_number] = {'time':receive_timestamp,'metadata':robot_info['Robot1']['Status']}
    