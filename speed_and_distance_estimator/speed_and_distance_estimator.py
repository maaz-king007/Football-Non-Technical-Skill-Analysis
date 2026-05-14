import cv2
import sys 
sys.path.append('../')
from utils import measure_distance, get_foot_position

class SpeedAndDistance_Estimator():
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24
    
    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}

        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue 
            
            number_of_frames = len(object_tracks)
            
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    if start_position is None or end_position is None:
                        continue
                    
                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    speed_meteres_per_second = distance_covered / time_elapsed
                    speed_km_per_hour = speed_meteres_per_second * 3.6

                    if object not in total_distance:
                        total_distance[object] = {}
                    
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    
                    total_distance[object][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]

                        # ============================================
                        # DEBUGGING DISTANCE TO BALL
                        # ============================================
                        
                        # 1. Check if Ball Data Exists
                        if 'ball' in tracks and frame_num_batch < len(tracks['ball']):
                            ball_track = tracks['ball'][frame_num_batch]
                            
                            # 2. Check if Ball ID (1) is in the track
                            if 1 in ball_track:
                                ball_data = ball_track[1]
                                
                                # 3. Check if we have the TRANSFORMED position (Meters)
                                if 'position_transformed' in ball_data and ball_data['position_transformed'] is not None:
                                    ball_pos = ball_data['position_transformed']
                                    player_pos = tracks[object][frame_num_batch][track_id]['position_transformed']
                                    
                                    if player_pos is not None:
                                        dist = measure_distance(player_pos, ball_pos)
                                        tracks[object][frame_num_batch][track_id]['distance_to_ball'] = dist
                                    else:
                                        # Only print occasionally to avoid spam
                                        if frame_num_batch % 100 == 0: print(f"Frame {frame_num_batch}: Player {track_id} transform missing.")
                                else:
                                    if frame_num_batch % 100 == 0: print(f"Frame {frame_num_batch}: Ball transform missing (Out of pitch?).")
                            else:
                                if frame_num_batch % 100 == 0: print(f"Frame {frame_num_batch}: Ball Not Detected.")

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees":
                    continue 
                
                for _, track_info in object_tracks[frame_num].items():
                    if "speed" in track_info:
                        speed = track_info.get('speed', None)
                        distance = track_info.get('distance', None)
                        
                        # Retrieve Distance to Ball
                        dist_to_ball = track_info.get('distance_to_ball', None) 
                        
                        if speed is None or distance is None:
                            continue
                        
                        bbox = track_info['bbox']
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1] += 40
                        position = tuple(map(int, position))
                        
                        cv2.putText(frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame, f"Run: {distance:.2f} m", (position[0], position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        
                        # Draw Distance to Ball (Blue)
                        if dist_to_ball is not None:
                            cv2.putText(frame, f"Ball: {dist_to_ball:.2f} m", (position[0], position[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        
            output_frames.append(frame)
        
        return output_frames