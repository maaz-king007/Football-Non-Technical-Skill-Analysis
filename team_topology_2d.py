import cv2
import numpy as np

class TeamTopologyVisualizer:
    def __init__(self, width=400, height=300, pitch_length=105, pitch_width=68, invert_x=False, invert_y=False):
        # Canvas Settings
        self.width = width
        self.height = height
        
        # FLIP SETTINGS: Change these to True/False to fix mirroring issues
        self.invert_x = invert_x
        self.invert_y = invert_y
        
        # Real World Dimensions
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

        self.scale_x = self.width / self.pitch_length
        self.scale_y = self.height / self.pitch_width
        
        # Colors
        self.COLOR_PITCH = (50, 180, 50)     
        self.COLOR_LINES = (255, 255, 255)   
        self.COLOR_TEAM_0 = (255, 255, 255)  
        self.COLOR_TEAM_1 = (50, 50, 50)     
        self.COLOR_SUPPORT = (255, 255, 0)   
        self.COLOR_PRESSURE = (0, 0, 255)    

        self.MAX_PASS_DIST = 30.0   
        self.PRESSURE_DIST = 20.0   

    def convert_to_pixels(self, pos_meters):
        x, y = pos_meters
        
        px = int(x * self.scale_x)
        py = int(y * self.scale_y)
        
        # --- THE FIX: FLIP COORDINATES IF NEEDED ---
        if self.invert_x:
            px = self.width - px
        if self.invert_y:
            py = self.height - py
            
        return (px, py)

    def draw_pitch(self):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = self.COLOR_PITCH
        
        # Simple Pitch Lines
        cv2.rectangle(img, (0, 0), (self.width, self.height), self.COLOR_LINES, 2)
        mid_x = int(self.width / 2)
        cv2.line(img, (mid_x, 0), (mid_x, self.height), self.COLOR_LINES, 2)
        cv2.circle(img, (mid_x, int(self.height/2)), 20, self.COLOR_LINES, 2)
        return img

    def draw_frame(self, tracks_frame, team_with_ball):
        # 1. Prepare Canvas
        frame = self.draw_pitch()
        
        # 2. Find Ball Carrier
        ball_carrier_id = None
        ball_carrier_pos_m = None
        
        for track_id, data in tracks_frame.items():
            if data.get('has_ball', False):
                ball_carrier_id = track_id
                ball_carrier_team = data.get('team')
                if 'position_transformed' in data and data['position_transformed'] is not None:
                    ball_carrier_pos_m = data['position_transformed']
                    ball_carrier_pos_px = self.convert_to_pixels(ball_carrier_pos_m)
                break
        
        # 3. Draw Links
        if ball_carrier_id is not None and ball_carrier_pos_m is not None:
            for track_id, data in tracks_frame.items():
                if track_id == ball_carrier_id: continue
                if 'position_transformed' not in data or data['position_transformed'] is None: continue

                other_pos_m = data['position_transformed']
                other_pos_px = self.convert_to_pixels(other_pos_m)
                dist_m = np.linalg.norm(np.array(ball_carrier_pos_m) - np.array(other_pos_m))

                # Support (Teammates)
                if data.get('team') == ball_carrier_team:
                    if dist_m < self.MAX_PASS_DIST:
                        cv2.line(frame, ball_carrier_pos_px, other_pos_px, self.COLOR_SUPPORT, 1)
                
                # Pressure (Opponents)
                elif data.get('team') != ball_carrier_team:
                    if dist_m < self.PRESSURE_DIST:
                        thickness = 2 if dist_m < 5 else 1
                        cv2.line(frame, ball_carrier_pos_px, other_pos_px, self.COLOR_PRESSURE, thickness)

        # 4. Draw Players
        for track_id, data in tracks_frame.items():
            if 'position_transformed' not in data or data['position_transformed'] is None: continue
            
            pos_px = self.convert_to_pixels(data['position_transformed'])
            team = data.get('team')
            color = self.COLOR_TEAM_0 if team == 1 else self.COLOR_TEAM_1 
            
            if track_id == ball_carrier_id:
                cv2.circle(frame, pos_px, 6, (0, 255, 0), -1) 
            
            cv2.circle(frame, pos_px, 4, color, -1)

        return frame