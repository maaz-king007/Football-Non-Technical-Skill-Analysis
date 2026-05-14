import sys 
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        # Increased threshold to 100 (more forgiving)
        self.max_player_ball_distance = 100 
    
    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)
        
        closest_player_id = -1
        min_distance = 99999
        
        # 1. First, find the ABSOLUTE closest player (Greedy search)
        for player_id, player in players.items():
            player_bbox = player['bbox']
            
            # Use Feet Coordinates (Bottom Center)
            x1, y1, x2, y2 = player_bbox
            player_feet_x = (x1 + x2) / 2
            player_feet_y = y2 
            player_feet_position = (player_feet_x, player_feet_y)
            
            # Calculate distance
            distance = measure_distance(player_feet_position, ball_position)
            
            if distance < min_distance:
                min_distance = distance
                closest_player_id = player_id

        # 2. Only reject if the CLOSEST player is still ridiculously far away
        if min_distance < self.max_player_ball_distance:
            return closest_player_id
        else:
            return -1