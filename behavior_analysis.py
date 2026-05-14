import numpy as np

class BehaviorAnalyzer:
    def __init__(self):
        # SETTINGS: TUNED FOR REALISTIC DETECTION
        self.SPRINT_SPEED = 10       # km/h
        self.PRESSING_DIST = 20.0    # Meters
        self.CLOSE_DISTANCE = 2.0    # Meters
        
        # MEMORY: Stores previous positions to calculate "Direction"
        self.previous_positions = {} 
        self.decision_timers = {}

    def analyze_behavior(self, player_id, player_data, all_players, ball_pos, team_with_ball):
        """
        Determines behaviors using Vector Math with Improved Pressing Logic.
        """
        # 1. SAFETY CHECK: PLAYER POSITION
        if 'position_transformed' in player_data and player_data['position_transformed'] is not None:
            current_pos = np.array(player_data['position_transformed'])
        else:
            return {"action": "IDLE", "color": (220, 220, 220)}

        # 2. SAFETY CHECK: BALL POSITION
        if ball_pos is None:
            return {"action": "IDLE", "color": (220, 220, 220)}
        
        ball_pos = np.array(ball_pos)

        # 3. EXTRACT METRICS
        speed = player_data.get('speed', 0)
        has_ball = player_data.get('has_ball', False)
        my_team = player_data.get('team', -1)
        
        # Update Memory
        prev_pos = self.previous_positions.get(player_id, current_pos)
        self.previous_positions[player_id] = current_pos

        # 4. CONTEXT CALCULATIONS (Distances)
        dist_to_ball = np.linalg.norm(current_pos - ball_pos)

        nearest_opponent_dist = 999
        nearest_teammate_dist = 999
        
        for other_id, other_data in all_players.items():
            if other_id == player_id: continue
            
            if 'position_transformed' in other_data and other_data['position_transformed'] is not None:
                other_pos = np.array(other_data['position_transformed'])
                dist = np.linalg.norm(current_pos - other_pos)
                
                if other_data['team'] != my_team:
                    if dist < nearest_opponent_dist: nearest_opponent_dist = dist
                else:
                    if dist < nearest_teammate_dist: nearest_teammate_dist = dist

        # --- BEHAVIOR LOGIC ---

        # A. HOLDING (Decision Making)
        if has_ball:
            self.decision_timers[player_id] = self.decision_timers.get(player_id, 0) + 1
            time_held = self.decision_timers[player_id] / 24.0
            label = f"HOLDING ({time_held:.1f}s)"
            color = (0, 255, 0) if time_held < 2.0 else (0, 165, 255) 
            return {"action": label, "color": color}
        else:
            self.decision_timers[player_id] = 0

        # B. IMPROVED PRESSING LOGIC
        # ---------------------------------------------------------
        # Upgrade: Dynamic Speed Threshold.
        # If player is far (>5m), they must SPRINT (>10km/h) to count as pressing.
        # If player is close (<5m), they can slow down to JOCKEY (>5km/h) and still count.
        required_speed = 5.0 if dist_to_ball < 5.0 else self.SPRINT_SPEED

        if speed > required_speed and team_with_ball != -1 and team_with_ball != my_team:
            
            movement_vector = current_pos - prev_pos
            vector_to_ball = ball_pos - current_pos
            
            norm_move = np.linalg.norm(movement_vector)
            norm_ball = np.linalg.norm(vector_to_ball)
            
            if norm_move > 0 and norm_ball > 0:
                movement_dir = movement_vector / norm_move
                ball_dir = vector_to_ball / norm_ball
                
                # Dot Product: Check if facing the ball
                alignment = np.dot(movement_dir, ball_dir)
                
                # Upgrade: "Closing Down" Check
                # Check if they are actually getting closer compared to last frame
                prev_dist_to_ball = np.linalg.norm(prev_pos - ball_pos)
                is_closing_down = dist_to_ball < prev_dist_to_ball

                # COMBINED CONDITION:
                # 1. Aligned towards ball (>0.4) OR extremely close (<2m - chaos/tackle zone)
                # 2. Within pressing range (<20m)
                # 3. Actually getting closer (Closing Down)
                if (alignment > 0.4 or dist_to_ball < 2.0) and \
                   dist_to_ball < self.PRESSING_DIST and \
                   is_closing_down:
                    
                     return {"action": "PRESSING", "color": (0, 0, 255)} # Red

        # C. SPACE CREATION
        if (speed > self.SPRINT_SPEED and 
            nearest_opponent_dist > 5.0 and  
            team_with_ball == my_team):
            return {"action": "SPACE CREATION", "color": (255, 0, 0)} # Blue

        # D. COVERING
        if (speed < 5.0 and 
            nearest_teammate_dist < self.CLOSE_DISTANCE and 
            team_with_ball != my_team):
            return {"action": "COVERING", "color": (0, 255, 255)} # Yellow

        return None