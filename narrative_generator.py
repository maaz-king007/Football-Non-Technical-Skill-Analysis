import google.generativeai as genai
import os

# CONFIGURATION
# Get your free key here: https://aistudio.google.com/
# Make sure your main.py sets this environment variable, or paste it directly here if testing
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyCIzytxB_aRHe9oMmowQEIipH-n3YVMe24"

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def generate_nts_report(player_stats, team_stats):
    """
    Sends tracking data to Gemini to generate a NON-TECHNICAL (Psychological) profile.
    """
    
    # 1. INTERPRET THE DATA AS 'NTS' SIGNALS
    pressure_environment = "High" if team_stats['blocked_passes'] > team_stats['open_passes'] else "Moderate"
    
    # Avoid division by zero
    time_with_ball = player_stats.get('Time with Ball (s)', 0)
    denom = time_with_ball + 1
    work_ratio = player_stats.get('Total Distance (m)', 0) / denom
    
    resilience_tag = "High Selflessness" if work_ratio > 20 else "Balanced"

    # 2. CONSTRUCT THE PSYCHOLOGICAL PROMPT
    prompt = f"""
    You are an expert Sports Psychologist and Performance Analyst.
    Your task is to write a 'Non-Technical Skills (NTS) Profile' for a specific athlete based on their tracking data.

    **STRICT CONSTRAINT:** - Do NOT mention technical skills (e.g., passing, shooting, dribbling, ball control).
    - Focus ONLY on cognitive and behavioral attributes: **Composure, Spatial Awareness, Resilience, Decision Making, and Work Ethic.**

    ### BEHAVIORAL DATA INPUTS
    - **Player ID:** {player_stats['Player ID']} (Archetype: {player_stats.get('Archetype', 'Unknown')})
    - **Cognitive Load / Pressure:** The player operated in a tactical environment where {team_stats['blocked_passes']} passing channels were blocked by opponent 'shadows' (Gray Zones). 
    - **Composure Indicator:** They held possession for {time_with_ball} seconds within this environment.
    - **Resilience / Grit:** They covered {player_stats.get('Total Distance (m)', 0)} meters total, showing a {resilience_tag} work ratio.
    - **Aggression / Focus:** They spent {player_stats.get('Time Pressing (s)', 0)} seconds actively pressing (chasing) opponents.

    ### OUTPUT FORMAT
    Write a 3-sentence psychological assessment:
    1.  **Sentence 1 (Awareness & Positioning):** Analyze their ability to navigate space and recognize the "Shadow Zones" (defensive cover). Did they find pockets of space?
    2.  **Sentence 2 (Composure & Decision Making):** Assess their mental state. Did they rush, or did they show patience under pressure (based on possession time vs pressure)?
    3.  **Sentence 3 (Resilience & Leadership):** Verdict on their work ethic and willingness to work for the team without the ball.

    **Tone:** Clinical, psychological, and professional. Use terms like "Situational Awareness," "Cognitive Processing," and "Mental Stamina."
    """

    # 3. CALL GEMINI MODEL
    try:
        # CHANGED MODEL NAME HERE TO THE STABLE VERSION
        model = genai.GenerativeModel('gemini-2.0-flash') 
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Fallback text if API fails again, so the report doesn't look broken
        return (f"NTS Analysis unavailable (API Error). "
                f"Data indicates {resilience_tag} work ethic and {pressure_environment} cognitive load handling.")