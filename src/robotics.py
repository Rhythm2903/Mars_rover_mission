"""
Mars Rover Mission - Robotics Simulation & Decision Engine
Updated with protocols for all 8 authentic Martian features.
"""

TERRAIN_PROTOCOLS = {
    "Bright dune": {
        "protocol": "TRACTION MODE: Reduce speed, increase tire torque simulation, monitor wheel slip.",
        "speed_modifier": 0.3, "risk_factor": 0.5, "science_yield": 25, "status": "CAUTIOUS_TRAVERSE"
    },
    "Craters": {
        "protocol": "HARD HALT: Emergency reverse, rotate 90 degrees clockwise, execute bypass routing.",
        "speed_modifier": 0.0, "risk_factor": 0.8, "science_yield": 10, "status": "AVOIDANCE_MANEUVER"
    },
    "Dark dune": {
        "protocol": "NORMAL CAUTION: Proceed at half-speed, scan subsurface with radar.",
        "speed_modifier": 0.5, "risk_factor": 0.3, "science_yield": 40, "status": "CAUTIOUS_TRAVERSE"
    },
    "Impact ejecta": {
        "protocol": "SCIENCE STOP: Full halt. Deploy robotic arm, activate APXS spectrometer, cache sample.",
        "speed_modifier": 0.0, "risk_factor": 0.0, "science_yield": 150, "status": "SCIENCE_OPERATIONS"
    },
    "Other": {
        "protocol": "CRUISE MODE: Max operational speed, clear plains ahead.",
        "speed_modifier": 1.0, "risk_factor": 0.0, "science_yield": 0, "status": "NORMAL_DRIVE"
    },
    "Slope streak": {
        "protocol": "SLOPE WARNING: Adjust suspension chassis angle, monitor mass-center shifting against landslide risk.",
        "speed_modifier": 0.4, "risk_factor": 0.6, "science_yield": 60, "status": "CAUTIOUS_TRAVERSE"
    },
    "Spider": {
        "protocol": "ROUGH TERRAIN PROTOCOL: Enable independent 6-wheel steering, navigate trough crevices carefully.",
        "speed_modifier": 0.2, "risk_factor": 0.4, "science_yield": 90, "status": "NORMAL_DRIVE"
    },
    "Swiss cheese": {
        "protocol": "CRITICAL CAVITY HAZARD: High deep-pit risk. Complete reroute around structural void fields.",
        "speed_modifier": 0.0, "risk_factor": 0.9, "science_yield": 20, "status": "AVOIDANCE_MANEUVER"
    }
}

class RoverSimulation:
    def __init__(self, hardware_specs):
        self.speed_factor = hardware_specs["specs"]["speed_factor"]
        self.detection_bonus = hardware_specs["specs"]["detection_bonus"]
        self.total_science_points = 0
        self.accumulated_damage_risk = 0.0
        self.distance_traveled_km = 0.0
        self.current_status = "READY"

    def encounter_terrain(self, classified_terrain):
        if classified_terrain not in TERRAIN_PROTOCOLS:
            return {"error": f"Unknown terrain type: {classified_terrain}"}

        protocol_data = TERRAIN_PROTOCOLS[classified_terrain]
        actual_speed = 5.0 * protocol_data["speed_modifier"] * self.speed_factor
        mitigated_risk = max(0.0, protocol_data["risk_factor"] - self.detection_bonus)
        
        self.current_status = protocol_data["status"]
        self.total_science_points += protocol_data["science_yield"]
        self.accumulated_damage_risk += mitigated_risk
        
        if actual_speed > 0:
            self.distance_traveled_km += (actual_speed * 0.1)

        return {
            "terrain": classified_terrain,
            "status": self.current_status,
            "action_taken": protocol_data["protocol"],
            "current_speed_kmh": round(actual_speed, 2),
            "risk_incurred": round(mitigated_risk, 2)
        }

    def get_telemetry_report(self):
        return {
            "total_science_points": self.total_science_points,
            "accumulated_damage_risk": round(self.accumulated_damage_risk, 2),
            "distance_traveled_km": round(self.distance_traveled_km, 2),
            "rover_health": "NOMINAL" if self.accumulated_damage_risk < 2.0 else "CRITICAL RISK"
        }