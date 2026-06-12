"""
Mars Rover Mission - Finance & Parametric Costing Module
Handles hardware catalogs, budget configurations, and Science ROI calculations.
"""

# 1. The Hardware Component Catalog (Cost in $ Millions, Power in Watts, Mass in kg)
HARDWARE_CATALOG = {
    "compute": {
        "Standard Rad-Hardened": {"cost": 150, "power": 15, "mass": 5, "cnn_speed_modifier": 1.0},
        "Neuromorphic AI Edge":  {"cost": 350, "power": 45, "mass": 8, "cnn_speed_modifier": 3.0}
    },
    "sensors": {
        "Basic Stereo Navcams":  {"cost": 50,  "power": 10, "mass": 3, "hazard_detection_bonus": 0.0},
        "Advanced LiDAR Suite":  {"cost": 200, "power": 35, "mass": 12, "hazard_detection_bonus": 0.25}
    },
    "power": {
        "Solar Arrays":          {"cost": 80,  "power_generation": 100, "mass": 25, "operational_hours": 10},
        "MMRTG (Nuclear)":       {"cost": 300, "power_generation": 120, "mass": 45, "operational_hours": 24}
    }
}

BASE_CHASSIS_COST = 500  
LAUNCH_COST_PER_KG = 1.2 

def calculate_mission_budget(config):
    total_hardware_cost = BASE_CHASSIS_COST
    total_mass = 150  
    net_power_budget = 0
    operational_hours = 0
    
    comp_choice = config.get("compute")
    sens_choice = config.get("sensors")
    pwr_choice = config.get("power")
    
    # Process Compute
    comp_specs = HARDWARE_CATALOG["compute"][comp_choice]
    total_hardware_cost += comp_specs["cost"]
    total_mass += comp_specs["mass"]
    net_power_budget -= comp_specs["power"]
    
    # Process Sensors
    sens_specs = HARDWARE_CATALOG["sensors"][sens_choice]
    total_hardware_cost += sens_specs["cost"]
    total_mass += sens_specs["mass"]
    net_power_budget -= sens_specs["power"]
    
    # Process Power
    pwr_specs = HARDWARE_CATALOG["power"][pwr_choice]
    total_hardware_cost += pwr_specs["cost"]
    total_mass += pwr_specs["mass"]
    net_power_budget += pwr_specs["power_generation"]
    operational_hours = pwr_specs["operational_hours"]
    
    # Launch Penalty
    launch_cost = total_mass * LAUNCH_COST_PER_KG
    total_mission_cost = total_hardware_cost + launch_cost
    
    return {
        "total_cost_millions": round(total_mission_cost, 2),
        "total_mass_kg": total_mass,
        "net_power_watts": net_power_budget,
        "operational_hours_per_sol": operational_hours,
        "specs": {
            "speed_factor": comp_specs["cnn_speed_modifier"],
            "detection_bonus": sens_specs["hazard_detection_bonus"]
        }
    }

def calculate_science_roi(science_points, total_cost):
    if total_cost <= 0:
        return 0.0
    return round(science_points / total_cost, 4)