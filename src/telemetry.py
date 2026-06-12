"""
Mars Rover Mission - Telemetry Visualization Engine
Generates 2D path trajectory tracking maps and mission performance dashboards.
"""
import os
import matplotlib.pyplot as plt

def generate_mission_dashboard(path_history, financial_report, telemetry_summary):
    """
    Plots a 2D coordinate map of the rover's journey and saves it as a PNG dashboard.
    """
    # Initialize a clean, dark-themed space telemetry plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract coordinates and details from history
    x_coords = [step["x"] for step in path_history]
    y_coords = [step["y"] for step in path_history]
    
    # Plot the continuous tracks left by the rover wheels
    ax.plot(x_coords, y_coords, color='#d66025', linestyle='--', linewidth=2, alpha=0.8, label='Rover Tracks')
    
    # Custom marker definitions for distinct terrain detections
    marker_mapping = {
        "Craters":       {"marker": "x", "color": "#ff4d4d", "size": 100, "label": "Crater (Avoided)"},
        "Swiss cheese":  {"marker": "H", "color": "#ff3399", "size": 100, "label": "Swiss Cheese Cavity"},
        "Bright dune":   {"marker": "s", "color": "#ffcc00", "size": 80,  "label": "Dune Field (Slow)"},
        "Dark dune":     {"marker": "s", "color": "#cc9900", "size": 80,  "label": "Dark Dune"},
        "Impact ejecta": {"marker": "*", "color": "#33cc33", "size": 150, "label": "Impact Ejecta (Sampled)"},
        "Spider":        {"marker": "p", "color": "#9933ff", "size": 100, "label": "Spider Formation"},
        "Slope streak":  {"marker": "^", "color": "#3399ff", "size": 90,  "label": "Slope Streak"},
        "Other":         {"marker": "o", "color": "#aaaaaa", "size": 50,  "label": "Smooth Plains"}
    }
    
    # Plot each encountered location with its corresponding engineering marker
    tracked_labels = set()
    for step in path_history:
        terrain = step["terrain"]
        cfg = marker_mapping.get(terrain, {"marker": "o", "color": "#ffffff", "size": 50, "label": "Unknown"})
        
        # Avoid duplicate legend entries
        lbl = cfg["label"] if cfg["label"] not in tracked_labels else ""
        if lbl:
            tracked_labels.add(cfg["label"])
            
        ax.scatter(step["x"], step["y"], marker=cfg["marker"], color=cfg["color"], s=cfg["size"], label=lbl, zorder=5)
        
    # Mark landing site (0,0)
    ax.scatter(0, 0, marker="D", color="#00ffcc", s=120, label="Landing Site (Gusev Crater)", zorder=6)

    # Style and annotate the telemetry graph
    ax.set_title("🛰️ AUTONOMOUS ROVER NAVIGATION & HAZARD MAP", fontsize=14, pad=15, fontweight='bold', color='#00ffcc')
    ax.set_xlabel("X-Coordinate Offset (Meters)", fontsize=10, color='#aaaaaa')
    ax.set_ylabel("Y-Coordinate Offset (Meters)", fontsize=10, color='#aaaaaa')
    ax.grid(True, linestyle=':', alpha=0.3, color='#555555')
    
    # Place metadata info box onto the dashboard canvas
    info_text = (
        f"Mission Cost: ${financial_report['total_cost_millions']}M\n"
        f"Science Score: {telemetry_summary['total_science_points']} pts\n"
        f"Distance Run: {telemetry_summary['distance_traveled_km']} km\n"
        f"System Status: {telemetry_summary['rover_health']}"
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='#222222', alpha=0.8, edgecolor='#00ffcc')
    ax.text(0.02, 0.95, info_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
    
    # Position the legend off to the side cleanly
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, edgecolor='#444444')
    
    # Save the output image asset
    output_dir = os.path.join("outputs", "telemetry")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "mission_dashboard.png")
    
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"📊 Visual Mission Telemetry Dashboard generated and saved to: {save_path}")