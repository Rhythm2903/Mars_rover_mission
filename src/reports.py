"""
Mars Rover Mission - Mission Audit Report Exporter
Generates clean markdown executive summaries detailing finance, robotics, and science metrics.
"""
import os

def export_executive_summary(config, budget_report, telemetry, final_roi, steps_history):
    """
    Generates a formal markdown report file tracking the complete mission profile.
    """
    report_dir = os.path.join("outputs", "financial_reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "mission_audit_report.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 🛰️ MARS AUTONOMOUS EXPLORATION MISSION AUDIT RECIPIENT\n")
        f.write("## Aero-Space Systems Engineering & Financial Analysis\n\n")
        f.write("---\n\n")
        
        f.write("### 💰 1. HARDWARE CONFIGURATION & FINANCIAL LOGISTICS\n")
        f.write(f"- **Compute Core:** {config['compute']}\n")
        f.write(f"- **Sensor Array:** {config['sensors']}\n")
        f.write(f"- **Power Architecture:** {config['power']}\n")
        f.write(f"- **Total Vehicle Mass:** {budget_report['total_mass_kg']} kg\n")
        f.write(f"- **Total Mission Cost:** ${budget_report['total_cost_millions']} Million\n\n")
        
        f.write("### 🤖 2. ROBOTICS TELEMETRY & OPERATIONS SUMMARY\n")
        f.write(f"- **Total Distance Traveled:** {telemetry['distance_traveled_km']} km\n")
        f.write(f"- **Accumulated Structural Risk:** {telemetry['accumulated_damage_risk']}\n")
        f.write(f"- **Final Hull Integrity Status:** {telemetry['rover_health']}\n")
        f.write(f"- **Gross Science Yield:** {telemetry['total_science_points']} Discovery Points\n\n")
        
        f.write("### 🚀 3. CROSS-DOMAIN PERFORMANCE METRIC\n")
        f.write(f"> **SCIENTIFIC RETURN ON INVESTMENT (ROI):** {final_roi} pts / $ Millions\n\n")
        
        f.write("---\n\n")
        f.write("### 🧭 4. CHRONOLOGICAL MISSION LOG (AI VISION PATHWAY)\n")
        f.write("| Step | Telemetry Target File | CNN Classified Terrain | Operational Action Protocol |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        
        for step in steps_history:
            f.write(f"| {step['step']} | {step['filename']} | {step['terrain']} | {step['protocol']} |\n")
            
    print(f"📄 Executive Mission Audit Report successfully written to: {report_path}")