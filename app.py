# Save this as app.py in your main mars_rover_mission folder
import logging
import os
import torch
import random
import time
import sys
import traceback
import pandas as pd  # Leveraged for real-time chart structures
import streamlit as st
from PIL import Image
from torchvision import transforms
from groq import Groq
from dotenv import load_dotenv

logging.basicConfig(
    filename='app_error.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s'
)

# Load environmental variables from the local .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Import your functional physics and financial backends
from src.finance import calculate_mission_budget, calculate_science_roi, HARDWARE_CATALOG
from src.robotics import RoverSimulation
from src.vision import MarsTerrainCNN, LABELS

# Initialize Streamlit Page Config
st.set_page_config(page_title="Mars Rover Mission Control", layout="wide", page_icon="🛰️")

# --- MEMORY-EFFICIENT NEURAL NETWORK CACHE ---
@st.cache_resource
def load_cached_vision_model():
    """Loads the weights file onto the CPU or GPU once to prevent loop timeouts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MarsTerrainCNN(num_classes=8)
    weights_path = os.path.join("outputs", "models", "mars_terrain_cnn.pth")
    
    if os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
            model = model.to(device)
        except Exception as e:
            if device.type == "cuda":
                cpu_device = torch.device("cpu")
                model.load_state_dict(torch.load(weights_path, map_location=cpu_device, weights_only=True))
                model = model.to(cpu_device)
                device = cpu_device
            else:
                raise
        model.eval()
        return model, device
    else:
        return None, device

# Boot the hardware network model safely on launch
with st.spinner("Calibrating onboard AI Vision Core arrays..."):
    AI_MODEL, COMPUTING_DEVICE = load_cached_vision_model()

# --- SESSION STATE INITIALIZATION ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Welcome to JPL Systems Engineering Control. I am your Space Systems Consultant. Let's design your Martian rover. What primary scientific objectives are we targeting for this mission?"}
    ]
if "rover_config" not in st.session_state:
    st.session_state.rover_config = {"compute": "Standard Rad-Hardened", "sensors": "Basic Stereo Navcams", "power": "Solar Arrays"}

# --- SIDEBAR: SYSTEMS CONFIGURATION & PHYSICS PANEL ---
st.sidebar.title("🛠️ Mission Control Panel")

st.sidebar.subheader("🤖 Hardware Manifest")
st.session_state.rover_config["compute"] = st.sidebar.selectbox(
    "Select Compute Brain:", list(HARDWARE_CATALOG["compute"].keys()), index=list(HARDWARE_CATALOG["compute"].keys()).index(st.session_state.rover_config["compute"])
)
st.session_state.rover_config["sensors"] = st.sidebar.selectbox(
    "Select Sensor Suite:", list(HARDWARE_CATALOG["sensors"].keys()), index=list(HARDWARE_CATALOG["sensors"].keys()).index(st.session_state.rover_config["sensors"])
)
st.session_state.rover_config["power"] = st.sidebar.selectbox(
    "Select Power Infrastructure:", list(HARDWARE_CATALOG["power"].keys()), index=list(HARDWARE_CATALOG["power"].keys()).index(st.session_state.rover_config["power"])
)

st.sidebar.markdown("---")
st.sidebar.subheader("🌪️ Environmental Conditions")
dust_storm_severity = st.sidebar.slider("Martian Dust Storm Severity Factor:", min_value=1.0, max_value=2.5, value=1.0, step=0.1)

# Run financial computations in real-time
budget = calculate_mission_budget(st.session_state.rover_config)

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Live Parametric Fiscal Audit")
st.sidebar.metric(label="Total Mission Capital Expenditure", value=f"${budget['total_cost_millions']} M")
st.sidebar.metric(label="Calculated Payload Mass Allocation", value=f"{budget['total_mass_kg']} kg")

adjusted_hours = max(2.0, round(budget['operational_hours_per_sol'] / dust_storm_severity, 1))
st.sidebar.metric(label="Net Operational Window per Sol", value=f"{adjusted_hours} Hours", 
                  delta=f"-{round(budget['operational_hours_per_sol'] - adjusted_hours, 1)}h due to dust" if dust_storm_severity > 1.0 else None, delta_color="inverse")

if GROQ_API_KEY:
    st.sidebar.success("🔑 Groq Key Status: Connected (.env)")
else:
    st.sidebar.error("❌ Groq Key Status: Missing (.env)")

# --- MAIN LAYOUT DIVISION ---
st.title("🛰️ Mars Autonomous Rover Design & Deployment Center")
st.subheader("Systems Engineering Interface powered by Groq LLM & PyTorch Vision")

tab1, tab2 = st.tabs(["💬 Systems Consultation Chat Agent", "🧭 Live Autonomous Simulation Run"])

# --- TAB 1: AI SYSTEMS ENGINEERING CONSULTANT ---
with tab1:
    st.markdown("### Discuss Architecture Constraints with the Lead Aerospace Agent")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_prompt := st.chat_input("Ask about parts, request cost optimizations, or ask for explanations..."):
        with st.chat_message("user"):
            st.markdown(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        if not GROQ_API_KEY:
            st.error("Inference halted. The GROQ_API_KEY variable is missing from your local .env file.")
        else:
            try:
                client = Groq(api_key=GROQ_API_KEY)
                system_instruction = (
                    f"You are a NASA JPL principal aerospace systems design engineer. You are actively assessing a design with "
                    f"a compute core of {st.session_state.rover_config['compute']}, sensor layout of {st.session_state.rover_config['sensors']}, "
                    f"and powered via {st.session_state.rover_config['power']}. The user has dialed the planetary Dust Storm Severity "
                    f"multiplier to {dust_storm_severity}x. Guide the user in building and safety-testing their vehicle under these parameters."
                )
                messages_payload = [{"role": "system", "content": system_instruction}] + [
                    {"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history
                ]
                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=messages_payload,
                    temperature=0.7
                )
                agent_response = completion.choices[0].message.content
                with st.chat_message("assistant"):
                    st.markdown(agent_response)
                st.session_state.chat_history.append({"role": "assistant", "content": agent_response})
            except Exception as e:
                st.error(f"Inference Engine Connection Error: {e}")


# --- TAB 2: LIVE VISION SIMULATION ENGINE ---
with tab2:
    st.markdown("### 🖥️ Mission Control Telemetry Screen")
    
    if AI_MODEL is None:
        st.error("❌ Critical System Error: Onboard model weights (.pth) not found inside 'outputs/models/'. App execution halted.")
    else:
        launch_btn = st.button("🚀 Authorize Launch Sequence & Begin Exploration")
        
        # Sticky Real-time Stats Header Counters
        metric_cols = st.columns(4)
        m_dist = metric_cols[0].empty()
        m_science = metric_cols[1].empty()
        m_risk = metric_cols[2].empty()
        m_roi = metric_cols[3].empty()
        
        m_dist.metric("Total Traveled Distance", "0.0 km")
        m_science.metric("Gross Science Points", "0 pts")
        m_risk.metric("Hull Damage Accrued", "0.0")
        m_roi.metric("Mission Return Ratio (ROI)", "0.0")
        
        st.markdown("---")
        
        # UPPER HALF: Live Simulation Tracking Monitors
        display_cols = st.columns([1, 1.2])
        with display_cols[0]:
            st.subheader("📹 Live Camera Feed & AI Analysis")
            camera_placeholder = st.empty()
            status_placeholder = st.empty()
            
        with display_cols[1]:
            # FIX COMPLETE: Replaced plain text log box with an interactive 2D Radar chart frame
            st.subheader("🗺️ Real-Time Radar Trajectory Tracking Map")
            map_placeholder = st.empty()

        st.markdown("---")
        
        # MIDDLE SECTION: Real-time Analytics Graph Streams
        st.subheader("📈 Real-Time Data Stream Logs")
        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.markdown("**Cumulative Science Yield Profile (pts)**")
            science_chart_placeholder = st.empty()
        with chart_cols[1]:
            st.markdown("**Rover Battery Charge / Power Consumption Profile (%)**")
            power_chart_placeholder = st.empty()

        st.markdown("---")
        
        # LOWER HALF: Historical Grid Image Gallery Slots
        st.subheader("📸 Mission Capture Gallery (Historical Frame Archive)")
        row1_cols = st.columns(4)
        row2_cols = st.columns(4)
        gallery_slots = row1_cols + row2_cols

        st.markdown("---")
        export_placeholder = st.empty()
        diagnostic_box = st.empty()

        if launch_btn:
            with st.spinner("Executing launch sequence... this may take a few seconds"):
                try:
                    rover = RoverSimulation(budget)
                    base_data_dir = os.path.join("data", "raw")
                    
                    if not os.path.exists(base_data_dir):
                        raise FileNotFoundError(f"The raw data folder could not be located at path: {base_data_dir}")

                    image_pool = []
                    for label in LABELS:
                        folder = os.path.join(base_data_dir, label)
                        if os.path.exists(folder):
                            images = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                            image_pool.extend(images[:3])
                    
                    if len(image_pool) == 0:
                        raise ValueError(f"No valid frames found inside sub-folders under: {base_data_dir}")

                    random.shuffle(image_pool)
                    mission_steps = image_pool[:8]
                    
                    transform_pipeline = transforms.Compose([
                        transforms.Resize((128, 128)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    ])
                    
                    current_x, current_y = 0, 0
                    science_history = [0]
                    battery_history = [100.0]
                    report_text_log = "### 🧭 CHRONOLOGICAL AUTONOMOUS TELEMETRY REPORT\n\n"
                    
                    # Initialize spatial Pandas dataframe structure starting at landing pad (0,0)
                    coordinate_tracking_df = pd.DataFrame([{"X-Coordinate Offset": 0, "Y-Coordinate Offset": 0, "Telemetry Indicator": "Landing Pad"}])
                    map_placeholder.scatter_chart(
                        coordinate_tracking_df, x="X-Coordinate Offset", y="Y-Coordinate Offset", color="Telemetry Indicator", height=320
                    )

                    for step, img_path in enumerate(mission_steps, 1):
                        if not os.path.exists(img_path):
                            raise FileNotFoundError(f"Missing image step target during processing link: {img_path}")
                        
                        raw_img = Image.open(img_path)
                        if raw_img.mode != 'RGB':
                            raw_img = raw_img.convert('RGB')
                        camera_placeholder.image(raw_img, caption=f"Scanning Telemetry Asset: {os.path.basename(img_path)}", width=640)

                        input_tensor = transform_pipeline(raw_img)
                        input_tensor = input_tensor.unsqueeze(0).to(COMPUTING_DEVICE)

                        with torch.no_grad():
                            logits = AI_MODEL(input_tensor)
                            _, pred_idx = torch.max(logits, 1)
                            prediction_string = LABELS[pred_idx.item()]

                        action = rover.encounter_terrain(prediction_string)
                        action["risk_incurred"] = round(action["risk_incurred"] * dust_storm_severity, 2)
                        rover.accumulated_damage_risk = round(rover.accumulated_damage_risk + (action["risk_incurred"] * 0.1), 2)

                        telemetry = rover.get_telemetry_report()

                        power_drain_rate = 4.5 * dust_storm_severity
                        if "Neuromorphic" in st.session_state.rover_config["compute"]:
                            power_drain_rate *= 0.5
                        current_battery = max(0.0, round(100.0 - (step * power_drain_rate), 1))

                        science_history.append(telemetry["total_science_points"])
                        battery_history.append(current_battery)

                        with status_placeholder.container():
                            st.markdown(f"**Active Frame Classification:** `{prediction_string}`")
                            if "HARD HALT" in action["action_taken"] or "CRITICAL" in action["action_taken"]:
                                st.error(f"🚨 **EMERGENCY RESPONSE PROTOCOL:**\n\n{action['action_taken']}")
                            elif "TRACTION" in action["action_taken"] or "WARNING" in action["action_taken"]:
                                st.warning(f"⚠️ **HAZARD MITIGATION MODE:**\n\n{action['action_taken']}")
                            else:
                                st.success(f"✅ **NOMINAL OPERATIONS:**\n\n{action['action_taken']}")

                        final_roi = calculate_science_roi(telemetry["total_science_points"], budget["total_cost_millions"])
                        m_dist.metric("Total Traveled Distance", f"{telemetry['distance_traveled_km']} km")
                        m_science.metric("Gross Science Points", f"{telemetry['total_science_points']} pts")
                        m_risk.metric("Hull Damage Accrued", f"{telemetry['accumulated_damage_risk']}")
                        m_roi.metric("Mission Return Ratio (ROI)", f"{final_roi} pts/$M")

                        if action["current_speed_kmh"] > 0:
                            current_x += random.choice([60, 100, 140])
                            current_y += random.choice([40, 90, 130])
                        else:
                            current_x += random.choice([-100, -60])
                            current_y += random.choice([10, 25])
                        
                        # Append the fresh positional coordinate step directly into the dataframe layout
                        fresh_point = pd.DataFrame([{"X-Coordinate Offset": current_x, "Y-Coordinate Offset": current_y, "Telemetry Indicator": f"Step {step}: {prediction_string}"}])
                        coordinate_tracking_df = pd.concat([coordinate_tracking_df, fresh_point], ignore_index=True)
                        
                        # Animate vector coordinates step-by-step
                        map_placeholder.scatter_chart(
                            coordinate_tracking_df, x="X-Coordinate Offset", y="Y-Coordinate Offset", color="Telemetry Indicator", height=320
                        )

                        report_text_log += f"**Step {step}** | File: {os.path.basename(img_path)} | Detected: {prediction_string} | Action: {action['action_taken']}\n\n"

                        science_chart_placeholder.line_chart(science_history, height=180)
                        power_chart_placeholder.line_chart(battery_history, height=180)

                        progress_value = int(step / len(mission_steps) * 100)
                        st.progress(progress_value)

                        with gallery_slots[step - 1]:
                            st.image(raw_img, caption=f"Frame {step} Archive", width=280)
                            status_text = (
                                f"**🚩 Step {step}/8**\n\n"
                                f"**AI Vision:** `{prediction_string}`\n\n"
                                f"**Maneuver:**\n{action['action_taken']}\n\n"
                                f"**Risk:** {action['risk_incurred']}"
                            )
                            if "HARD HALT" in action["action_taken"] or "CRITICAL" in action["action_taken"]:
                                st.error(status_text)
                            elif "TRACTION" in action["action_taken"] or "WARNING" in action["action_taken"]:
                                st.warning(status_text)
                            else:
                                st.success(status_text)

                        time.sleep(1.0)
                    
                    final_manifest_content = (
                        f"# MARS MISSION TELEMETRY AUDIT REPORT\n"
                        f"Configuration: {st.session_state.rover_config}\n"
                        f"Financial Cost: ${budget['total_cost_millions']} Million\n"
                        f"Distance Closed: {telemetry['distance_traveled_km']} km\n"
                        f"Science Score: {telemetry['total_science_points']} pts\n"
                        f"Final Efficiency Ratio ROI: {final_roi} pts/$M\n\n"
                        f"{report_text_log}"
                    )

                    st.success("🏁 Exploration run successfully concluded! Final telemetry sheets compiled.")

                    with export_placeholder.container():
                        st.subheader("📥 Export Final Flight Manifest Sheets")
                        st.download_button(
                            label="💾 Download Official Mission Audit Manifest (.txt)",
                            data=final_manifest_content,
                            file_name="mars_rover_flight_manifest.txt",
                            mime="text/plain"
                        )

                except Exception as e:
                    with diagnostic_box.container():
                        st.error(f"### ⚙️ Pipeline Error Detected\n`{str(e)}`")
                        st.code(traceback.format_exc(), language="python")
