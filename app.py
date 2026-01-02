import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from PIL import Image
import database
import detector
import time

# Page configuration
st.set_page_config(page_title="RIPAS Parking Simulator", page_icon="🚗", layout="wide")

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-image: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
        color: white;
        font-weight: bold;
        border: none;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #4facfe;
    }
    </style>
    """, unsafe_allow_html=True)

# App Navigation
st.sidebar.title("🚔 RIPAS System")
page = st.sidebar.radio("Navigate", ["Dashboard", "Camera 1 (Entry)", "Camera 2 (Exit)"])

# Load detector
@st.cache_resource
def load_ripas_detector():
    return detector.get_detector()

det = load_ripas_detector()

if page == "Dashboard":
    st.title("📊 Parking Management Dashboard")
    
    # Summary Metrics
    records = database.get_all_records()
    total_cars = len(records)
    cars_inside = len([r for r in records if r["Exit Time"] is None])
    total_unpaid = len([r for r in records if r["Paid"] == "No" and r["Exit Time"] is None])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Vehicles", total_cars)
    col2.metric("Currently Inside", cars_inside)
    col3.metric("Pending Payment", total_unpaid)
    
    st.subheader("Vehicle Log")
    if records:
        st.table(records)
        
        st.divider()
        st.subheader("💳 Payment Terminal (Simulator)")
        unpaid_plates = [r["Plate Number"] for r in records if r["Paid"] == "No" and r["Exit Time"] is None]
        if unpaid_plates:
            selected_plate = st.selectbox("Select Plate to Pay", unpaid_plates)
            if st.button("Processing Payment"):
                with st.spinner("Verifying..."):
                    time.sleep(1)
                    database.mark_as_paid(selected_plate)
                    st.success(f"Payment successful for {selected_plate}!")
                    st.rerun()
        else:
            st.info("No unpaid vehicles currently inside.")
        
        st.divider()
        if st.button("🗑️ Clear All Records", type="secondary"):
            database.clear_db()
            st.warning("All records cleared.")
            time.sleep(1)
            st.rerun()
    else:
        st.info("No records found. Start scans at Camera 1.")

elif page == "Camera 1 (Entry)":
    st.title("📹 Camera 1 - Entrance Gate")
    st.write("Scan vehicle at the entry point.")
    
    uploaded_file = st.file_uploader("Upload Image or Video", type=['jpg', 'jpeg', 'png', 'mp4'])
    
    if uploaded_file is not None:
        if "video" in uploaded_file.type:
            st.video(uploaded_file)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_file.getvalue())
                temp_filename = tfile.name
            
            if st.button("🚀 Process Entrance Video"):
                progress_bar = st.progress(0)
                frame_placeholder = st.empty()
                status_box = st.info("Starting scan...")
                
                detected_plates = []
                
                for result in det.process_video(temp_filename):
                    progress_bar.progress(result["progress"])
                    frame_placeholder.image(result["frame"], channels="BGR", use_container_width=True)
                    
                    if result["plate"] != "UNKNOWN":
                        plate = result["plate"]
                        msg = database.record_entry(plate)
                        st.toast(f"Logged Vehicle: {plate}", icon="🚗")
                        detected_plates.append(plate)
                        status_box.info(f"✅ Last Detection: {plate}")
                
                progress_bar.progress(1.0)
                status_box.success(f"Finished! Logged {len(detected_plates)} vehicles.")
                st.balloons()
            os.remove(temp_filename)
        else:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            if img is not None:
                st.image(img, channels="BGR", use_container_width=True)
                if st.button("Scan Plate"):
                    with st.spinner("Processing..."):
                        plate, annotated, conf = det.process_image(img)
                        st.image(annotated, channels="BGR", use_container_width=True)
                        if plate != "UNKNOWN":
                            msg = database.record_entry(plate)
                            st.success(f"Plate: {plate} - {msg}")
            else:
                st.error("Image decode failed.")

elif page == "Camera 2 (Exit)":
    st.title("📹 Camera 2 - Exit Gate")
    st.write("Verify payment before vehicle exit.")
    
    uploaded_file = st.file_uploader("Upload Image or Video", type=['jpg', 'jpeg', 'png', 'mp4'])
    
    if uploaded_file is not None:
        if "video" in uploaded_file.type:
            st.video(uploaded_file)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_file.getvalue())
                temp_filename = tfile.name
            
            if st.button("🚀 Process Exit Video"):
                progress_bar = st.progress(0)
                frame_placeholder = st.empty()
                status_box = st.info("Starting verification...")
                
                processed_count = 0
                for result in det.process_video(temp_filename):
                    progress_bar.progress(result["progress"])
                    frame_placeholder.image(result["frame"], channels="BGR", use_container_width=True)
                    
                    if result["plate"] != "UNKNOWN":
                        plate = result["plate"]
                        success, msg = database.record_exit(plate)
                        processed_count += 1
                        if success:
                            st.toast(f"Gate Open: {plate}", icon="✅")
                            status_box.success(f"✅ {plate}: {msg}")
                        else:
                            st.toast(f"Gate Locked: {plate}", icon="❌")
                            status_box.error(f"❌ {plate}: {msg}")
                            
                progress_bar.progress(1.0)
                status_box.info(f"Finished. Verified {processed_count} vehicles.")
            os.remove(temp_filename)
        else:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            if img is not None:
                st.image(img, channels="BGR", use_container_width=True)
                if st.button("Scan Plate"):
                    with st.spinner("Processing..."):
                        plate, annotated, conf = det.process_image(img)
                        st.image(annotated, channels="BGR", use_container_width=True)
                        if plate != "UNKNOWN":
                            success, msg = database.record_exit(plate)
                            if success: st.success(msg)
                            else: st.error(msg)
            else:
                st.error("Image decode failed.")
