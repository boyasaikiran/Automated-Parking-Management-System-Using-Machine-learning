import os
import json
import datetime
import cv2
import numpy as np
import easyocr
import random
import pandas as pd
import streamlit as st  # Added Streamlit
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from ultralytics import YOLO
from PIL import Image

# --- Configuration (Paths are now relative to the ROOT folder) ---
DB_FILE = '3_main_application/parking_db.json'
TOTAL_SPOTS = 100
PARKING_RATE_PER_HOUR = 2.50
IMAGE_FOLDER = '3_main_application/car_images'
COLOR_MODEL_PATH = 'my_color_model.h5'         # Correct path from root
LPN_MODEL_PATH = 'best_lpn_detector.pt'      # Correct path from root
TEMP_UPLOAD_FOLDER = 'temp_uploads'

# Create temp folder if it doesn't exist
os.makedirs(TEMP_UPLOAD_FOLDER, exist_ok=True)
# ---------------------

# --- 1. Load All Models (Cached for Streamlit) ---
@st.cache_resource
def load_color_model():
    print("Loading Color Classifier...")
    try:
        model = load_model(COLOR_MODEL_PATH)
        print("Color Classifier loaded.")
        return model
    except Exception as e:
        st.error(f"FATAL: Could not load Color Model from {COLOR_MODEL_PATH}. {e}")
        st.stop() # Stop the app if model loading fails

@st.cache_resource
def load_lpn_detector():
    print("Loading LPN Detector...")
    try:
        model = YOLO(LPN_MODEL_PATH)
        print("LPN Detector loaded.")
        return model
    except Exception as e:
        st.error(f"FATAL: Could not load LPN Model from {LPN_MODEL_PATH}. {e}")
        st.stop()

@st.cache_resource
def load_ocr_reader():
    print("Loading EasyOCR Reader...")
    try:
        # Ensure CPU is used if no GPU available
        reader = easyocr.Reader(['en'], gpu=False) 
        print("EasyOCR Reader loaded.")
        return reader
    except Exception as e:
        st.error(f"FATAL: Could not load EasyOCR. {e}")
        st.stop()

# Load models when the script runs
color_model = load_color_model()
lpn_detector_model = load_lpn_detector()
ocr_reader = load_ocr_reader()
COLOR_CLASSES = ['Black', 'Blue', 'Brown', 'Green', 'Grey', 'Orange', 'Red', 'Silver', 'Violet', 'White', 'Yellow']


# --- 2. Database Functions ---
def initialize_database():
    if not os.path.exists(DB_FILE):
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(DB_FILE), exist_ok=True) 
        db_data = {"total_spots": TOTAL_SPOTS, "occupied_spots": {}}
        write_db(db_data)

def read_db():
    try:
        with open(DB_FILE, 'r') as f:
            # Handle empty file case
            content = f.read()
            if not content:
                initialize_database()
                return {"total_spots": TOTAL_SPOTS, "occupied_spots": {}}
            return json.loads(content)
    except FileNotFoundError:
        initialize_database()
        return read_db()
    except json.JSONDecodeError:
        st.error(f"Error reading {DB_FILE}. It might be corrupted. Initializing.")
        initialize_database()
        return {"total_spots": TOTAL_SPOTS, "occupied_spots": {}}


def write_db(data):
     # Ensure parent directory exists before writing
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    with open(DB_FILE, 'w') as f:
        json.dump(data, f, indent=2)

# --- 3. ML Processing Functions (Modified for Streamlit output) ---

def get_car_color(image_path):
    """Predicts car color using our trained Keras model."""
    try:
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predictions = color_model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_color = COLOR_CLASSES[predicted_index]
        confidence = np.max(predictions[0])

        st.write(f"**Color Prediction:** {predicted_color} (Confidence: {confidence:.2f})")
        return predicted_color
    except Exception as e:
        st.error(f"Error in color prediction: {e}")
        return "Unknown"

def get_license_plate(image_path):
    """Finds LPN using YOLO, then reads with EasyOCR."""
    try:
        cv_img = cv2.imread(image_path)
        if cv_img is None:
            st.error(f"Could not read image from path: {image_path}")
            return "ERROR"

        # Use CPU for YOLO inference
        results = lpn_detector_model(image_path, device='cpu') 

        best_lpn = "NOT_FOUND"

        for result in results:
            if len(result.boxes) == 0:
                st.write("LPN Detector found no plates.")
                continue

            box = result.boxes[0]
            coords = box.xyxy[0].cpu().numpy().astype(int)
            confidence = box.conf[0].cpu().numpy()

            st.write(f"**LPN Detector:** Found plate (Confidence: {confidence:.2f})")

            x1, y1, x2, y2 = coords
            # Add padding, ensuring coordinates stay within image bounds
            pad = 5
            h, w = cv_img.shape[:2]
            crop_y1 = max(0, y1 - pad)
            crop_y2 = min(h, y2 + pad)
            crop_x1 = max(0, x1 - pad)
            crop_x2 = min(w, x2 + pad)
            
            # Check if crop dimensions are valid
            if crop_y1 >= crop_y2 or crop_x1 >= crop_x2:
                 st.warning("Detected plate bounding box is invalid or too small after padding.")
                 continue

            cropped_plate = cv_img[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Use EasyOCR
            ocr_results = ocr_reader.readtext(cropped_plate, detail=0)

            if not ocr_results:
                st.write("OCR could not read text from the cropped plate.")
                continue

            raw_text = " ".join(ocr_results)
            cleaned_lpn = "".join(e for e in raw_text if e.isalnum()).upper()

            st.write(f"**OCR Result:** {raw_text}  ->  **Cleaned LPN:** {cleaned_lpn}")
            best_lpn = cleaned_lpn
            break

        return best_lpn

    except Exception as e:
        st.error(f"Error in LPN processing: {e}")
        return "ERROR"


# --- 4. Streamlit App Interface ---

st.set_page_config(page_title="TrueML Parking Manager", layout="wide")
st.title("TrueML Parking Manager 🚗")

# Initialize database
initialize_database()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Check-In Car", "Check-Out Car", "View Occupancy"])

# --- Main Page Content based on Navigation ---

if page == "Check-In Car":
    st.header("Car Check-In")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "png", "jpeg"], key="uploader")
        if uploaded_file is not None:
            temp_file_path = os.path.join(TEMP_UPLOAD_FOLDER, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.image_to_process = temp_file_path
            st.session_state.source_type = "upload" # Track source


    with col2:
        st.write("Or test with a random image:")
        if st.button("Load Random Image from Test Folder"):
            try:
                if not os.path.exists(IMAGE_FOLDER):
                    st.error(f"Test image folder not found at: {IMAGE_FOLDER}")
                else:
                    test_images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('jpg', 'png', 'jpeg'))]
                    if test_images:
                        random_image_name = random.choice(test_images)
                        image_path = os.path.join(IMAGE_FOLDER, random_image_name)
                        st.session_state.image_to_process = image_path
                        st.session_state.source_type = "random" # Track source
                        st.rerun() # Rerun to display the loaded image
                    else:
                        st.warning(f"No test images found in {IMAGE_FOLDER}")
            except Exception as e:
                 st.error(f"Error loading random image: {e}")


    # --- Process the selected image ---
    if "image_to_process" in st.session_state and st.session_state.image_to_process:
        image_path = st.session_state.image_to_process

        try:
            st.image(image_path, caption="Image to Check-In", use_container_width=True)

            if st.button("Confirm Check-In", key="confirm_checkin"):
                with st.spinner("Processing image with ML models..."):
                    db = read_db()

                    if len(db.get('occupied_spots', {})) >= db.get('total_spots', TOTAL_SPOTS):
                        st.error("Sorry, the parking lot is full.")
                    else:
                        color = get_car_color(image_path)
                        lpn = get_license_plate(image_path)

                        if lpn == "NOT_FOUND" or lpn == "ERROR":
                            st.error("Could not read license plate. Please try a clearer image or check model.")
                        else:
                            occupied_spots = db.get('occupied_spots', {})
                            already_parked = False
                            for car in occupied_spots.values():
                                if car.get('lpn') == lpn:
                                    st.error(f"ERROR: Car with LPN {lpn} is already parked.")
                                    already_parked = True
                                    break

                            if not already_parked:
                                spot_num = -1
                                total_spots = db.get('total_spots', TOTAL_SPOTS)
                                for i in range(1, total_spots + 1):
                                    if str(i) not in occupied_spots:
                                        spot_num = str(i)
                                        break

                                if spot_num == -1: # Should not happen if check above works
                                    st.error("Error: Could not find an available spot, but lot is not full.")
                                else:
                                    entry_time = datetime.datetime.now().isoformat()
                                    car_data = {"lpn": lpn, "color": color, "entry_time": entry_time}
                                    db['occupied_spots'][spot_num] = car_data
                                    write_db(db)

                                    st.success(f"Car {color} {lpn} checked in!")
                                    st.balloons()
                                    st.json({
                                        "Assigned Spot": spot_num,
                                        "LPN": lpn,
                                        "Color": color,
                                        "Entry Time": entry_time
                                    })
                # Clean up logic after button press
                # Check if it was an uploaded file before deleting
                source = st.session_state.get("source_type", "unknown")
                if source == "upload" and os.path.exists(image_path):
                     try:
                         os.remove(image_path)
                         print(f"Removed temp file: {image_path}")
                     except Exception as e:
                         print(f"Error removing temp file {image_path}: {e}")

                # Clear session state regardless of source
                del st.session_state.image_to_process
                if "source_type" in st.session_state:
                    del st.session_state.source_type
                st.rerun() # Rerun to clear the image and button state


        except Exception as e:
            st.error(f"An error occurred: {e}")
            if "image_to_process" in st.session_state:
                 del st.session_state.image_to_process # Clear state on error too


elif page == "Check-Out Car":
    st.header("Car Check-Out")

    db = read_db()
    occupied_spots = db.get('occupied_spots', {})
    occupied_lpns = [data.get('lpn', 'N/A') for data in occupied_spots.values()]

    if not occupied_lpns:
        st.info("Parking lot is currently empty. No cars to check out.")
    else:
        lpn_to_exit = st.selectbox("Select License Plate Number to Check-Out:", options=sorted(list(set(occupied_lpns)))) # Ensure unique LPNs

        if st.button("Check-Out Vehicle", key="checkout_button"):
            found_spot = None
            car_data = None

            for spot, data in occupied_spots.items():
                if data.get('lpn') == lpn_to_exit:
                    found_spot = spot
                    car_data = data
                    break

            if found_spot is None:
                st.error(f"ERROR: Car with LPN {lpn_to_exit} not found (this shouldn't happen with selectbox).")
            else:
                try:
                    entry_time = datetime.datetime.fromisoformat(car_data.get('entry_time', ''))
                    exit_time = datetime.datetime.now()
                    duration = exit_time - entry_time
                    duration_hours = max(0, duration.total_seconds() / 3600) # Ensure non-negative
                    fee = duration_hours * PARKING_RATE_PER_HOUR

                    del db['occupied_spots'][found_spot]
                    write_db(db)

                    st.success(f"SUCCESS: Car {lpn_to_exit} Checked Out")
                    st.metric(label="Total Parking Fee", value=f"${fee:.2f}")
                    st.json({
                        "Spot": found_spot,
                        "Car": f"{car_data.get('color', 'N/A')} {car_data.get('lpn', 'N/A')}",
                        "Duration": f"{duration_hours:.2f} hours"
                    })
                    st.rerun() # Rerun to update the selectbox

                except ValueError:
                    st.error("Error calculating fee: Invalid entry time format in database.")
                except Exception as e:
                    st.error(f"An unexpected error occurred during check-out: {e}")


elif page == "View Occupancy":
    st.header("Current Parking Occupancy")

    db = read_db()
    occupied_spots = db.get('occupied_spots', {})
    total_spots = db.get('total_spots', TOTAL_SPOTS)

    st.metric("Occupied Spots", f"{len(occupied_spots)} / {total_spots}")

    if not occupied_spots:
        st.info("The parking lot is empty.")
    else:
        data_for_df = []
        for spot, data in occupied_spots.items():
            # Add spot number and handle potentially missing keys gracefully
            entry = {
                'spot': spot,
                'lpn': data.get('lpn', 'N/A'),
                'color': data.get('color', 'N/A'),
                'entry_time': data.get('entry_time', 'N/A')
            }
            data_for_df.append(entry)

        df = pd.DataFrame(data_for_df)
        df = df[['spot', 'lpn', 'color', 'entry_time']] # Ensure column order
        st.dataframe(df, use_container_width=True)

# --- End of Streamlit App ---
# The original main() function and if __name__ == "__main__": block are removed
# as Streamlit runs the script from top to bottom.