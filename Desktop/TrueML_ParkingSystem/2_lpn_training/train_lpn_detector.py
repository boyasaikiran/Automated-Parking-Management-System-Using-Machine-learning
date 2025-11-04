from ultralytics import YOLO
import os
import shutil
import time # Import time to estimate duration

# --- Configuration ---
DATA_YAML_PATH = 'indian_lpn_dataset.yaml' # Path to your dataset configuration file
EPOCHS = 15 # *** FURTHER REDUCED Epochs for maximum speed ***
IMG_SIZE = 640 # Keep standard size initially
MODEL_NAME = 'yolov8n.pt' # *** SWITCHED BACK to Nano model for maximum CPU speed ***
NUM_WORKERS = 6 # Adjusted workers for Ryzen 5 5600H (6 cores)

# --- Output directory and final model name ---
SAVE_PROJECT_DIR = '../'
SAVE_RUN_NAME = 'lpn_training_run_fastest' # New name for this run
FINAL_MODEL_NAME = 'best_lpn_detector_fastest.pt' # New final name
# --------------------------------------------------------

def main():
    print(f"Loading pre-trained YOLO model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    print(f"Starting FASTEST training ({EPOCHS} epochs) on LPN dataset...")
    start_time = time.time() # Record start time

    results = model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        device='cpu', # Force CPU training
        workers=NUM_WORKERS,

        # --- Data Augmentation Parameters (Kept moderate) ---
        degrees=5,      # Slightly reduced rotation
        translate=0.05, # Slightly reduced translation
        scale=0.05,     # Slightly reduced scale
        shear=2,        # Slightly reduced shear
        perspective=0.0, # Disabled perspective for speed
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.5,     # Reduced mosaic probability slightly for speed
        mixup=0.0,      # Disabled mixup for speed
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        # ------------------------------------

        # --- Specify Save Location ---
        project=SAVE_PROJECT_DIR,
        name=SAVE_RUN_NAME,
        exist_ok=True,
        # ---------------------------

        # --- Early Stopping ---
        patience=10 # Stop if no improvement for 10 epochs
        # ---------------------
    )

    end_time = time.time() # Record end time
    training_duration = end_time - start_time
    print(f"Training complete. Total duration: {training_duration / 60:.2f} minutes")

    # --- Move and Rename the Best Model ---
    try:
        run_dir = results.save_dir
        best_model_source_path = os.path.join(run_dir, 'weights', 'best.pt')
        final_model_dest_path = os.path.join(SAVE_PROJECT_DIR, FINAL_MODEL_NAME)

        if os.path.exists(best_model_source_path):
            print(f"Moving best model from: {best_model_source_path}")
            print(f"To destination: {final_model_dest_path}")
            shutil.move(best_model_source_path, final_model_dest_path)
            print(f"\n--- SUCCESS ---")
            print(f"Best model successfully moved and renamed to '{FINAL_MODEL_NAME}' in the root project folder.")
            # Optional: Clean up the temporary run folder
            # print(f"Cleaning up temporary folder: {run_dir}")
            # shutil.rmtree(run_dir)
        else:
            print(f"Error: Could not find the best model at {best_model_source_path}")
            print("Please manually copy 'best.pt' from the latest run folder.")
    except AttributeError:
         print("Error accessing results.save_dir. Ultralytics version might differ.")
         print("Please manually copy 'best.pt' from the latest run folder.")
    except Exception as e:
        print(f"An error occurred while moving the model: {e}")
        print("Please manually copy 'best.pt' from the latest run folder.")
    # -----------------------------------------

if __name__ == '__main__':
    main()