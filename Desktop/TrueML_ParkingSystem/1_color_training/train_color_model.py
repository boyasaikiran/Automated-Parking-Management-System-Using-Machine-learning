import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2B0 # Using EfficientNetV2B0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization # Added BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
# Callbacks are simplified as we don't have validation data
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

print("TensorFlow Version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# --- Configuration ---
# Point ONLY to the training directory
BASE_DATA_DIR = 'dataset/car_color_dataset' # Base folder containing 'train'
TRAIN_DIR = os.path.join(BASE_DATA_DIR, 'train')

# Check if training directory exists and is structured correctly
def check_directory(dir_path, dir_name):
    if not os.path.exists(dir_path):
        print(f"ERROR: {dir_name} directory not found at: {dir_path}")
        return False
    try:
        if not any(os.path.isdir(os.path.join(dir_path, i)) for i in os.listdir(dir_path)):
            print(f"ERROR: {dir_name} directory ({dir_path}) does not contain class subdirectories.")
            return False
    except Exception as e:
        print(f"Error checking {dir_name} directory ({dir_path}): {e}")
        return False
    return True

if not check_directory(TRAIN_DIR, "Training"):
    print("Exiting due to missing or incorrectly structured training directory.")
    exit()

# Standard input size for EfficientNetV2B0
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
# You might want fewer epochs if not using early stopping based on validation
EPOCHS = 50 # Reduced epochs slightly, adjust as needed
# Save model in the recommended native Keras format in the root folder
MODEL_SAVE_PATH = '../my_color_model_efficientnet_train_only.keras' # Changed save name
# ---------------------

# --- 1. Setup Data Generator (Training Only) ---

# Apply augmentation to training data
train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalize pixel values
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

# Training data generator
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# Get class information
num_classes = len(train_generator.class_indices)
class_indices = train_generator.class_indices
index_to_class = {v: k for k, v in class_indices.items()}
print(f"Found {train_generator.samples} training images belonging to {num_classes} classes.")
print("Class Indices:", class_indices)

# --- 2. Calculate Class Weights ---
try:
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weight_dict = dict(zip(np.unique(train_generator.classes), class_weights))
    print("Calculated Class Weights:", class_weight_dict)
except Exception as e:
    print(f"Warning: Could not compute class weights. Training without them. Error: {e}")
    class_weight_dict = None


# --- 3. Build the Model (Transfer Learning with EfficientNetV2B0) ---
base_model = EfficientNetV2B0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# --- 4. Compile the Model ---
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n--- Model Summary (EfficientNetV2B0 + Custom Head) ---")
model.summary()

# --- 5. Define Callbacks (Monitoring Training Loss) ---

# Stop training early if training loss doesn't improve enough
# NOTE: Monitoring training loss can lead to stopping too early or overfitting
early_stop = EarlyStopping(
    patience=10,
    monitor='loss', # Monitor training loss
    restore_best_weights=False, # Cannot restore based on training loss effectively
    verbose=1,
    min_delta=0.001 # Require some minimum improvement
)

# Reduce learning rate if training loss plateaus
lr_scheduler = ReduceLROnPlateau(
    monitor='loss', # Monitor training loss
    patience=3,
    verbose=1,
    factor=0.5,
    min_lr=1e-6
)

# Removed ModelCheckpoint to save only at the end
callback_list = [early_stop, lr_scheduler]

# --- 6. Train the Model (No Validation Data) ---
print("\n--- Starting Training (No Validation) ---")
history = None # Initialize history
try:
    # *** Removed steps_per_epoch argument ***
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        callbacks=callback_list,
        class_weight=class_weight_dict
    )
    print("\nTraining completed.")

    # --- 7. Save the Final Model ---
    print(f"Saving final model to {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)
    print("Model saved successfully.")

except Exception as e:
    print(f"\nAn error occurred during training: {e}")

# --- 8. Evaluation and Plotting Removed ---

print("\nScript finished.")