import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

def load_eog_data(file_path):
    """
    Assumes .mat file has:
    'data': (trials, time_points, channels) e.g. (800, 2000, 2)
    'labels_speed': (800, 1) -> 5, 6, ..., 20
    'labels_dir': (800, 1) -> 0 for Left, 1 for Right
    """
    mat = scipy.io.load_mat(file_path)
    X = mat['data'] 
    y_speed = mat['labels_speed'] - 5 # Shift to 0-15 range
    y_dir = mat['labels_dir']
    return X, y_speed, y_dir

# 1. Setup Architecture (Hybrid CNN-LSTM)
def build_hybrid_model(input_shape, num_speeds):
    model = models.Sequential([
        # CNN layers to extract spatial features (slopes/peaks)
        layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        # LSTM layers to capture temporal velocity
        layers.LSTM(100, return_sequences=False),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer (Multi-class for speed)
        layers.Dense(num_speeds, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# 2. Execution
# Replace with your actual path
X, y_speed, y_dir = load_eog_data('my_eog_study.mat')

# Split data (ensure stratified split to keep speed balance)
X_train, X_test, y_train, y_test = train_test_split(X, y_speed, test_size=0.2, stratify=y_speed)

model = build_hybrid_model(input_shape=(X.shape[1], X.shape[2]), num_speeds=16)
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy for Speed Classification: {test_acc:.2f}")