#!/usr/bin/env python

import yaml
import os
import subprocess
import numpy as np
import hashlib
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

def get_project_root():
    """Get the absolute path to the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '..'))

def load_config():
    """Load configuration from config.yaml file."""
    config_path = os.path.join(get_project_root(), 'config.yaml')
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def print_text(text, label):
    """Print text with its classification label."""
    if label == 0:
        print(text, '\nis not spam!')
    else:
        print(text, '\nis spam!')

def confirm_checksum(filename, true_checksum):
    """Verify file integrity using checksum."""
    try:
        with open(filename, 'rb') as f:
            file_data = f.read()
            calculated_checksum = hashlib.sha1(file_data).hexdigest()
        is_valid = calculated_checksum == true_checksum
        if not is_valid:
            print(f'Checksum does not match for {filename}. Please redownload the data.')
        return is_valid
    except Exception as e:
        print(f"Error checking checksum: {e}")
        return False

def get_confusion_matrix(true_labels, pred_labels):
    """Calculate confusion matrix from true and predicted labels."""
    return sk_confusion_matrix(true_labels, pred_labels)

def save_as_csv(df, file_path, filename):
    """Save a DataFrame to a CSV file."""
    os.makedirs(file_path, exist_ok=True)
    
    full_path = os.path.join(file_path, filename)
    df.to_csv(full_path, index=False)
    print(f"DataFrame Saved: {full_path}")

def save_as_npz(file_path, **kwargs):
    """Save data as a NPZ file."""
    np.savez(file_path, **kwargs, allow_pickle=True)
    print(f"Data Saved: {file_path}")