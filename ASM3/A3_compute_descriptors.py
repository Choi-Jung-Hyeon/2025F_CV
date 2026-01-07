"""
A3_compute_descriptors.py
Image Retrieval - Hybrid Descriptor Computation

학번: 2021313692

Algorithm:
1. CNN Feature: Global Average Pooling (14x14x512 → 512) + L2 Normalization
2. SIFT Feature: Bag of Visual Words (K=1024) + L2 Normalization
3. Final: Concatenate CNN(512) + BoW(1024) = 1536-dimensional descriptor
"""

import numpy as np
import os
import pickle
import struct
from sklearn.cluster import MiniBatchKMeans

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, desc=None, total=None):
        if desc:
            print(desc)
        return iterable


# ============================================================================
# Configuration
# ============================================================================
STUDENT_ID = "2021313692"
NUM_IMAGES = 2000
SIFT_DIM = 128
CNN_SHAPE = (14, 14, 512)
CNN_DIM = 512
BOW_K = 1024  # Number of visual words
FINAL_DIM = CNN_DIM + BOW_K  # 512 + 1024 = 1536

# [수정된 경로]
# 압축 해제 구조에 맞춰 'features' 폴더를 경로에 포함시켰습니다.
SIFT_DIR = "./A3_P2_Features/features/sift/"
CNN_DIR = "./A3_P2_Features/features/cnn/"

CODEBOOK_FILE = "codebook.pkl"
OUTPUT_FILE = f"A3_{STUDENT_ID}.des"

# For sampling SIFT features for K-Means training
NUM_SAMPLE_IMAGES = 200  # Number of images to sample for codebook training
MAX_SIFT_PER_IMAGE = 500  # Max SIFT features to use per image for training


# ============================================================================
# Feature Loading Functions
# ============================================================================
def load_sift_feature(filepath):
    """
    Load SIFT features from binary file.
    Format: n × 128 unsigned char values (1 byte each)
    
    Returns:
        numpy array of shape (n, 128), dtype=float32
    """
    with open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(-1, SIFT_DIM).astype(np.float32)


def load_cnn_feature(filepath):
    """
    Load CNN features from binary file.
    Format: 14 × 14 × 512 float values (4 bytes each)
    
    Returns:
        numpy array of shape (14, 14, 512), dtype=float32
    """
    with open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape(CNN_SHAPE)


# ============================================================================
# Descriptor Computation Functions
# ============================================================================
def compute_cnn_descriptor(cnn_feature):
    """
    Compute CNN descriptor using Global Average Pooling (GAP).
    
    Input: (14, 14, 512) feature map
    Output: (512,) L2-normalized descriptor
    
    GAP: Average pooling over spatial dimensions, reducing to channel dimension only.
    """
    # Global Average Pooling: average over spatial dimensions (height, width)
    gap = np.mean(cnn_feature, axis=(0, 1))  # (512,)
    
    # L2 Normalization
    norm = np.linalg.norm(gap)
    if norm > 1e-10:
        gap = gap / norm
    
    return gap


def train_codebook(sift_dir, num_images, k=BOW_K):
    """
    Train a visual vocabulary (codebook) using K-Means clustering.
    
    Samples SIFT features from a subset of images to build the codebook.
    Uses MiniBatchKMeans for efficiency.
    """
    print(f"Training codebook with K={k} clusters...")
    print(f"Sampling SIFT features from {NUM_SAMPLE_IMAGES} images...")
    
    # Collect SIFT features from sampled images
    all_sift = []
    
    # Sample image indices (reproducible)
    np.random.seed(42)
    sample_indices = np.random.choice(num_images, min(NUM_SAMPLE_IMAGES, num_images), replace=False)
    
    for idx in tqdm(sample_indices, desc="Sampling SIFT features"):
        filepath = os.path.join(sift_dir, f"{idx:04d}.sift")
        if os.path.exists(filepath):
            sift = load_sift_feature(filepath)
            # Subsample if too many features per image
            if len(sift) > MAX_SIFT_PER_IMAGE:
                indices = np.random.choice(len(sift), MAX_SIFT_PER_IMAGE, replace=False)
                sift = sift[indices]
            all_sift.append(sift)
    
    all_sift = np.vstack(all_sift)
    print(f"Total SIFT features for training: {len(all_sift)}")
    
    # Train MiniBatchKMeans (efficient for large datasets)
    print("Training K-Means clustering...")
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        batch_size=10000,
        max_iter=100,
        n_init=3,
        verbose=0
    )
    kmeans.fit(all_sift)
    
    print("Codebook training complete!")
    return kmeans


def compute_bow_histogram(sift_features, kmeans):
    """
    Compute Bag of Visual Words (BoW) histogram.
    
    Assigns each SIFT feature to the nearest cluster center (visual word)
    and builds a frequency histogram.
    
    Input: (n, 128) SIFT features
    Output: (K,) L2-normalized histogram
    """
    if len(sift_features) == 0:
        return np.zeros(kmeans.n_clusters, dtype=np.float32)
    
    # Find nearest cluster for each SIFT feature (hard assignment)
    labels = kmeans.predict(sift_features)
    
    # Build histogram (count occurrences of each visual word)
    histogram = np.bincount(labels, minlength=kmeans.n_clusters).astype(np.float32)
    
    # L2 Normalization
    norm = np.linalg.norm(histogram)
    if norm > 1e-10:
        histogram = histogram / norm
    
    return histogram


# ============================================================================
# File I/O
# ============================================================================
def save_descriptors(descriptors, filepath):
    """
    Save descriptors in the required binary format.
    
    Format:
        N (int32, 4 bytes): number of images
        D (int32, 4 bytes): descriptor dimension
        data (N × D float32): descriptor values row by row
    """
    num_images, dim = descriptors.shape
    
    with open(filepath, 'wb') as f:
        # Write header: N, D as signed 32-bit integers
        f.write(struct.pack('i', num_images))  # N
        f.write(struct.pack('i', dim))          # D
        
        # Write descriptors as 32-bit floats
        descriptors.astype(np.float32).tofile(f)


def verify_output_file(filepath):
    """
    Verify the output file format and contents.
    """
    print("\nVerifying output file...")
    
    with open(filepath, 'rb') as f:
        # Read header
        N = struct.unpack('i', f.read(4))[0]
        D = struct.unpack('i', f.read(4))[0]
        
        # Read data
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(N, D)
    
    print(f"  N (num images): {N}")
    print(f"  D (dimension): {D}")
    print(f"  Data shape: {data.shape}")
    print(f"  Data dtype: {data.dtype}")
    
    # Check normalization
    cnn_norms = np.linalg.norm(data[:, :CNN_DIM], axis=1)
    bow_norms = np.linalg.norm(data[:, CNN_DIM:], axis=1)
    print(f"  CNN norm range: [{cnn_norms.min():.4f}, {cnn_norms.max():.4f}]")
    print(f"  BoW norm range: [{bow_norms.min():.4f}, {bow_norms.max():.4f}]")
    
    # Verify file size
    expected_size = 4 + 4 + N * D * 4
    actual_size = os.path.getsize(filepath)
    print(f"  Expected file size: {expected_size:,} bytes")
    print(f"  Actual file size: {actual_size:,} bytes")
    print(f"  Size match: {expected_size == actual_size}")
    
    return expected_size == actual_size


# ============================================================================
# Main Processing
# ============================================================================
def compute_all_descriptors():
    """
    Main function to compute hybrid descriptors for all images.
    """
    print("=" * 60)
    print("Image Retrieval - Descriptor Computation")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Descriptor dimension: {FINAL_DIM} (CNN: {CNN_DIM} + BoW: {BOW_K})")
    print("=" * 60)
    
    # Check if feature directories exist
    if not os.path.exists(SIFT_DIR):
        print(f"Error: SIFT directory not found: {SIFT_DIR}")
        print("Please ensure the features are extracted in ./features/sift/")
        return
    if not os.path.exists(CNN_DIR):
        print(f"Error: CNN directory not found: {CNN_DIR}")
        print("Please ensure the features are extracted in ./features/cnn/")
        return
    
    # Step 1: Load or train codebook
    if os.path.exists(CODEBOOK_FILE):
        print(f"\nLoading existing codebook from '{CODEBOOK_FILE}'...")
        with open(CODEBOOK_FILE, 'rb') as f:
            kmeans = pickle.load(f)
        print(f"Codebook loaded! (K={kmeans.n_clusters})")
    else:
        print(f"\nNo existing codebook found. Training new codebook...")
        kmeans = train_codebook(SIFT_DIR, NUM_IMAGES)
        
        # Save codebook for future use
        print(f"Saving codebook to '{CODEBOOK_FILE}'...")
        with open(CODEBOOK_FILE, 'wb') as f:
            pickle.dump(kmeans, f)
        print("Codebook saved!")
    
    # Step 2: Compute descriptors for all images
    print(f"\nComputing descriptors for {NUM_IMAGES} images...")
    all_descriptors = np.zeros((NUM_IMAGES, FINAL_DIM), dtype=np.float32)
    
    for idx in tqdm(range(NUM_IMAGES), desc="Processing images"):
        # File paths
        sift_path = os.path.join(SIFT_DIR, f"{idx:04d}.sift")
        cnn_path = os.path.join(CNN_DIR, f"{idx:04d}.cnn")
        
        # Load and compute CNN descriptor (512-dim)
        cnn_feature = load_cnn_feature(cnn_path)
        cnn_desc = compute_cnn_descriptor(cnn_feature)
        
        # Load and compute SIFT BoW descriptor (1024-dim)
        sift_features = load_sift_feature(sift_path)
        bow_desc = compute_bow_histogram(sift_features, kmeans)
        
        # Concatenate: [CNN(512), BoW(1024)]
        all_descriptors[idx] = np.concatenate([cnn_desc, bow_desc])
    
    # Step 3: Save descriptors to binary file
    print(f"\nSaving descriptors to '{OUTPUT_FILE}'...")
    save_descriptors(all_descriptors, OUTPUT_FILE)
    
    # Step 4: Verify output file
    success = verify_output_file(OUTPUT_FILE)
    
    # Summary
    print("\n" + "=" * 60)
    if success:
        print("SUCCESS: Descriptor computation complete!")
    else:
        print("WARNING: Output file verification failed!")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"File size: {os.path.getsize(OUTPUT_FILE):,} bytes")
    print("=" * 60)


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    compute_all_descriptors()
