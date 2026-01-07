"""
A3_compute_descriptors.py - Image Retrieval Descriptor
2021313692 최중현
"""

import numpy as np
import os
import pickle
import struct
from sklearn.cluster import MiniBatchKMeans

STUDENT_ID = "2021313692"
NUM_IMAGES = 2000
K = 1024  # visual words

SIFT_DIR = "./features/sift/"
CNN_DIR = "./features/cnn/"


def load_sift(path):
    """SIFT feature 로드 (n x 128, uint8)"""
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(-1, 128).astype(np.float32)


def load_cnn(path):
    """CNN feature 로드 (14x14x512, float32)"""
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape(14, 14, 512)


def train_codebook():
    """K-Means로 visual vocabulary 학습"""
    print("Codebook 학습 중...")
    np.random.seed(42)
    
    # 200개 이미지에서 SIFT 샘플링
    all_sift = []
    sample_idx = np.random.choice(NUM_IMAGES, 200, replace=False)
    
    for i in sample_idx:
        path = os.path.join(SIFT_DIR, f"{i:04d}.sift")
        if os.path.exists(path):
            sift = load_sift(path)
            if len(sift) > 500:
                sift = sift[np.random.choice(len(sift), 500, replace=False)]
            all_sift.append(sift)
    
    all_sift = np.vstack(all_sift)
    print(f"학습 데이터: {len(all_sift)} features")
    
    kmeans = MiniBatchKMeans(n_clusters=K, random_state=42, 
                             batch_size=10000, max_iter=100, n_init=3)
    kmeans.fit(all_sift)
    return kmeans


def compute_cnn_desc(cnn_feat):
    """CNN descriptor: GAP + L2 norm"""
    gap = np.mean(cnn_feat, axis=(0,1))  # 14x14 평균 -> 512
    norm = np.linalg.norm(gap)
    return gap / norm if norm > 1e-10 else gap


def compute_bow_desc(sift_feat, kmeans):
    """BoW histogram + L2 norm"""
    if len(sift_feat) == 0:
        return np.zeros(K, dtype=np.float32)
    
    labels = kmeans.predict(sift_feat)
    hist = np.bincount(labels, minlength=K).astype(np.float32)
    norm = np.linalg.norm(hist)
    return hist / norm if norm > 1e-10 else hist


def save_descriptors(desc, path):
    """Binary format으로 저장"""
    N, D = desc.shape
    with open(path, 'wb') as f:
        f.write(struct.pack('i', N))
        f.write(struct.pack('i', D))
        desc.astype(np.float32).tofile(f)


def main():
    output_file = f"A3_{STUDENT_ID}.des"
    codebook_file = "codebook.pkl"
    
    # feature 디렉토리 확인
    if not os.path.exists(SIFT_DIR) or not os.path.exists(CNN_DIR):
        print("Error: ./features/ 디렉토리가 없습니다.")
        return
    
    # Codebook 로드 또는 학습
    if os.path.exists(codebook_file):
        print(f"Codebook 로드: {codebook_file}")
        with open(codebook_file, 'rb') as f:
            kmeans = pickle.load(f)
    else:
        kmeans = train_codebook()
        with open(codebook_file, 'wb') as f:
            pickle.dump(kmeans, f)
        print(f"Codebook 저장: {codebook_file}")
    
    # 모든 이미지에 대해 descriptor 계산
    print(f"\n{NUM_IMAGES}개 이미지 처리 중...")
    descriptors = np.zeros((NUM_IMAGES, 512 + K), dtype=np.float32)
    
    for i in range(NUM_IMAGES):
        if i % 200 == 0:
            print(f"  {i}/{NUM_IMAGES}")
        
        cnn = load_cnn(os.path.join(CNN_DIR, f"{i:04d}.cnn"))
        sift = load_sift(os.path.join(SIFT_DIR, f"{i:04d}.sift"))
        
        cnn_desc = compute_cnn_desc(cnn)
        bow_desc = compute_bow_desc(sift, kmeans)
        
        descriptors[i] = np.concatenate([cnn_desc, bow_desc])
    
    # 저장 및 검증
    save_descriptors(descriptors, output_file)
    
    fsize = os.path.getsize(output_file)
    print(f"\n완료: {output_file}")
    print(f"크기: {fsize:,} bytes")
    print(f"차원: {descriptors.shape[1]}")


if __name__ == "__main__":
    main()
