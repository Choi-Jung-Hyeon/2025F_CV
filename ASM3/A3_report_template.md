# Assignment 3 - Part 2 Report Template
# Image Retrieval using Hybrid Descriptors

## 학번: 2021313692

---

## 1. 개요 (Overview)

본 과제에서는 SIFT 특징과 CNN 특징을 결합한 하이브리드 이미지 디스크립터를 설계하고 구현하여, 
2000개의 이미지에 대한 내용 기반 이미지 검색(Content-Based Image Retrieval) 시스템을 구축하였다.

**핵심 아이디어**: 
- CNN 특징은 전역적이고 의미론적인(semantic) 정보를 포착
- SIFT 특징은 지역적이고 기하학적인(geometric) 세부 정보를 포착
- 두 특징을 결합하여 서로의 장점을 활용

---

## 2. 방법론 (Methodology)

### 2.1 전체 구조

최종 디스크립터는 1536차원의 벡터로 구성된다:
- CNN 기반 글로벌 특징: 512차원
- SIFT 기반 로컬 특징: 1024차원

```
Input Image
    ├─→ CNN Features (14×14×512)
    │      └─→ Global Average Pooling → L2 Norm → 512-dim
    │
    └─→ SIFT Features (n×128)
           └─→ K-Means BoW (K=1024) → L2 Norm → 1024-dim

Final Descriptor = [CNN(512) | BoW(1024)] = 1536-dim
```

### 2.2 CNN 특징 추출

**Feature Extraction**:
- Pre-trained VGG16 네트워크의 Conv5_3 레이어 사용
- 출력: 14×14×512 feature map

**Global Average Pooling (GAP)**:
- Spatial 차원(14×14)을 평균하여 channel 차원(512)만 남김
- 위치 불변성(translation invariance) 확보
- 과적합 감소 및 계산 효율성 향상

**정규화**:
```python
gap = np.mean(cnn_feature, axis=(0, 1))  # (512,)
gap = gap / np.linalg.norm(gap)          # L2 normalization
```

### 2.3 SIFT 특징 추출 및 Bag of Visual Words

**K-Means 클러스터링**:
- 200개 샘플 이미지에서 SIFT 특징 수집 (최대 500개/이미지)
- K=1024 클러스터로 visual vocabulary 생성
- MiniBatchKMeans 사용 (효율성)

**BoW 히스토그램 생성**:
```python
# Hard assignment: 각 SIFT 특징을 가장 가까운 클러스터에 할당
labels = kmeans.predict(sift_features)
histogram = np.bincount(labels, minlength=1024)
histogram = histogram / np.linalg.norm(histogram)  # L2 normalization
```

### 2.4 최종 디스크립터

두 특징 벡터를 concatenate하여 최종 1536차원 디스크립터 생성:
```python
descriptor = np.concatenate([cnn_descriptor, bow_descriptor])
```

---

## 3. 구현 세부사항 (Implementation Details)

### 3.1 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| CNN 차원 | 512 | VGG16 Conv5_3 채널 수 |
| BoW 클러스터 수 (K) | 1024 | Visual words 개수 |
| 샘플링 이미지 수 | 200 | Codebook 학습용 |
| 최대 SIFT/이미지 | 500 | 샘플링 시 |
| 총 디스크립터 차원 | 1536 | 512 + 1024 |

### 3.2 성능 최적화

1. **Codebook 사전 계산**: 
   - K-Means 학습 결과를 `codebook.pkl`에 저장
   - 재실행 시 로딩하여 시간 단축

2. **MiniBatchKMeans 사용**:
   - 대용량 SIFT 특징에 대한 효율적인 클러스터링
   - batch_size=10000, max_iter=100

3. **L2 정규화**:
   - 스케일 불변성 확보
   - 거리 기반 유사도 측정에 유리

---

## 4. 결과 및 분석 (Results and Analysis)

### 4.1 검색 성능

eval.exe를 사용한 평가 결과:
- **L1 Distance**: [여기에 결과 입력]
- **L2 Distance**: [여기에 결과 입력]
- **최종 Accuracy**: [여기에 결과 입력] / 4.0

(예시: Accuracy = 3.1150 → 77.9% 정확도)

### 4.2 특징별 기여도 분석

**CNN 특징의 역할**:
- 객체의 카테고리, 색상, 질감 등 high-level semantic 정보 포착
- 전역적 구조 이해에 강점
- 작은 변형에 robust

**SIFT BoW 특징의 역할**:
- 지역적 keypoint 패턴 포착
- 기하학적 배치 정보 보존
- 세밀한 texture 차이 구분

**결합 효과**:
- CNN만 사용: 큰 범주는 잘 맞추나 세부 구분 약함
- SIFT만 사용: 세부 패턴은 좋으나 의미론적 이해 부족
- **하이브리드**: 두 특징의 complementary한 특성 활용

### 4.3 한계점 및 개선 방안

**한계점**:
1. 고정된 K=1024가 모든 이미지에 최적이 아닐 수 있음
2. 단순 concatenation은 두 특징의 중요도를 동등하게 취급
3. 계산 비용이 높음 (특히 첫 실행 시)

**개선 방안**:
1. Weighted concatenation (learned weights)
2. Fisher Vector 또는 VLAD 사용
3. 더 최신 CNN (ResNet, EfficientNet 등)
4. Attention mechanism을 통한 adaptive pooling

---

## 5. 결론 (Conclusion)

본 과제에서는 CNN 기반 전역 특징과 SIFT 기반 지역 특징을 결합한 하이브리드 디스크립터를 
설계 및 구현하였다. 실험 결과, 두 특징의 complementary한 특성이 효과적으로 결합되어 
높은 검색 성능을 달성하였다. 

특히 Global Average Pooling과 Bag of Visual Words 기법을 통해 계산 효율성과 
검색 정확도를 모두 확보할 수 있었다. 향후 더 정교한 특징 결합 방법과 최신 딥러닝 
아키텍처를 적용하면 성능을 더욱 향상시킬 수 있을 것으로 기대된다.

---

## 참고문헌 (References)

1. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv:1409.1556.

2. Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. International Journal of Computer Vision, 60(2), 91-110.

3. Sivic, J., & Zisserman, A. (2003). Video Google: A text retrieval approach to object matching in videos. ICCV.

4. Lin, M., Chen, Q., & Yan, S. (2013). Network in network. arXiv:1312.4400. (GAP 제안)

---

# 작성 가이드

이 템플릿을 워드나 라텍스로 옮겨 작성하세요:
- 분량: 2페이지 권장 (1페이지 초과 ~ 3페이지 미만)
- 그림/도표 추가하면 더 좋습니다
- "4.1 검색 성능" 부분에 실제 eval.exe 결과를 넣으세요
- 필요시 수식 추가 가능 (예: GAP, BoW)
