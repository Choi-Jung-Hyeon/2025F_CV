# Part 2 실행 가이드

## 사전 준비

1. **Feature 파일 다운로드 및 압축 해제**
   - `A3_P2_Features.zip` 파일을 다운로드합니다
   - 압축을 해제하여 `./features/` 디렉토리가 생성되도록 합니다
   - 구조:
     ```
     ./features/
       ├── sift/
       │   ├── 0000.sift
       │   ├── 0001.sift
       │   └── ... (총 2000개)
       └── cnn/
           ├── 0000.cnn
           ├── 0001.cnn
           └── ... (총 2000개)
     ```

## 실행 방법

### 1단계: Descriptor 생성 (시간 소요: 약 10-30분)

```bash
python A3_compute_descriptors.py
```

**실행 과정**:
- Codebook 학습 (K=1024 클러스터)
- 2000개 이미지에 대한 descriptor 계산
- `A3_2021313692.des` 파일 생성
- `codebook.pkl` 파일 생성 (재사용 가능)

**출력 예시**:
```
============================================================
Image Retrieval - Descriptor Computation
Student ID: 2021313692
Output file: A3_2021313692.des
Descriptor dimension: 1536 (CNN: 512 + BoW: 1024)
============================================================

Training codebook with K=1024 clusters...
Sampling SIFT features from 200 images...
...
Codebook training complete!
Codebook saved!

Computing descriptors for 2000 images...
Processing images: 100%|██████████| 2000/2000
...

SUCCESS: Descriptor computation complete!
Output file: A3_2021313692.des
File size: 12,304,008 bytes
============================================================
```

### 2단계: 성능 평가 (선택 사항)

Windows에서 제공된 `eval.exe`를 사용하여 점수 확인:

```cmd
eval.exe A3_2021313692.des
```

**예상 출력**:
```
A3_2021313692.des 3.1150 (L1: 2.7135 / L2: 3.1150)
```

- Accuracy 범위: 0~4
- 3.0 이상이면 매우 좋은 성적입니다!

## 알고리즘 요약

**Hybrid Descriptor (1536차원)**:
1. **CNN Features (512차원)**
   - VGG16 Conv5_3 레이어 출력 (14×14×512)
   - Global Average Pooling (GAP)
   - L2 Normalization

2. **SIFT Features (1024차원)**
   - K-Means 클러스터링 (K=1024)
   - Bag of Visual Words (BoW) 히스토그램
   - L2 Normalization

3. **최종 결합**
   - Concatenate: [CNN(512) | BoW(1024)] = 1536차원
   - CNN: 전역적 의미론적 특징 (semantic features)
   - SIFT: 지역적 기하학적 특징 (local geometric features)

## 주의사항

⚠️ **시간 관련**:
- 처음 실행 시 codebook 학습이 포함되어 시간이 오래 걸립니다
- 이후 실행은 `codebook.pkl`을 로드하므로 훨씬 빠릅니다

⚠️ **메모리 관련**:
- RAM 최소 4GB 이상 권장
- 부족하면 `NUM_SAMPLE_IMAGES`를 줄여보세요 (200 → 100)

⚠️ **파일 크기**:
- 생성된 `.des` 파일 크기: 약 12MB
- 최대 허용: 32,768,008 bytes (4096차원 기준)
- 현재 1536차원이므로 여유 있습니다

## 생성되는 파일

1. **A3_2021313692.des** (필수 제출)
   - 2000×1536 descriptor 행렬
   - Binary 포맷 (header: N, D + data)

2. **codebook.pkl** (권장 제출)
   - K-Means 클러스터 중심 (1024×128)
   - 재실행 시 로딩하여 시간 단축

## 문제 해결

### Q1: "No module named 'sklearn'" 에러
```bash
pip install scikit-learn --break-system-packages
```

### Q2: "Memory Error" 발생
- `A3_compute_descriptors.py` 파일에서:
  ```python
  NUM_SAMPLE_IMAGES = 100  # 200 → 100으로 변경
  ```

### Q3: 성능이 낮게 나옴 (< 2.0)
- Feature 파일이 제대로 로드되었는지 확인
- Codebook 학습이 정상적으로 완료되었는지 확인
- 필요 시 `codebook.pkl` 삭제 후 재학습

## 참고 자료

- VGG16: https://arxiv.org/abs/1409.1556
- Bag of Visual Words: Sivic & Zisserman, 2003
- SIFT: Lowe, 2004
