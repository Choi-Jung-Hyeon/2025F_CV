"""
A3_Fmat.py - Fundamental Matrix Computation
2021313692 최중현
"""

import numpy as np
import cv2
from compute_avg_reproj_error import compute_avg_reproj_error


def compute_F_raw(M):
    """Basic 8-point algorithm"""
    N = M.shape[0]
    x1, y1, x2, y2 = M[:,0], M[:,1], M[:,2], M[:,3]
    
    # Build constraint matrix A (Af = 0)
    A = np.column_stack([
        x2*x1, x2*y1, x2,
        y2*x1, y2*y1, y2,
        x1, y1, np.ones(N)
    ])
    
    # SVD로 해 구하기
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    
    # Rank-2 강제
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt
    
    return F


def compute_F_norm(M, img_size1=None, img_size2=None):
    """Normalized 8-point algorithm"""
    pts1, pts2 = M[:, :2], M[:, 2:]
    
    # 정규화 행렬 계산
    T1 = _get_norm_matrix(pts1, img_size1)
    T2 = _get_norm_matrix(pts2, img_size2)
    
    # 점들 정규화
    ones = np.ones((len(pts1), 1))
    pts1_h = np.hstack([pts1, ones])
    pts2_h = np.hstack([pts2, ones])
    
    pts1_n = (T1 @ pts1_h.T).T
    pts2_n = (T2 @ pts2_h.T).T
    
    M_norm = np.column_stack([pts1_n[:,:2], pts2_n[:,:2]])
    
    # 정규화된 점으로 F 계산 후 역변환
    F_n = compute_F_raw(M_norm)
    F = T2.T @ F_n @ T1
    F = F / np.linalg.norm(F)
    
    return F


def _get_norm_matrix(pts, img_size):
    """정규화 변환 행렬 생성"""
    if img_size is not None:
        # 이미지 중심 기준 정규화 (과제 요구사항)
        w, h = img_size
        cx, cy = w/2, h/2
        sx, sy = 2/w, 2/h
        T = np.array([[sx, 0, -sx*cx],
                      [0, sy, -sy*cy],
                      [0, 0, 1]])
    else:
        # Hartley 정규화
        cx, cy = np.mean(pts, axis=0)
        centered = pts - [cx, cy]
        dist = np.mean(np.sqrt(np.sum(centered**2, axis=1)))
        s = np.sqrt(2) / max(dist, 1e-10)
        T = np.array([[s, 0, -s*cx],
                      [0, s, -s*cy],
                      [0, 0, 1]])
    return T


def compute_F_mine(M, img_size1=None, img_size2=None):
    """RANSAC 기반 robust estimation"""
    N = M.shape[0]
    best_F, best_cnt = None, 0
    
    for _ in range(2000):
        idx = np.random.choice(N, 8, replace=False)
        try:
            F_cand = compute_F_norm(M[idx], img_size1, img_size2)
        except:
            continue
        
        err = _sampson_error(M, F_cand)
        cnt = np.sum(err < 1.0)
        
        if cnt > best_cnt:
            best_cnt = cnt
            best_F = F_cand
    
    # Inlier로 재추정
    if best_F is not None:
        inliers = _sampson_error(M, best_F) < 1.0
        if np.sum(inliers) >= 8:
            best_F = compute_F_norm(M[inliers], img_size1, img_size2)
    
    return best_F


def _sampson_error(M, F):
    """Sampson error 계산"""
    N = M.shape[0]
    p1 = np.hstack([M[:,:2], np.ones((N,1))])
    p2 = np.hstack([M[:,2:], np.ones((N,1))])
    
    Fp1 = (F @ p1.T).T
    Ftp2 = (F.T @ p2.T).T
    
    num = np.sum(p2 * Fp1, axis=1)**2
    denom = Fp1[:,0]**2 + Fp1[:,1]**2 + Ftp2[:,0]**2 + Ftp2[:,1]**2
    
    return np.sqrt(num / (denom + 1e-10))


def draw_epipolar_line(img, line, color):
    """Epipolar line 그리기"""
    h, w = img.shape[:2]
    a, b, c = line
    pts = []
    
    if abs(b) > 1e-10:
        y = -c/b
        if 0 <= y <= h: pts.append((0, int(y)))
        y = -(a*(w-1)+c)/b
        if 0 <= y <= h: pts.append((w-1, int(y)))
    if abs(a) > 1e-10:
        x = -c/a
        if 0 <= x <= w: pts.append((int(x), 0))
        x = -(b*(h-1)+c)/a
        if 0 <= x <= w: pts.append((int(x), h-1))
    
    pts = list(set(pts))
    if len(pts) >= 2:
        cv2.line(img, pts[0], pts[1], color, 2)


def visualize_epipolar(img1, img2, M, F, title):
    """Epipolar line 시각화 (q 누르면 종료)"""
    colors = [(0,0,255), (0,255,0), (255,0,0)]  # RGB
    
    while True:
        idx = np.random.choice(len(M), 3, replace=False)
        vis1, vis2 = img1.copy(), img2.copy()
        
        for i, j in enumerate(idx):
            x1, y1, x2, y2 = M[j]
            c = colors[i]
            
            # 점 표시
            cv2.circle(vis1, (int(x1), int(y1)), 5, c, -1)
            cv2.circle(vis2, (int(x2), int(y2)), 5, c, -1)
            
            # Epipolar line 계산 및 그리기
            p1 = np.array([x1, y1, 1])
            p2 = np.array([x2, y2, 1])
            draw_epipolar_line(vis2, F @ p1, c)      # l in img2
            draw_epipolar_line(vis1, F.T @ p2, c)    # m in img1
        
        cv2.imshow(title, np.hstack([vis1, vis2]))
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyWindow(title)
            break


def process_pair(img1_path, img2_path, match_path, name):
    """이미지 쌍 처리"""
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    M = np.loadtxt(match_path)
    
    sz1 = (img1.shape[1], img1.shape[0])
    sz2 = (img2.shape[1], img2.shape[0])
    
    F_raw = compute_F_raw(M)
    F_norm = compute_F_norm(M, sz1, sz2)
    F_mine = compute_F_mine(M, sz1, sz2)
    
    print(f"\nAverage Reprojection Errors ({name})")
    print(f"Raw = {compute_avg_reproj_error(M, F_raw)}")
    print(f"Norm = {compute_avg_reproj_error(M, F_norm)}")
    print(f"Mine = {compute_avg_reproj_error(M, F_mine)}")
    
    visualize_epipolar(img1, img2, M, F_mine, f"Epipolar - {name}")


if __name__ == "__main__":
    pairs = [
        ('temple1.png', 'temple2.png', 'temple_matches.txt', 'temple1.png and temple2.png'),
        ('house1.jpg', 'house2.jpg', 'house_matches.txt', 'house1.jpg and house2.jpg'),
        ('library1.jpg', 'library2.jpg', 'library_matches.txt', 'library1.jpg and library2.jpg'),
    ]
    
    for p in pairs:
        process_pair(*p)
