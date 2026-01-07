"""
A3_Fmat.py
Fundamental Matrix Computation and Epipolar Line Visualization

학번: 2021313692

Part 1-1: Fundamental Matrix Computation
- compute_F_raw: 8-point algorithm without normalization
- compute_F_norm: 8-point algorithm with normalization
- compute_F_mine: RANSAC-based robust estimation

Part 1-2: Epipolar Line Visualization
"""

import numpy as np
import cv2
import os
import sys

# Import the provided error computation function
sys.path.append('./A3_P1_Data')
from compute_avg_reproj_error import compute_avg_reproj_error


# ============================================================================
# Part 1-1: Fundamental Matrix Computation
# ============================================================================

def compute_F_raw(M):
    """
    Compute fundamental matrix using the basic 8-point algorithm.
    
    Args:
        M: (N, 4) array of correspondences. Each row is (x1, y1, x2, y2)
    
    Returns:
        F: (3, 3) fundamental matrix
    """
    N = M.shape[0]
    
    # Extract points
    x1, y1 = M[:, 0], M[:, 1]  # Points in image 1
    x2, y2 = M[:, 2], M[:, 3]  # Points in image 2
    
    # Build the constraint matrix A
    # Each correspondence gives one equation: x2'*F*x1 = 0
    # [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1] * [f11, f12, ..., f33].T = 0
    A = np.column_stack([
        x2 * x1, x2 * y1, x2,
        y2 * x1, y2 * y1, y2,
        x1, y1, np.ones(N)
    ])
    
    # Solve using SVD: A*f = 0
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    
    # Enforce rank-2 constraint
    U_f, S_f, Vt_f = np.linalg.svd(F)
    S_f[2] = 0  # Set smallest singular value to 0
    F = U_f @ np.diag(S_f) @ Vt_f
    
    return F


def compute_F_norm(M, img_size1=None, img_size2=None):
    """
    Compute fundamental matrix using normalized 8-point algorithm.
    
    Normalization (as per assignment requirement):
    - Translation: move image center to origin (0, 0)
    - Scaling: fit image into unit square [(-1, -1), (+1, +1)]
    
    Args:
        M: (N, 4) array of correspondences. Each row is (x1, y1, x2, y2)
        img_size1, img_size2: (width, height) of images (optional)
    
    Returns:
        F: (3, 3) fundamental matrix (in original coordinates)
    """
    # Extract points
    pts1 = M[:, 0:2]  # (N, 2)
    pts2 = M[:, 2:4]  # (N, 2)
    
    # Compute normalization transformation for each image set
    T1 = get_normalization_matrix(pts1, img_size1)
    T2 = get_normalization_matrix(pts2, img_size2)
    
    # Normalize points
    pts1_h = np.column_stack([pts1, np.ones(len(pts1))])  # (N, 3)
    pts2_h = np.column_stack([pts2, np.ones(len(pts2))])  # (N, 3)
    
    pts1_norm = (T1 @ pts1_h.T).T  # (N, 3)
    pts2_norm = (T2 @ pts2_h.T).T  # (N, 3)
    
    # Create normalized correspondence matrix
    M_norm = np.column_stack([
        pts1_norm[:, 0], pts1_norm[:, 1],
        pts2_norm[:, 0], pts2_norm[:, 1]
    ])
    
    # Compute F using normalized points
    F_norm = compute_F_raw(M_norm)
    
    # Un-normalize: F = T2'.T @ F_norm @ T1
    F = T2.T @ F_norm @ T1
    
    # Normalize F so that ||F|| = 1 (optional but good practice)
    F = F / np.linalg.norm(F)
    
    return F


def get_normalization_matrix(pts, img_size=None):
    """
    Compute normalization matrix to:
    - Translate image center (or point centroid) to origin
    - Scale to fit in unit square [(-1, -1), (+1, +1)]
    
    Args:
        pts: (N, 2) array of points
        img_size: (width, height) of image. If None, use point centroid.
    
    Returns:
        T: (3, 3) normalization transformation matrix
    """
    if img_size is not None:
        # Use image center as specified in assignment
        width, height = img_size
        center_x = width / 2.0
        center_y = height / 2.0
        # Scale to fit image into unit square
        scale_x = 2.0 / width
        scale_y = 2.0 / height
        scale = min(scale_x, scale_y)  # Use uniform scaling
    else:
        # Use point centroid (Hartley's normalization)
        center_x = np.mean(pts[:, 0])
        center_y = np.mean(pts[:, 1])
        
        # Scale so that average distance from origin is sqrt(2)
        pts_centered = pts - np.array([center_x, center_y])
        mean_dist = np.mean(np.sqrt(np.sum(pts_centered**2, axis=1)))
        
        if mean_dist < 1e-10:
            scale = 1.0
        else:
            scale = np.sqrt(2) / mean_dist
    
    # Transformation matrix: first translate, then scale
    # T = S @ Trans
    # [s  0  -s*center_x]
    # [0  s  -s*center_y]
    # [0  0  1          ]
    T = np.array([
        [scale, 0, -scale * center_x],
        [0, scale, -scale * center_y],
        [0, 0, 1]
    ])
    
    return T


def compute_F_mine(M, img_size1=None, img_size2=None, num_iterations=2000, threshold=1.0):
    """
    Compute fundamental matrix using RANSAC for robust estimation.
    
    Args:
        M: (N, 4) array of correspondences
        img_size1, img_size2: (width, height) of images (optional)
        num_iterations: number of RANSAC iterations
        threshold: inlier threshold for reprojection error
    
    Returns:
        F: (3, 3) fundamental matrix
    """
    N = M.shape[0]
    best_F = None
    best_inliers = 0
    best_error = float('inf')
    
    for _ in range(num_iterations):
        # Randomly sample 8 points
        indices = np.random.choice(N, 8, replace=False)
        M_sample = M[indices]
        
        # Compute F from sample using normalized 8-point
        try:
            F_candidate = compute_F_norm(M_sample, img_size1, img_size2)
        except:
            continue
        
        # Count inliers using Sampson distance or simple point-to-line distance
        errors = compute_sampson_error(M, F_candidate)
        inliers = np.sum(errors < threshold)
        
        if inliers > best_inliers:
            best_inliers = inliers
            best_F = F_candidate
            best_error = np.mean(errors[errors < threshold]) if inliers > 0 else float('inf')
    
    # Refine F using all inliers
    if best_F is not None:
        errors = compute_sampson_error(M, best_F)
        inlier_mask = errors < threshold
        if np.sum(inlier_mask) >= 8:
            M_inliers = M[inlier_mask]
            best_F = compute_F_norm(M_inliers, img_size1, img_size2)
    
    return best_F


def compute_sampson_error(M, F):
    """
    Compute Sampson error (first-order approximation of geometric error).
    
    Args:
        M: (N, 4) correspondences
        F: (3, 3) fundamental matrix
    
    Returns:
        errors: (N,) array of errors
    """
    N = M.shape[0]
    
    # Homogeneous coordinates
    pts1 = np.column_stack([M[:, 0:2], np.ones(N)])  # (N, 3)
    pts2 = np.column_stack([M[:, 2:4], np.ones(N)])  # (N, 3)
    
    # Compute Fx1 and F'x2
    Fx1 = (F @ pts1.T).T  # (N, 3) - epipolar lines in image 2
    Ftx2 = (F.T @ pts2.T).T  # (N, 3) - epipolar lines in image 1
    
    # Sampson error
    # e = (x2' * F * x1)^2 / ((Fx1)_1^2 + (Fx1)_2^2 + (Ftx2)_1^2 + (Ftx2)_2^2)
    numerator = np.sum(pts2 * (F @ pts1.T).T, axis=1) ** 2
    denominator = Fx1[:, 0]**2 + Fx1[:, 1]**2 + Ftx2[:, 0]**2 + Ftx2[:, 1]**2
    
    errors = np.sqrt(numerator / (denominator + 1e-10))
    
    return errors


# ============================================================================
# Part 1-2: Epipolar Line Visualization
# ============================================================================

def compute_epipolar_line(F, pt, which='left'):
    """
    Compute epipolar line corresponding to a point.
    
    Args:
        F: (3, 3) fundamental matrix
        pt: (2,) or (3,) point (homogeneous or not)
        which: 'left' if pt is in left image, 'right' if pt is in right image
    
    Returns:
        line: (3,) epipolar line [a, b, c] where ax + by + c = 0
    """
    if len(pt) == 2:
        pt = np.array([pt[0], pt[1], 1.0])
    
    if which == 'left':
        # Point in left image -> line in right image: l' = F @ p
        line = F @ pt
    else:
        # Point in right image -> line in left image: l = F.T @ p'
        line = F.T @ pt
    
    return line


def draw_epipolar_line(img, line, color, thickness=2):
    """
    Draw an epipolar line on the image.
    
    Args:
        img: image to draw on
        line: [a, b, c] where ax + by + c = 0
        color: BGR color tuple
        thickness: line thickness
    """
    h, w = img.shape[:2]
    a, b, c = line
    
    # Find intersection points with image boundaries
    points = []
    
    # Left boundary (x = 0): by + c = 0 -> y = -c/b
    if abs(b) > 1e-10:
        y = -c / b
        if 0 <= y <= h:
            points.append((0, int(y)))
    
    # Right boundary (x = w-1): a*(w-1) + by + c = 0 -> y = -(a*(w-1) + c)/b
    if abs(b) > 1e-10:
        y = -(a * (w - 1) + c) / b
        if 0 <= y <= h:
            points.append((w - 1, int(y)))
    
    # Top boundary (y = 0): ax + c = 0 -> x = -c/a
    if abs(a) > 1e-10:
        x = -c / a
        if 0 <= x <= w:
            points.append((int(x), 0))
    
    # Bottom boundary (y = h-1): ax + b*(h-1) + c = 0 -> x = -(b*(h-1) + c)/a
    if abs(a) > 1e-10:
        x = -(b * (h - 1) + c) / a
        if 0 <= x <= w:
            points.append((int(x), h - 1))
    
    # Remove duplicates and get two distinct points
    points = list(set(points))
    
    if len(points) >= 2:
        cv2.line(img, points[0], points[1], color, thickness)


def visualize_epipolar_lines(img1, img2, M, F, image_pair_name):
    """
    Interactive visualization of epipolar lines.
    Randomly select 3 correspondences, visualize epipolar lines.
    Press any key except 'q' to select new correspondences.
    Press 'q' to quit.
    
    Args:
        img1, img2: images
        M: (N, 4) correspondences
        F: (3, 3) fundamental matrix
        image_pair_name: name of the image pair for window title
    """
    colors = [
        (0, 0, 255),    # Red (BGR)
        (0, 255, 0),    # Green
        (255, 0, 0)     # Blue
    ]
    
    window_name = f"Epipolar Lines - {image_pair_name}"
    
    while True:
        # Randomly select 3 correspondences
        N = M.shape[0]
        indices = np.random.choice(N, 3, replace=False)
        
        # Create copies to draw on
        vis1 = img1.copy()
        vis2 = img2.copy()
        
        for i, idx in enumerate(indices):
            x1, y1, x2, y2 = M[idx]
            color = colors[i]
            
            # Draw points
            cv2.circle(vis1, (int(x1), int(y1)), 5, color, -1)
            cv2.circle(vis2, (int(x2), int(y2)), 5, color, -1)
            
            # Compute and draw epipolar lines
            # l_i: epipolar line in img2 corresponding to p_i in img1
            l = compute_epipolar_line(F, (x1, y1), 'left')
            draw_epipolar_line(vis2, l, color)
            
            # m_i: epipolar line in img1 corresponding to q_i in img2
            m = compute_epipolar_line(F, (x2, y2), 'right')
            draw_epipolar_line(vis1, m, color)
        
        # Combine images side by side
        vis = np.hstack([vis1, vis2])
        
        # Display
        cv2.imshow(window_name, vis)
        
        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            cv2.destroyWindow(window_name)
            break


# ============================================================================
# Main Execution
# ============================================================================

def process_image_pair(img1_path, img2_path, matches_path, image_pair_name):
    """
    Process a single image pair: compute F matrices and visualize epipolar lines.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {image_pair_name}")
    print(f"{'='*60}")
    
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None:
        print(f"Error: Could not load {img1_path}")
        return
    if img2 is None:
        print(f"Error: Could not load {img2_path}")
        return
    
    # Get image sizes (width, height)
    img_size1 = (img1.shape[1], img1.shape[0])
    img_size2 = (img2.shape[1], img2.shape[0])
    
    # Load correspondences
    M = np.loadtxt(matches_path)
    print(f"Loaded {M.shape[0]} correspondences")
    print(f"Image 1 size: {img_size1}, Image 2 size: {img_size2}")
    
    # Compute fundamental matrices
    F_raw = compute_F_raw(M)
    F_norm = compute_F_norm(M, img_size1, img_size2)
    F_mine = compute_F_mine(M, img_size1, img_size2)
    
    # Compute and print average reprojection errors
    error_raw = compute_avg_reproj_error(M, F_raw)
    error_norm = compute_avg_reproj_error(M, F_norm)
    error_mine = compute_avg_reproj_error(M, F_mine)
    
    print(f"\nAverage Reprojection Errors ({image_pair_name})")
    print(f"Raw = {error_raw}")
    print(f"Norm = {error_norm}")
    print(f"Mine = {error_mine}")
    
    # Visualize epipolar lines using F_mine
    visualize_epipolar_lines(img1, img2, M, F_mine, image_pair_name)


def main():
    print("=" * 60)
    print("Assignment 3 - Part 1: Fundamental Matrix")
    print("=" * 60)
    
    # [수정] 데이터가 있는 디렉토리 지정
    data_dir = './A3_P1_Data'
    
    # 이미지 쌍 정의
    image_pairs = [
        ('temple1.png', 'temple2.png', 'temple_matches.txt', 'temple1.png and temple2.png'),
        ('house1.jpg', 'house2.jpg', 'house_matches.txt', 'house1.jpg and house2.jpg'),
        ('library1.jpg', 'library2.jpg', 'library_matches.txt', 'library1.jpg and library2.jpg'),
    ]
    
    # 각 쌍 처리
    for img1_name, img2_name, matches_name, pair_desc in image_pairs:
        # 경로를 결합하여 파일 찾기
        img1_path = os.path.join(data_dir, img1_name)
        img2_path = os.path.join(data_dir, img2_name)
        matches_path = os.path.join(data_dir, matches_name)
        
        process_image_pair(img1_path, img2_path, matches_path, pair_desc)
    
    print("\n" + "=" * 60)
    print("All image pairs processed!")
    print("=" * 60)

if __name__ == "__main__":
    main()