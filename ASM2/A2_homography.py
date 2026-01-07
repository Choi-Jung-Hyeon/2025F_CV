import cv2
import numpy as np
import os
import time

os.makedirs('./result', exist_ok=True)


def match_features_manual(des1, des2):
    n1, n2 = len(des1), len(des2)
    
    dist_matrix = np.zeros((n1, n2), dtype=np.int32)
    for i in range(n1):
        for j in range(n2):
            xor_result = np.bitwise_xor(des1[i], des2[j])
            dist_matrix[i, j] = np.sum([bin(byte).count('1') for byte in xor_result])
    
    matches = []
    for i in range(n1):
        sorted_indices = np.argsort(dist_matrix[i])
        best_idx = sorted_indices[0]
        second_idx = sorted_indices[1]
        best_dist = dist_matrix[i, best_idx]
        second_dist = dist_matrix[i, second_idx]
        
        if second_dist > 0 and best_dist / second_dist < 0.75:
            match = cv2.DMatch()
            match.queryIdx = i
            match.trainIdx = best_idx
            match.distance = float(best_dist)
            matches.append(match)
    
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def feature_detection_and_matching(img_desk, img_cover):
    orb = cv2.ORB_create()
    
    kp_desk = orb.detect(img_desk, None)
    kp_cover = orb.detect(img_cover, None)
    kp_desk, des_desk = orb.compute(img_desk, kp_desk)
    kp_cover, des_cover = orb.compute(img_cover, kp_cover)
    
    print(f"[2-1] ORB Features Extracted:")
    print(f"      - cv_desk: {len(kp_desk)} keypoints")
    print(f"      - cv_cover: {len(kp_cover)} keypoints")
    
    all_matches = match_features_manual(des_desk, des_cover)
    
    distance_threshold = 50
    good_matches = [m for m in all_matches if m.distance < distance_threshold]
    
    print(f"      - Matches after ratio test: {len(all_matches)}")
    print(f"      - Good matches (dist < {distance_threshold}): {len(good_matches)}")
    
    if len(good_matches) >= 15:
        matches = good_matches
    else:
        matches = all_matches
    
    print(f"      - Total matches used: {len(matches)}")
    print(f"      - Top-10 match distances: {[m.distance for m in matches[:10]]}")
    
    return kp_desk, kp_cover, matches


def compute_normalization_matrix(points):
    mean_x = np.mean(points[:, 0])
    mean_y = np.mean(points[:, 1])
    
    centered = points - np.array([mean_x, mean_y])
    distances = np.sqrt(centered[:, 0]**2 + centered[:, 1]**2)
    max_distance = np.max(distances)
    
    if max_distance > 0:
        scale = np.sqrt(2) / max_distance
    else:
        scale = 1.0
    
    T = np.array([
        [scale, 0, -scale * mean_x],
        [0, scale, -scale * mean_y],
        [0, 0, 1]
    ], dtype=np.float64)
    
    return T


def compute_homography(srcP, destP):
    srcP = np.array(srcP, dtype=np.float64)
    destP = np.array(destP, dtype=np.float64)
    n = len(srcP)
    
    T_S = compute_normalization_matrix(srcP)
    T_D = compute_normalization_matrix(destP)
    
    srcP_h = np.hstack([srcP, np.ones((n, 1))])
    destP_h = np.hstack([destP, np.ones((n, 1))])
    
    srcP_norm = (T_S @ srcP_h.T).T
    destP_norm = (T_D @ destP_h.T).T
    
    srcP_norm_2d = srcP_norm[:, :2] / srcP_norm[:, 2:3]
    destP_norm_2d = destP_norm[:, :2] / destP_norm[:, 2:3]
    
    A = []
    for i in range(n):
        x, y = srcP_norm_2d[i]
        xp, yp = destP_norm_2d[i]
        A.append([-x, -y, -1, 0, 0, 0, xp * x, xp * y, xp])
        A.append([0, 0, 0, -x, -y, -1, yp * x, yp * y, yp])
    
    A = np.array(A, dtype=np.float64)
    
    U, S, Vh = np.linalg.svd(A)
    h = Vh[-1, :]
    H_N = h.reshape(3, 3)
    
    H = np.linalg.inv(T_D) @ H_N @ T_S
    
    if H[2, 2] != 0:
        H = H / H[2, 2]
    
    return H


def compute_projection_error(H, srcP, destP):
    n = len(srcP)
    srcP_h = np.hstack([srcP, np.ones((n, 1))]).T
    
    projected = H @ srcP_h
    
    w = projected[2:3, :]
    w = np.where(np.abs(w) < 1e-10, 1e-10, w)
    
    projected_2d = projected[:2, :] / w
    projected_2d = projected_2d.T
    
    errors = np.sqrt(np.sum((projected_2d - destP)**2, axis=1))
    errors = np.where(np.isfinite(errors), errors, float('inf'))
    
    return errors


def compute_homography_ransac(srcP, destP, th):
    srcP = np.array(srcP, dtype=np.float64)
    destP = np.array(destP, dtype=np.float64)
    n = len(srcP)
    
    best_H = None
    best_inlier_count = 0
    best_inliers = None
    
    max_iterations = 1000
    start_time = time.time()
    
    for iteration in range(max_iterations):
        if time.time() - start_time > 2.5:
            break
        
        indices = np.random.choice(n, 4, replace=False)
        src_sample = srcP[indices]
        dest_sample = destP[indices]
        
        try:
            H = compute_homography(src_sample, dest_sample)
        except:
            continue
        
        if not np.isfinite(H).all():
            continue
        
        errors = compute_projection_error(H, srcP, destP)
        
        inliers = errors < th
        inlier_count = np.sum(inliers)
        
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_H = H
            best_inliers = inliers
    
    if best_inliers is not None and np.sum(best_inliers) >= 4:
        inlier_srcP = srcP[best_inliers]
        inlier_destP = destP[best_inliers]
        best_H = compute_homography(inlier_srcP, inlier_destP)
    
    elapsed_time = time.time() - start_time
    print(f"[2-3] RANSAC completed in {elapsed_time:.2f} seconds")
    print(f"      - Iterations: {min(iteration + 1, max_iterations)}")
    print(f"      - Best inlier count: {best_inlier_count}/{n}")
    print(f"      - Inlier ratio: {best_inlier_count/n*100:.1f}%")
    
    return best_H


def warp_and_compose(cover_img, desk_img, H):
    h, w = desk_img.shape[:2]
    
    warped = cv2.warpPerspective(cover_img, H, (w, h))
    
    mask = cv2.warpPerspective(
        np.ones(cover_img.shape[:2], dtype=np.uint8) * 255, 
        H, (w, h)
    )
    
    composed = desk_img.copy()
    composed[mask > 0] = warped[mask > 0]
    
    return warped, composed


def main():
    print("=" * 60)
    print("Part 2: Homography")
    print("=" * 60)
    
    cv_cover = cv2.imread('cv_cover.jpg', cv2.IMREAD_GRAYSCALE)
    cv_desk = cv2.imread('cv_desk.png', cv2.IMREAD_GRAYSCALE)
    hp_cover = cv2.imread('hp_cover.jpg', cv2.IMREAD_GRAYSCALE)
    
    if cv_cover is None or cv_desk is None or hp_cover is None:
        print("Error: Could not read images!")
        return
    
    print(f"Images loaded:")
    print(f"  - cv_cover.jpg: {cv_cover.shape}")
    print(f"  - cv_desk.png: {cv_desk.shape}")
    print(f"  - hp_cover.jpg: {hp_cover.shape}")
    
    print("\n" + "-" * 60)
    print("2-1. Feature Detection and Matching")
    print("-" * 60)
    
    kp_desk, kp_cover, matches = feature_detection_and_matching(
        cv_desk, cv_cover
    )
    
    top10_matches = matches[:10]
    match_img = cv2.drawMatches(
        cv_desk, kp_desk, cv_cover, kp_cover, top10_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    cv2.imwrite('./result/part2_matches.png', match_img)
    print(f"\n      Top-10 matches saved to './result/part2_matches.png'")
    
    cv2.imshow('Top-10 Matches', match_img)
    cv2.waitKey(1000)
    
    srcP = np.array([kp_cover[m.trainIdx].pt for m in matches], dtype=np.float64)
    destP = np.array([kp_desk[m.queryIdx].pt for m in matches], dtype=np.float64)
    
    print(f"\n      Total matched points: {len(srcP)}")
    
    print("\n" + "-" * 60)
    print("2-2. Homography with Normalization")
    print("-" * 60)
    
    H_norm = None
    if len(srcP) >= 15:
        H_norm = compute_homography(srcP, destP)
        print(f"[2-2] Homography matrix (with normalization):")
        print(H_norm)
        
        warped_norm, composed_norm = warp_and_compose(cv_cover, cv_desk, H_norm)
        
        cv2.imshow('Homography (Norm) - Warped', warped_norm)
        cv2.imshow('Homography (Norm) - Composed', composed_norm)
        cv2.waitKey(1000)
    else:
        print(f"Warning: Not enough matches ({len(srcP)} < 15)")
    
    print("\n" + "-" * 60)
    print("2-3. Homography with RANSAC")
    print("-" * 60)
    
    threshold = 10.0
    
    H_ransac = compute_homography_ransac(srcP, destP, threshold)
    print(f"[2-3] Homography matrix (with RANSAC):")
    print(H_ransac)
    
    print("\n" + "-" * 60)
    print("2-4. Image Warping")
    print("-" * 60)
    
    warped_ransac, composed_ransac = warp_and_compose(cv_cover, cv_desk, H_ransac)
    
    cv2.imwrite('./result/part2_homography.png', composed_ransac)
    print(f"[2-4] cv_cover composed result saved to './result/part2_homography.png'")
    
    cv2.imshow('Homography (RANSAC) - Warped', warped_ransac)
    cv2.imshow('Homography (RANSAC) - Composed', composed_ransac)
    cv2.waitKey(1000)
    
    print("\n" + "-" * 60)
    print("2-4c. HP Cover Composition")
    print("-" * 60)
    
    hp_cover_resized = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))
    
    warped_hp, composed_hp = warp_and_compose(hp_cover_resized, cv_desk, H_ransac)
    
    cv2.imwrite('./result/part2_hp_homography.png', composed_hp)
    
    cv2.imshow('HP Cover - Warped', warped_hp)
    cv2.imshow('HP Cover - Composed', composed_hp)
    print(f"[2-4c] hp_cover composition saved to './result/part2_hp_homography.png'")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"- ORB features: {len(kp_cover)} (cover), {len(kp_desk)} (desk)")
    print(f"- Total matches: {len(matches)}")
    print(f"- Threshold for RANSAC: {threshold} pixels")
    print(f"- Results saved to ./result/")
    print("=" * 60)
    
    print("\nPress any key to close all windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()