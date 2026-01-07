"""
Introduction to Computer Vision - Assignment #1
Part 3: Corner Detection

Student ID: 2021313692
Name: Choi Jung Hyun
"""

import numpy as np
import cv2
import time
import os


def manual_pad_edge(img, pad_h, pad_w):
    """Manual edge padding (replicate nearest pixel)"""
    h, w = img.shape
    padded = np.zeros((h + 2*pad_h, w + 2*pad_w), dtype=img.dtype)
    
    padded[pad_h:h+pad_h, pad_w:w+pad_w] = img
    
    for i in range(pad_h):
        padded[i, pad_w:w+pad_w] = img[0, :]
        padded[h+pad_h+i, pad_w:w+pad_w] = img[-1, :]
    for j in range(pad_w):
        padded[pad_h:h+pad_h, j] = img[:, 0]
        padded[pad_h:h+pad_h, w+pad_w+j] = img[:, -1]
    
    padded[:pad_h, :pad_w] = img[0, 0]
    padded[:pad_h, w+pad_w:] = img[0, -1]
    padded[h+pad_h:, :pad_w] = img[-1, 0]
    padded[h+pad_h:, w+pad_w:] = img[-1, -1]
    
    return padded


def cross_correlation_2d(img, kernel):
    """2D Cross-Correlation"""
    img_h, img_w = img.shape
    kernel_h, kernel_w = kernel.shape
    
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    
    padded_img = manual_pad_edge(img, pad_h, pad_w)
    filtered_img = np.zeros_like(img, dtype=np.float64)
    
    for i in range(img_h):
        for j in range(img_w):
            window = padded_img[i:i+kernel_h, j:j+kernel_w]
            filtered_img[i, j] = np.sum(window * kernel)
    
    return filtered_img


def get_gaussian_filter_2d(size, sigma):
    """2D Gaussian filter kernel"""
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    
    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float64)
    
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    kernel = kernel / np.sum(kernel)
    return kernel


def compute_corner_response(img):
    """Compute Harris corner response"""
    # Sobel filters
    sobel_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ], dtype=np.float64)
    
    sobel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ], dtype=np.float64)
    
    Ix = cross_correlation_2d(img, sobel_x)
    Iy = cross_correlation_2d(img, sobel_y)
    
    # Second moment matrix elements
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy
    
    # 5x5 uniform window
    window = np.ones((5, 5), dtype=np.float64)
    Sx2 = cross_correlation_2d(Ix2, window)
    Sy2 = cross_correlation_2d(Iy2, window)
    Sxy = cross_correlation_2d(Ixy, window)
    
    # Harris response: R = λ₁λ₂ - κ(λ₁+λ₂)²
    kappa = 0.04
    det_M = Sx2 * Sy2 - Sxy * Sxy
    trace_M = Sx2 + Sy2
    R = det_M - kappa * (trace_M ** 2)
    
    # Negative to 0, normalize to [0,1]
    R[R < 0] = 0
    if R.max() > 0:
        R = R / R.max()
    
    return R


def non_maximum_suppression_win(R, winSize):
    """Window-based Non-Maximum Suppression"""
    h, w = R.shape
    suppressed_R = np.zeros_like(R)
    half_win = winSize // 2
    threshold = 0.1
    
    for i in range(h):
        for j in range(w):
            current_R = R[i, j]
            
            if current_R <= threshold:
                continue
            
            y_min = max(0, i - half_win)
            y_max = min(h, i + half_win + 1)
            x_min = max(0, j - half_win)
            x_max = min(w, j + half_win + 1)
            
            window = R[y_min:y_max, x_min:x_max]
            
            if current_R >= np.max(window):
                suppressed_R[i, j] = current_R
    
    return suppressed_R


def process_image(image_path):
    """Process image for corner detection"""
    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Cannot load image {image_path}")
        return
    
    img_original = img.copy()
    img = img.astype(np.float64)
    
    print(f"\n{'='*60}")
    print(f"Processing: {filename}.png")
    print(f"Image shape: {img.shape}")
    
    # 3-1. Gaussian filtering (7, 1.5)
    print("\n3-1. Applying Gaussian filter (7, 1.5)...")
    gaussian_kernel = get_gaussian_filter_2d(7, 1.5)
    filtered_img = cross_correlation_2d(img, gaussian_kernel)
    
    # 3-2. Compute corner response
    print("\n3-2. Computing corner response...")
    start_time = time.time()
    R = compute_corner_response(filtered_img)
    corner_time = time.time() - start_time
    
    print(f"   Computation time: {corner_time:.6f} seconds")
    print(f"   Response - Min: {R.min():.4f}, Max: {R.max():.4f}")
    print(f"   Pixels above 0.1: {np.count_nonzero(R > 0.1)}")
    
    result_dir = './result'
    os.makedirs(result_dir, exist_ok=True)
    
    # Save and visualize raw corner response
    R_vis = (R * 255).astype(np.uint8)
    raw_output_path = f'{result_dir}/part_3_corner_raw_{filename}.png'
    cv2.imwrite(raw_output_path, R_vis)
    print(f"   Saved: {raw_output_path}")
    
    cv2.imshow(f'Part 3 - Corner Response Raw ({filename})', R_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 3-3.a) Threshold > 0.1 (green pixels)
    print("\n3-3.a) Thresholding (> 0.1)...")
    
    img_bin = cv2.cvtColor(img_original, cv2.COLOR_GRAY2RGB)
    threshold = 0.1
    corner_mask = R > threshold
    img_bin[corner_mask] = [0, 255, 0]
    
    # 3-3.b) Save and visualize binary result
    bin_output_path = f'{result_dir}/part_3_corner_bin_{filename}.png'
    cv2.imwrite(bin_output_path, img_bin)
    print(f"   Saved: {bin_output_path}")
    
    cv2.imshow(f'Part 3 - Corner Binary ({filename})', img_bin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 3-3.c) Non-Maximum Suppression (11x11)
    print("\n3-3.c) Applying Non-Maximum Suppression (11x11 window)...")
    start_time = time.time()
    suppressed_R = non_maximum_suppression_win(R, 11)
    nms_time = time.time() - start_time
    
    print(f"   Computation time: {nms_time:.6f} seconds")
    print(f"   Detected corners: {np.count_nonzero(suppressed_R)}")
    
    # 3-3.d) Draw green circles and save
    img_sup = cv2.cvtColor(img_original, cv2.COLOR_GRAY2RGB)
    corner_positions = np.argwhere(suppressed_R > 0)
    
    for y, x in corner_positions:
        cv2.circle(img_sup, (x, y), 3, (0, 255, 0), 1)
    
    sup_output_path = f'{result_dir}/part_3_corner_sup_{filename}.png'
    cv2.imwrite(sup_output_path, img_sup)
    print(f"   Saved: {sup_output_path}")
    
    cv2.imshow(f'Part 3 - Corner Suppressed ({filename})', img_sup)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """Main execution function"""
    print("="*60)
    print("Computer Vision Assignment #1 - Part 3: Corner Detection")
    print("Student ID: 2021313692, Name: Choi Jung Hyun")
    print("="*60)
    
    images = ['shapes.png', 'lenna.png']
    
    for image_path in images:
        if os.path.exists(image_path):
            process_image(image_path)
        else:
            print(f"\nWarning: {image_path} not found")
    
    print("\n" + "="*60)
    print("Part 3 completed successfully!")
    print("="*60)
    print("\nAll parts of Assignment #1 are now complete!")
    print("Check the './result' directory for output images.")


if __name__ == "__main__":
    main()