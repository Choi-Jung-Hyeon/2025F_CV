"""
Introduction to Computer Vision - Assignment #1
Part 2: Edge Detection

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


def compute_image_gradient(img):
    """Compute image gradient using Sobel filters"""
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
    
    mag = np.sqrt(Ix**2 + Iy**2)
    dir = np.arctan2(Iy, Ix)
    
    return mag, dir


def non_maximum_suppression_dir(mag, dir):
    """Non-Maximum Suppression with 8-direction quantization"""
    h, w = mag.shape
    suppressed_mag = np.zeros_like(mag)
    
    angle = np.rad2deg(dir) % 180
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            current_mag = mag[i, j]
            current_angle = angle[i, j]
            
            # Quantize to 8 directions
            if (0 <= current_angle < 22.5) or (157.5 <= current_angle < 180):
                neighbor1 = mag[i, j-1]
                neighbor2 = mag[i, j+1]
            elif 22.5 <= current_angle < 67.5:
                neighbor1 = mag[i-1, j+1]
                neighbor2 = mag[i+1, j-1]
            elif 67.5 <= current_angle < 112.5:
                neighbor1 = mag[i-1, j]
                neighbor2 = mag[i+1, j]
            else:
                neighbor1 = mag[i-1, j-1]
                neighbor2 = mag[i+1, j+1]
            
            if current_mag > neighbor1 and current_mag > neighbor2:
                suppressed_mag[i, j] = current_mag
    
    return suppressed_mag


def process_image(image_path):
    """Process image for edge detection"""
    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Cannot load image {image_path}")
        return
    
    img = img.astype(np.float64)
    
    print(f"\n{'='*60}")
    print(f"Processing: {filename}.png")
    print(f"Image shape: {img.shape}")
    
    # 2-1. Gaussian filtering (7, 1.5)
    print("\n2-1. Applying Gaussian filter (7, 1.5)...")
    gaussian_kernel = get_gaussian_filter_2d(7, 1.5)
    filtered_img = cross_correlation_2d(img, gaussian_kernel)
    
    # 2-2. Compute gradient
    print("\n2-2. Computing image gradient...")
    start_time = time.time()
    mag, dir = compute_image_gradient(filtered_img)
    gradient_time = time.time() - start_time
    
    print(f"   Computation time: {gradient_time:.6f} seconds")
    print(f"   Magnitude - Min: {mag.min():.2f}, Max: {mag.max():.2f}")
    
    result_dir = './result'
    os.makedirs(result_dir, exist_ok=True)
    
    # Save and visualize raw magnitude
    mag_normalized = (mag / mag.max() * 255).astype(np.uint8)
    raw_output_path = f'{result_dir}/part_2_edge_raw_{filename}.png'
    cv2.imwrite(raw_output_path, mag_normalized)
    print(f"   Saved: {raw_output_path}")
    
    cv2.imshow(f'Part 2 - Edge Magnitude Raw ({filename})', mag_normalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 2-3. Non-Maximum Suppression
    print("\n2-3. Applying Non-Maximum Suppression...")
    start_time = time.time()
    suppressed_mag = non_maximum_suppression_dir(mag, dir)
    nms_time = time.time() - start_time
    
    print(f"   Computation time: {nms_time:.6f} seconds")
    print(f"   Non-zero pixels: {np.count_nonzero(suppressed_mag)}")
    
    # Save and visualize suppressed magnitude
    sup_max = suppressed_mag.max()
    sup_normalized = (suppressed_mag / sup_max * 255).astype(np.uint8) if sup_max > 0 else suppressed_mag.astype(np.uint8)
    sup_output_path = f'{result_dir}/part_2_edge_sup_{filename}.png'
    cv2.imwrite(sup_output_path, sup_normalized)
    print(f"   Saved: {sup_output_path}")
    
    cv2.imshow(f'Part 2 - Edge After NMS ({filename})', sup_normalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """Main execution function"""
    print("="*60)
    print("Computer Vision Assignment #1 - Part 2: Edge Detection")
    print("Student ID: 2021313692, Name: Choi Jung Hyun")
    print("="*60)
    
    images = ['shapes.png', 'lenna.png']
    
    for image_path in images:
        if os.path.exists(image_path):
            process_image(image_path)
        else:
            print(f"\nWarning: {image_path} not found")
    
    print("\n" + "="*60)
    print("Part 2 completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()