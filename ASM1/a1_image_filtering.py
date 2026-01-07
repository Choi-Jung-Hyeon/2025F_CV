"""
Introduction to Computer Vision - Assignment #1
Part 1: Image Filtering

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


def cross_correlation_1d(img, kernel):
    """1D Cross-Correlation (auto-detect horizontal/vertical)"""
    img_h, img_w = img.shape
    kernel_h, kernel_w = kernel.shape
    
    is_horizontal = (kernel_h == 1 and kernel_w > 1)
    is_vertical = (kernel_h > 1 and kernel_w == 1)
    
    if not (is_horizontal or is_vertical):
        raise ValueError("Kernel must be 1D (1xN or Nx1)")
    
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    
    padded_img = manual_pad_edge(img, pad_h, pad_w)
    filtered_img = np.zeros_like(img, dtype=np.float64)
    
    if is_horizontal:
        for i in range(img_h):
            for j in range(img_w):
                window = padded_img[i+pad_h, j:j+kernel_w]
                filtered_img[i, j] = np.sum(window * kernel.flatten())
    else:
        for i in range(img_h):
            for j in range(img_w):
                window = padded_img[i:i+kernel_h, j+pad_w]
                filtered_img[i, j] = np.sum(window * kernel.flatten())
    
    return filtered_img


def get_gaussian_filter_1d(size, sigma):
    """1D Gaussian filter kernel"""
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    
    center = size // 2
    kernel = np.zeros((1, size), dtype=np.float64)
    
    for i in range(size):
        x = i - center
        kernel[0, i] = np.exp(-(x**2) / (2 * sigma**2))
    
    kernel = kernel / np.sum(kernel)
    return kernel


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


def create_grid_image(images, captions):
    """Create 3x3 grid image with captions"""
    cell_h, cell_w = images[0].shape
    margin = 40
    
    grid_h = cell_h * 3 + margin * 3
    grid_w = cell_w * 3 + margin * 3
    grid = np.ones((grid_h, grid_w), dtype=np.uint8) * 255
    
    for idx, (img, caption) in enumerate(zip(images, captions)):
        row = idx // 3
        col = idx % 3
        
        y_start = row * (cell_h + margin) + margin
        x_start = col * (cell_w + margin) + margin
        
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        grid[y_start:y_start+cell_h, x_start:x_start+cell_w] = img_uint8
        
        cv2.putText(grid, caption, (x_start + 10, y_start - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    
    return grid


def process_image(image_path):
    """Process image with Gaussian filtering"""
    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Cannot load image {image_path}")
        return
    
    img = img.astype(np.float64)
    
    print(f"\n{'='*60}")
    print(f"Processing: {filename}.png")
    print(f"Image shape: {img.shape}")
    
    # 1-2.d) 9 different Gaussian filterings
    print("\n1-2.d) Performing 9 different Gaussian filterings...")
    
    sizes = [5, 11, 17]
    sigmas = [1, 6, 11]
    
    filtered_images = []
    captions = []
    
    for size in sizes:
        for sigma in sigmas:
            kernel = get_gaussian_filter_2d(size, sigma)
            filtered = cross_correlation_2d(img, kernel)
            filtered_images.append(filtered)
            captions.append(f'{size}x{size} s={sigma}')
    
    grid_img = create_grid_image(filtered_images, captions)
    
    result_dir = './result'
    os.makedirs(result_dir, exist_ok=True)
    output_path = f'{result_dir}/part_1_gaussian_filtered_{filename}.png'
    cv2.imwrite(output_path, grid_img)
    print(f"   Saved: {output_path}")
    
    cv2.imshow(f'Part 1 - Gaussian Filtering ({filename})', grid_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 1-2.e) 1D vs 2D comparison (17x17, σ=6)
    print("\n1-2.e) Comparing 1D vs 2D Gaussian filtering (17x17, σ=6)...")
    
    size, sigma = 17, 6
    
    kernel_2d = get_gaussian_filter_2d(size, sigma)
    start_time = time.time()
    result_2d = cross_correlation_2d(img, kernel_2d)
    time_2d = time.time() - start_time
    
    kernel_1d_h = get_gaussian_filter_1d(size, sigma)
    kernel_1d_v = kernel_1d_h.T
    
    start_time = time.time()
    temp_result = cross_correlation_1d(img, kernel_1d_v)
    result_1d_seq = cross_correlation_1d(temp_result, kernel_1d_h)
    time_1d = time.time() - start_time
    
    diff_map = np.abs(result_2d - result_1d_seq)
    sum_diff = np.sum(diff_map)
    
    print(f"   2D filtering time: {time_2d:.6f} seconds")
    print(f"   1D sequential filtering time: {time_1d:.6f} seconds")
    print(f"   Speed improvement: {time_2d/time_1d:.2f}x")
    print(f"   Sum of absolute differences: {sum_diff:.10f}")
    
    # Visualize comparison
    result_2d_vis = np.clip(result_2d, 0, 255).astype(np.uint8)
    result_1d_vis = np.clip(result_1d_seq, 0, 255).astype(np.uint8)
    diff_vis = np.clip(diff_map / diff_map.max() * 255, 0, 255).astype(np.uint8) if diff_map.max() > 0 else diff_map.astype(np.uint8)
    
    comparison = np.hstack([result_2d_vis, result_1d_vis, diff_vis])
    
    cv2.imshow(f'Part 1 - 1D vs 2D Comparison ({filename})', comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """Main execution function"""
    print("="*60)
    print("Computer Vision Assignment #1 - Part 1: Image Filtering")
    print("Student ID: 2021313692, Name: Choi Jung Hyun")
    print("="*60)
    
    # 1-2.c) Print Gaussian kernels
    print("\n1-2.c) Gaussian filter kernels:")
    print("\nget_gaussian_filter_1d(5, 1):")
    kernel_1d = get_gaussian_filter_1d(5, 1)
    print(kernel_1d)
    
    print("\nget_gaussian_filter_2d(5, 1):")
    kernel_2d = get_gaussian_filter_2d(5, 1)
    print(kernel_2d)
    
    # 1-2.f) Process images
    images = ['lenna.png', 'shapes.png']
    
    for image_path in images:
        if os.path.exists(image_path):
            process_image(image_path)
        else:
            print(f"\nWarning: {image_path} not found")
    
    print("\n" + "="*60)
    print("Part 1 completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()