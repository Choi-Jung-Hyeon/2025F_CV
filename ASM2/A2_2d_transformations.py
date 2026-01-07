import cv2
import numpy as np
import os


def get_transformed_image(img, M):
    canvas_size = 801
    plane = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255
    origin = 400
    
    img_h, img_w = img.shape
    half_h, half_w = img_h // 2, img_w // 2
    
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return plane
    
    for py in range(canvas_size):
        for px in range(canvas_size):
            wx = px - origin
            wy = origin - py
            
            world_coord = np.array([wx, wy, 1.0])
            src_coord = M_inv @ world_coord
            src_wx = src_coord[0] / src_coord[2]
            src_wy = src_coord[1] / src_coord[2]
            
            src_ix = src_wx + half_w
            src_iy = half_h - src_wy
            
            src_ix = int(round(src_ix))
            src_iy = int(round(src_iy))
            
            if 0 <= src_ix < img_w and 0 <= src_iy < img_h:
                plane[py, px] = img[src_iy, src_ix]
    
    arrow_color = 0
    arrow_thickness = 1
    arrow_length = 350
    
    cv2.arrowedLine(plane, (origin - arrow_length, origin),
                    (origin + arrow_length, origin), arrow_color, arrow_thickness, tipLength=0.05)
    cv2.arrowedLine(plane, (origin, origin + arrow_length),
                    (origin, origin - arrow_length), arrow_color, arrow_thickness, tipLength=0.05)
    
    return plane


def get_translation_matrix(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)


def get_rotation_matrix(angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float64)


def get_scale_matrix(sx, sy):
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)


def get_flip_y_matrix():
    return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)


def get_flip_x_matrix():
    return np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)


def get_shear_x_matrix(angle_deg):
    shear_factor = np.tan(np.deg2rad(angle_deg))
    return np.array([[1, shear_factor, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)


def save_transformation_results(img):
    os.makedirs('./result', exist_ok=True)
    
    M_identity = np.eye(3, dtype=np.float64)
    plane_identity = get_transformed_image(img, M_identity)
    
    M_rotation = get_rotation_matrix(45)
    plane_rotation = get_transformed_image(img, M_rotation)
    
    M_scaling = get_scale_matrix(2.0, 1.0)
    plane_scaling = get_transformed_image(img, M_scaling)
    
    M_shearing = get_shear_x_matrix(45)
    plane_shearing = get_transformed_image(img, M_shearing)
    
    top_row = np.hstack([plane_identity, plane_rotation])
    bottom_row = np.hstack([plane_scaling, plane_shearing])
    combined = np.vstack([top_row, bottom_row])
    
    cv2.imwrite('./result/part1_transformations.png', combined)
    print("Saved: ./result/part1_transformations.png")


def main():
    img = cv2.imread('smile.png', cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Cannot load 'smile.png'")
        return
    
    print("Image loaded successfully.")
    print(f"Image size: {img.shape}")
    
    save_transformation_results(img)
    
    M = np.eye(3, dtype=np.float64)
    
    window_name = '2D Transformations'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    print("\n=== Interactive 2D Transformations ===")
    print("Controls:")
    print("  a/d: Move left/right (5px)")
    print("  w/s: Move up/down (5px)")
    print("  r/t: Rotate counter-clockwise/clockwise (5 degrees)")
    print("  f: Flip across y-axis")
    print("  g: Flip across x-axis")
    print("  x/c: Shrink/Enlarge along x (5%)")
    print("  y/u: Shrink/Enlarge along y (5%)")
    print("  h: Reset to initial state")
    print("  q: Quit")
    print("=" * 40)
    
    while True:
        plane = get_transformed_image(img, M)
        cv2.imshow(window_name, plane)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            print("Quit.")
            break
        elif key == ord('h'):
            M = np.eye(3, dtype=np.float64)
            print("Reset to initial state.")
        elif key == ord('a'):
            M = get_translation_matrix(-5, 0) @ M
            print("Move left by 5 pixels.")
        elif key == ord('d'):
            M = get_translation_matrix(5, 0) @ M
            print("Move right by 5 pixels.")
        elif key == ord('w'):
            M = get_translation_matrix(0, 5) @ M
            print("Move up by 5 pixels.")
        elif key == ord('s'):
            M = get_translation_matrix(0, -5) @ M
            print("Move down by 5 pixels.")
        elif key == ord('r'):
            M = get_rotation_matrix(5) @ M
            print("Rotate counter-clockwise by 5 degrees.")
        elif key == ord('t'):
            M = get_rotation_matrix(-5) @ M
            print("Rotate clockwise by 5 degrees.")
        elif key == ord('f'):
            M = get_flip_y_matrix() @ M
            print("Flip across y-axis.")
        elif key == ord('g'):
            M = get_flip_x_matrix() @ M
            print("Flip across x-axis.")
        elif key == ord('x'):
            M = get_scale_matrix(0.95, 1.0) @ M
            print("Shrink along x-axis by 5%.")
        elif key == ord('c'):
            M = get_scale_matrix(1.05, 1.0) @ M
            print("Enlarge along x-axis by 5%.")
        elif key == ord('y'):
            M = get_scale_matrix(1.0, 0.95) @ M
            print("Shrink along y-axis by 5%.")
        elif key == ord('u'):
            M = get_scale_matrix(1.0, 1.05) @ M
            print("Enlarge along y-axis by 5%.")
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()