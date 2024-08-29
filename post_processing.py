import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, filters

def calculate_combined_gradient(image):
    # Compute Sobel gradients
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude and direction
    gradient_magnitude, gradient_direction = cv2.cartToPolar(sobel_x, sobel_y, angleInDegrees=False)

    # Combine gradients into polar form
    gradient_polar = gradient_magnitude * np.exp(1j * np.radians(gradient_direction))

    # Stack magnitude and direction
    gradient_combined = np.stack([gradient_magnitude, gradient_direction], axis=-1)

    return gradient_combined

def estimate_orientation_angle(G, m, n, N):
    angle_sum_sin = 0
    angle_sum_cos = 0
    angle_list = []

    for i in range(max(0, m - N // 2), min(G.shape[0], m + N // 2 + 1)):
        for j in range(max(0, n - N // 2), min(G.shape[1], n + N // 2 + 1)):
            G_ij = G[i, j, 0]
            theta_ij = G[i, j, 1]
            angle_sum_sin += G_ij**2 * np.sin(2 * theta_ij)
            angle_sum_cos += G_ij**2 * np.cos(2 * theta_ij)
            angle_list.append(theta_ij)

    orientation_angle = 0.5 * np.arctan2(angle_sum_sin, angle_sum_cos) + 0.5 * np.pi
    angle_var = np.var(angle_list)
    return orientation_angle, angle_var

def calculate_orientation_angles(image, stride=20, neighborhood_size=10):
    gradient_combined = calculate_combined_gradient(image)
    height, width = image.shape
    orientation_matrix = np.zeros((height // stride, width // stride), dtype=float)
    var_matrix = np.zeros((height // stride, width // stride), dtype=float)

    for i in range(0, height, stride):
        for j in range(0, width, stride):
            orientation_angle, angle_var = estimate_orientation_angle(gradient_combined, i, j, neighborhood_size)
            orientation_matrix[i // stride, j // stride] = orientation_angle
            var_matrix[i // stride, j // stride] = angle_var

    return orientation_matrix, var_matrix

def connected_component_labeling(binary_image):
    # Label connected components
    labeled_image, num_labels = measure.label(binary_image, connectivity=2, return_num=True)
    return labeled_image, num_labels

def direction_decision(orientation_matrix, var_matrix, stride=20):
    height, width = orientation_matrix.shape
    flow_direction_matrix = np.zeros_like(orientation_matrix)

    for m in range(height):
        for n in range(width):
            # Calculate coherence
            coherence_score_pos = calculate_coherence_score(orientation_matrix, m, n, stride, direction=0)
            coherence_score_neg = calculate_coherence_score(orientation_matrix, m, n, stride, direction=np.pi)

            # Determine flow direction
            if coherence_score_pos > coherence_score_neg:
                flow_direction_matrix[m, n] = orientation_matrix[m, n]
            else:
                flow_direction_matrix[m, n] = orientation_matrix[m, n] + np.pi

    return flow_direction_matrix

def calculate_coherence_score(orientation_matrix, m, n, stride, direction):
    score = 0
    count = 0
    height, width = orientation_matrix.shape

    for i in range(max(0, m - 1), min(height, m + 2)):
        for j in range(max(0, n - 1), min(width, n + 2)):
            if i == m and j == n:
                continue
            angle_diff = abs(orientation_matrix[i, j] - (direction + orientation_matrix[m, n]))
            score += np.cos(angle_diff)
            count += 1

    return score / count if count > 0 else 0

def draw_orientation_lines(image, orientation_matrix, stride=20, line_length=20):
    height, width = image.shape[:2]
    image_copy = image.copy()

    for i in range(0, height, stride):
        for j in range(0, width, stride):
            angle = orientation_matrix[i // stride, j // stride]
            x, y = j, i
            x_end = int(x + line_length * np.cos(angle))
            y_end = int(y + line_length * np.sin(angle))
            cv2.line(image_copy, (x, y), (x_end, y_end), (0, 255, 0), 2)

    plt.figure(figsize=(10, 8))
    plt.imshow(image_copy)
    plt.show()


if __name__ == '__main__':
    # VERY IMPORTANT!!
    # if you haven't trained a suitable edge detection module, you can use canny to test the result, otherwise, you can
    # pass the edge result to post_processing and replace 'edges' with you predicted results
    # DO NOT USE IMAGES DIRECTLY!! SET A THRESHOlD
    # 如果没训练或者调试出了问题，用canny生成的edge先凑合一下，之后可以替换下面的代码为神经网络预测的edges
    image = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
    # Edge detection and binarization
    edges = filters.sobel(image)
    binary_image = edges > filters.threshold_otsu(edges)

    # Connected component labeling
    labeled_image, num_labels = connected_component_labeling(binary_image)

    # Orientation estimation
    orientation_matrix, var_matrix = calculate_orientation_angles(image)

    # Direction decision
    flow_direction_matrix = direction_decision(orientation_matrix, var_matrix)

    # Draw orientation lines
    draw_orientation_lines(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), flow_direction_matrix)
