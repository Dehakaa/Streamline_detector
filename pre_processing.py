import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    """
    Load an image from the specified file path.
    """
    image = cv2.imread(image_path)
    return image


def convert_to_grayscale(image):
    """
    Convert the image to grayscale using the red channel.
    """
    red_channel = image[:, :, 2]  # Extract the red channel
    return red_channel


def apply_bilateral_filter(image, sigma_s=15, sigma_t=30):
    """
    Apply bilateral filter to the grayscale image to smooth it while preserving edges.
    """
    smoothed_image = cv2.bilateralFilter(image, d=9, sigmaColor=sigma_t, sigmaSpace=sigma_s)
    return smoothed_image


def compute_gradient_magnitude(image):
    """
    Compute the gradient magnitude of the image using the Sobel operator.
    """
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
    return gradient_magnitude


def display_images(original_image, grayscale_image, smoothed_image, gradient_image):
    """
    Display the original, grayscale, smoothed, and gradient magnitude images.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(grayscale_image, cmap='gray')
    plt.title('Grayscale (Red Channel)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(smoothed_image, cmap='gray')
    plt.title('Smoothed Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Example usage
    image_path = 'firefly.png'

    # Step-by-step processing
    original_image = load_image(image_path)
    grayscale_image = convert_to_grayscale(original_image)
    smoothed_image = apply_bilateral_filter(grayscale_image)
    gradient_image = compute_gradient_magnitude(smoothed_image)

    # Display images
    display_images(original_image, grayscale_image, smoothed_image, gradient_image)
