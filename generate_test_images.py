import cv2
import numpy as np
import os

def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to an image"""
    row, col = image.shape
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_and_pepper_noise(image, prob=0.05):
    """Add salt and pepper noise to an image"""
    noisy = np.copy(image)
    # Salt noise
    salt_mask = np.random.random(image.shape) < prob/2
    noisy[salt_mask] = 255
    # Pepper noise
    pepper_mask = np.random.random(image.shape) < prob/2
    noisy[pepper_mask] = 0
    return noisy

def add_speckle_noise(image, sigma=0.1):
    """Add speckle (multiplicative) noise to an image"""
    row, col = image.shape
    gauss = sigma * np.random.randn(row, col)
    noisy = image + image * gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def main():
    # Create test_images directory if it doesn't exist
    os.makedirs('test_images', exist_ok=True)
    
    # Create a simple synthetic image
    size = 256
    # Create a clean image with some patterns
    clean = np.zeros((size, size), dtype=np.uint8)
    
    # Add some geometric patterns
    cv2.circle(clean, (size//2, size//2), 50, 255, -1)
    cv2.rectangle(clean, (50, 50), (100, 100), 255, -1)
    cv2.line(clean, (150, 150), (200, 200), 255, 2)
    
    # Save clean image
    cv2.imwrite('test_images/clean.png', clean)
    
    # Generate noisy versions
    # Gaussian noise
    noisy_gaussian = add_gaussian_noise(clean)
    cv2.imwrite('test_images/noisy_gaussian.png', noisy_gaussian)
    
    # Salt and pepper noise
    noisy_sp = add_salt_and_pepper_noise(clean)
    cv2.imwrite('test_images/noisy_salt_pepper.png', noisy_sp)
    
    # Speckle noise
    noisy_speckle = add_speckle_noise(clean)
    cv2.imwrite('test_images/noisy_speckle.png', noisy_speckle)
    
    print("Test images generated successfully in test_images directory:")
    print("1. clean.png - Original clean image")
    print("2. noisy_gaussian.png - Image with Gaussian noise")
    print("3. noisy_salt_pepper.png - Image with Salt & Pepper noise")
    print("4. noisy_speckle.png - Image with Speckle noise")

if __name__ == "__main__":
    main() 