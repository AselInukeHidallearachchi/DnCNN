import cv2
import numpy as np
import torch
from denoise_with_metrics import DnCNN, calculate_metrics
import os

def denoise_image(model, noisy_img):
    """Apply denoising to an image"""
    # Normalize to [0,1] range
    noisy_img_float = noisy_img.astype(np.float32)/255.0
    noisy_img_tensor = torch.from_numpy(noisy_img_float).unsqueeze(0).unsqueeze(0)
    
    # Denoise
    with torch.no_grad():
        denoised_img_tensor = model(noisy_img_tensor)
    
    # Convert back to numpy and uint8
    denoised_img = denoised_img_tensor.squeeze().numpy()
    denoised_img = np.clip(denoised_img, 0, 1)
    denoised_img = (denoised_img * 255.0).astype(np.uint8)
    
    return denoised_img

def main():
    # Initialize model
    print("Initializing DnCNN model...")
    model = DnCNN(channels=1, num_of_layers=17)
    model.eval()
    
    # Create output directory
    os.makedirs('test_results', exist_ok=True)
    
    # List of noise types to test
    noise_types = ['gaussian', 'salt_pepper', 'speckle']
    
    # Process each noise type
    for noise_type in noise_types:
        print(f"\n{'='*50}")
        print(f"Processing {noise_type} noise:")
        print(f"{'='*50}")
        
        # Load images
        noisy_path = f'test_images/noisy_{noise_type}.png'
        clean_path = 'test_images/clean.png'
        
        noisy_img = cv2.imread(noisy_path, 0)  # Read as grayscale
        clean_img = cv2.imread(clean_path, 0)  # Read as grayscale
        
        if noisy_img is None or clean_img is None:
            print(f"Error: Could not load images for {noise_type} noise")
            continue
        
        # Calculate metrics for noisy image
        noisy_metrics = calculate_metrics(noisy_img, clean_img)
        
        # Denoise image
        denoised_img = denoise_image(model, noisy_img)
        
        # Calculate metrics for denoised image
        denoised_metrics = calculate_metrics(denoised_img, clean_img)
        
        # Save results
        output_path = f'test_results/denoised_{noise_type}.png'
        cv2.imwrite(output_path, denoised_img)
        
        # Print metrics
        print("\nMetrics comparing to clean image:")
        print("\nNoisy Image Metrics:")
        print(f"PSNR: {noisy_metrics['psnr']:.2f} dB")
        print(f"SSIM: {noisy_metrics['ssim']:.4f}")
        print(f"MSE: {noisy_metrics['mse']:.6f}")
        print(f"NRMSE: {noisy_metrics['nrmse']:.6f}")
        
        print("\nDenoised Image Metrics:")
        print(f"PSNR: {denoised_metrics['psnr']:.2f} dB")
        print(f"SSIM: {denoised_metrics['ssim']:.4f}")
        print(f"MSE: {denoised_metrics['mse']:.6f}")
        print(f"NRMSE: {denoised_metrics['nrmse']:.6f}")
        
        # Calculate improvement
        psnr_improvement = denoised_metrics['psnr'] - noisy_metrics['psnr']
        ssim_improvement = denoised_metrics['ssim'] - noisy_metrics['ssim']
        mse_improvement = noisy_metrics['mse'] - denoised_metrics['mse']
        nrmse_improvement = noisy_metrics['nrmse'] - denoised_metrics['nrmse']
        
        print("\nImprovement after denoising:")
        print(f"PSNR Improvement: {psnr_improvement:+.2f} dB")
        print(f"SSIM Improvement: {ssim_improvement:+.4f}")
        print(f"MSE Reduction: {mse_improvement:+.6f}")
        print(f"NRMSE Reduction: {nrmse_improvement:+.6f}")
    
    print("\nDenoising complete! Results saved in test_results directory")

if __name__ == "__main__":
    main() 