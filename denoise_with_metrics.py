import torch
import torch.nn as nn
import cv2
import numpy as np
import math
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nrmse
import os

class DnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        
        # First layer
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        # Middle layers
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        
        # Last layer
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        
        self.dncnn = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        noise = self.dncnn(x)
        return x - noise  # Residual learning

def calculate_metrics(img1, img2, clean_img=None):
    """Calculate image quality metrics for grayscale images"""
    metrics = {}
    
    # Convert to float32 in range [0, 1]
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0
    
    # Calculate metrics
    metrics['psnr'] = psnr(img1, img2, data_range=1.0)
    metrics['mse'] = mse(img1, img2)
    metrics['nrmse'] = nrmse(img1, img2)
    metrics['ssim'] = ssim(img1, img2, data_range=1.0)
    
    # If clean image is provided, calculate additional metrics
    if clean_img is not None:
        clean_img = clean_img.astype(np.float32) / 255.0
        metrics['psnr_clean'] = psnr(clean_img, img2, data_range=1.0)
        metrics['ssim_clean'] = ssim(clean_img, img2, data_range=1.0)
        metrics['mse_clean'] = mse(clean_img, img2)
        metrics['nrmse_clean'] = nrmse(clean_img, img2)
    
    return metrics

def main():
    try:
        # Initialize model
        print("Initializing DnCNN model...")
        # We'll use single channel for grayscale
        model = DnCNN(channels=1, num_of_layers=17)
        model.eval()
        
        # Load and process image
        print("Loading and processing image...")
        noisy_img_color = cv2.imread('noisy.jpeg')
        if noisy_img_color is None:
            print("Error: Could not load noisy.jpeg")
            return
            
        # Convert to grayscale if it's a color image
        if len(noisy_img_color.shape) == 3:
            noisy_img = cv2.cvtColor(noisy_img_color, cv2.COLOR_BGR2GRAY)
        else:
            noisy_img = noisy_img_color
            
        print(f"Input image shape: {noisy_img.shape}")
        print(f"Input image dtype: {noisy_img.dtype}")
        print(f"Input image min: {noisy_img.min()}, max: {noisy_img.max()}")
            
        # Try to load clean image if available
        clean_img = None
        try:
            clean_img_color = cv2.imread('clean.jpeg')
            if clean_img_color is not None:
                if len(clean_img_color.shape) == 3:
                    clean_img = cv2.cvtColor(clean_img_color, cv2.COLOR_BGR2GRAY)
                else:
                    clean_img = clean_img_color
                print("Clean image found for reference metrics")
        except:
            print("No clean image found, will only calculate relative metrics")
            
        print("Converting image format...")
        # Normalize to [0,1] range
        noisy_img_float = noisy_img.astype(np.float32)/255.0
        noisy_img_tensor = torch.from_numpy(noisy_img_float).unsqueeze(0).unsqueeze(0)
        
        print(f"Tensor shape: {noisy_img_tensor.shape}")
        print(f"Tensor dtype: {noisy_img_tensor.dtype}")
        print(f"Tensor min: {noisy_img_tensor.min().item()}, max: {noisy_img_tensor.max().item()}")
        
        # Denoise
        print("Applying DnCNN denoising...")
        with torch.no_grad():
            denoised_img_tensor = model(noisy_img_tensor)
        
        print(f"Output tensor shape: {denoised_img_tensor.shape}")
        print(f"Output tensor dtype: {denoised_img_tensor.dtype}")
        print(f"Output tensor min: {denoised_img_tensor.min().item()}, max: {denoised_img_tensor.max().item()}")
        
        # Convert back to numpy and uint8
        denoised_img = denoised_img_tensor.squeeze().numpy()
        # Ensure the output is in [0,1] range
        denoised_img = np.clip(denoised_img, 0, 1)
        # Convert to uint8 [0,255] range
        denoised_img = (denoised_img * 255.0).astype(np.uint8)
        
        print(f"Final image shape: {denoised_img.shape}")
        print(f"Final image dtype: {denoised_img.dtype}")
        print(f"Final image min: {denoised_img.min()}, max: {denoised_img.max()}")
        
        # Calculate metrics
        print("Calculating image quality metrics...")
        metrics = calculate_metrics(noisy_img, denoised_img, clean_img)
        
        # Print metrics
        print("\nImage Quality Metrics (Noisy to Denoised):")
        print(f"PSNR: {metrics['psnr']:.2f} dB")
        print(f"MSE: {metrics['mse']:.6f}")
        print(f"NRMSE: {metrics['nrmse']:.6f}")
        print(f"SSIM: {metrics['ssim']:.4f}")
        
        if clean_img is not None:
            print("\nReference Metrics (Clean to Denoised):")
            print(f"PSNR: {metrics['psnr_clean']:.2f} dB")
            print(f"MSE: {metrics['mse_clean']:.6f}")
            print(f"NRMSE: {metrics['nrmse_clean']:.6f}")
            print(f"SSIM: {metrics['ssim_clean']:.4f}")
        
        # Save result
        print("\nSaving result...")
        cv2.imwrite('denoised.jpeg', denoised_img)
        print("Denoising complete! Saved as denoised.jpeg")
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        print("Detailed error:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 