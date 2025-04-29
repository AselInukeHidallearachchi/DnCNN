import cv2
import numpy as np
import torch
from denoise_with_metrics import DnCNN, calculate_metrics
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

def denoise_image(model, noisy_img):
    """Apply denoising to an image"""
    noisy_img_float = noisy_img.astype(np.float32)/255.0
    noisy_img_tensor = torch.from_numpy(noisy_img_float).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        denoised_img_tensor = model(noisy_img_tensor)
    
    denoised_img = denoised_img_tensor.squeeze().numpy()
    denoised_img = np.clip(denoised_img, 0, 1)
    denoised_img = (denoised_img * 255.0).astype(np.uint8)
    
    return denoised_img

def plot_comparison(clean_img, noisy_img, denoised_img, noise_type, metrics):
    """Create a comparison plot with metrics"""
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3)
    
    # Clean image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(clean_img, cmap='gray')
    ax1.set_title('Clean Image')
    ax1.axis('off')
    
    # Noisy image
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(noisy_img, cmap='gray')
    ax2.set_title(f'Noisy Image ({noise_type})')
    ax2.axis('off')
    
    # Denoised image
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(denoised_img, cmap='gray')
    ax3.set_title('Denoised Image')
    ax3.axis('off')
    
    # Add metrics text
    metrics_text = (
        f"PSNR: {metrics['psnr']:.2f} dB\n"
        f"SSIM: {metrics['ssim']:.4f}\n"
        f"MSE: {metrics['mse']:.6f}\n"
        f"NRMSE: {metrics['nrmse']:.6f}"
    )
    plt.figtext(0.5, 0.01, metrics_text, ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'demo_results/{noise_type}_comparison.png')
    plt.close()

def plot_metrics_comparison(metrics_dict):
    """Create a bar chart comparing metrics across noise types"""
    noise_types = list(metrics_dict.keys())
    metrics = ['psnr', 'ssim', 'mse', 'nrmse']
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()
    
    for i, metric in enumerate(metrics):
        values = [metrics_dict[nt][metric] for nt in noise_types]
        axs[i].bar(noise_types, values)
        axs[i].set_title(f'{metric.upper()} Comparison')
        axs[i].set_ylabel(metric.upper())
        
        # Add value labels on top of bars
        for j, v in enumerate(values):
            axs[i].text(j, v, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('demo_results/metrics_comparison.png')
    plt.close()

def main():
    # Initialize model
    print("Initializing DnCNN model...")
    model = DnCNN(channels=1, num_of_layers=17)
    model.eval()
    
    # Create output directory
    os.makedirs('demo_results', exist_ok=True)
    
    # Load clean image
    clean_img = cv2.imread('testsets/Set12/01.png', 0)
    
    # Process each noise type
    noise_types = ['gaussian', 'salt_pepper', 'speckle']
    all_metrics = {}
    
    for noise_type in noise_types:
        print(f"\nProcessing {noise_type} noise...")
        
        # Load noisy image
        noisy_img = cv2.imread(f'demo_images/{noise_type}_noisy.png', 0)
        
        # Denoise image
        denoised_img = denoise_image(model, noisy_img)
        
        # Calculate metrics
        metrics = calculate_metrics(denoised_img, clean_img)
        all_metrics[noise_type] = metrics
        
        # Create comparison plot
        plot_comparison(clean_img, noisy_img, denoised_img, noise_type, metrics)
        
        # Save denoised image
        cv2.imwrite(f'demo_results/{noise_type}_denoised.png', denoised_img)
        
        print(f"Results saved for {noise_type} noise")
    
    # Create metrics comparison plot
    plot_metrics_comparison(all_metrics)
    
    print("\nDemonstration complete! Results saved in demo_results directory")

if __name__ == "__main__":
    main() 