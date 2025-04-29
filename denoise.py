import torch
import torch.nn as nn
import cv2
import numpy as np
import math

class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=17):
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
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
                m.bias.data.zero_()
    
    def forward(self, x):
        out = self.dncnn(x)
        return out

def main():
    try:
        # Initialize model
        print("Initializing DnCNN model...")
        model = DnCNN(channels=1, num_of_layers=17)
        model.eval()
        
        # Load and process image
        print("Loading and processing image...")
        try:
            noisy_img = cv2.imread('noisy.png', 0)  # 0 for grayscale
            if noisy_img is None:
                print("Error: Could not load noisy.png")
                return
        except Exception as e:
            print(f"Error reading image file: {str(e)}")
            return
        
        print("Converting image format...")
        noisy_img = noisy_img.astype(np.float32)/255.0
        noisy_img = torch.from_numpy(noisy_img).unsqueeze(0).unsqueeze(0)
        
        # Denoise
        print("Applying DnCNN denoising...")
        with torch.no_grad():
            denoised_img = model(noisy_img)
        
        # Save result
        print("Saving result...")
        denoised_img = denoised_img.squeeze().numpy()*255.0
        denoised_img = np.clip(denoised_img, 0, 255).astype(np.uint8)
        cv2.imwrite('denoised.png', denoised_img)
        print("Denoising complete! Saved as denoised.png")
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        print("Detailed error:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()