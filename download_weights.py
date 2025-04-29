import os
import requests
import torch

def download_file(url, filename):
    """Download a file from a URL to a local file."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

def main():
    # Create model_zoo directory if it doesn't exist
    os.makedirs('model_zoo', exist_ok=True)
    
    # URL for the pretrained weights
    url = "https://drive.google.com/uc?export=download&id=1_Sfun9eN6WiaXJXoOKmqtCQxXBL3QVV7"
    weights_path = 'model_zoo/dncnn_gray.pth'
    
    print(f"Downloading pretrained weights to {weights_path}...")
    try:
        download_file(url, weights_path)
        # Verify the downloaded file
        if os.path.exists(weights_path):
            # Try to load the weights to verify they're valid
            torch.load(weights_path, map_location=torch.device('cpu'))
            print("Download successful and weights verified!")
        else:
            print("Error: File was not downloaded successfully")
    except Exception as e:
        print(f"Error downloading weights: {str(e)}")
        print("\nPlease try downloading the weights manually from:")
        print("https://drive.google.com/file/d/1_Sfun9eN6WiaXJXoOKmqtCQxXBL3QVV7/view?usp=sharing")
        print("And place the file in the model_zoo directory as 'dncnn_gray.pth'")

if __name__ == "__main__":
    main() 