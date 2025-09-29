import torch

def main():
    # Check if CUDA is available
    print("CUDA available:", torch.cuda.is_available())

    # If CUDA is available, print the device details
    if torch.cuda.is_available():
        print("Number of GPUs:", torch.cuda.device_count())
        print("Current GPU:", torch.cuda.current_device())
        print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))

if __name__ == "__main__":
    main()
