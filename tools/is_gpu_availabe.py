import torch

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available. GPU can be used.")
    else:
        print("CUDA is not available. GPU cannot be used.")