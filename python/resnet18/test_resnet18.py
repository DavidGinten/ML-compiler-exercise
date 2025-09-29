from transformers import AutoModelForImageClassification
import torch
# import torch.nn.functional as F # For Softmax
import time, os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
    
resnet18 = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
resnet18.eval()

def print_output():
    output = resnet18(torch.ones((1, 3, 224, 224)))
    #probs = F.softmax(output.logits , dim=1)
    #print(' '.join(f'{x:.4f}' for x in out.view(-1)))
    #print("Input: ", x)
    print(output.logits)

def benchmark():
    x = torch.ones((32, 1, 28, 28))
    #with torch.no_grad():
    # Warm-up
    for _ in range(10):
        _ = resnet18(x)

    start = time.time()
    for _ in range(100):
        _ = resnet18(x)
    end = time.time()

    print(f"Avg inference time: {(end - start) / 100:.6f} sec")



if __name__ == "__main__":
   print_output()
   #benchmark()