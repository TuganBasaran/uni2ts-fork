import torch
import sys

print("Loading data...")
try:
    data = torch.load('/Users/tugan_basaran/Desktop/Lessons/CS.401/Kodlar/Moirai/uni2ts/grad/moirai_embeddings.pt', map_location='cpu', weights_only=False)
    print("Type:", type(data))
    if isinstance(data, dict):
        print("Length:", len(data))
        keys = list(data.keys())[:3]
        print("Sample keys:", keys)
        if keys:
            key1 = keys[0]
            val1 = data[key1]
            print("Key type:", type(key1), "Value type:", type(val1))
            if isinstance(val1, dict):
                inner_keys = list(val1.keys())[:3]
                print("Inner keys:", inner_keys)
                if inner_keys:
                    inner_val = val1[inner_keys[0]]
                    print("Inner value type:", type(inner_val), getattr(inner_val, "shape", None))
except Exception as e:
    print("Error:", e)
