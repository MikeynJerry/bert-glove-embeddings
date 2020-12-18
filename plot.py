from matplotlib import pyplot as plt

import json

with open("losses.json", "r") as f:
    losses = json.load(f)

plt.figure(figsize=(8, 6))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(range(len(losses)), losses)
plt.title("L$_1$ Loss vs Epochs")
plt.show()
