import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.bar(["A", "B", "C"], [10, 20, 30])
print("Showing non-blocking window...")
plt.show(block=False)
plt.pause(0.5)
print("Non-blocking window shown successfully!")
plt.close(fig)
