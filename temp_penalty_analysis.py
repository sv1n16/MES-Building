import numpy as np
import matplotlib.pyplot as plt

weights = np.linspace(0, 500, 10)

delta_T = np.linspace(-5, 5, 200)

plt.figure()
for w in weights:
    plt.plot(delta_T, w * delta_T**2, label=f"w = {w}")

plt.xlabel("Temperature deviation (°C)")
plt.ylabel("Comfort penalty")
plt.title("Effect of comfort weight")
plt.legend()
plt.grid(True)
plt.show()
