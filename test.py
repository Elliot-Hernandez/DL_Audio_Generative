import numpy as np
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], np.array([[1, 2], [3, 4], [5, 6]]));
plt.savefig("spec_generate/spec_test.png")
plt.close()
