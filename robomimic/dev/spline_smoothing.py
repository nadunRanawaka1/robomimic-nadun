import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# Example noisy predictions
x = np.linspace(0, 10, 100)
predictions = np.sin(x) + np.random.normal(0, 0.5, size=x.shape)  # Noisy sine wave

# Fit a smoothing spline
spline = UnivariateSpline(x, predictions, s=1.0)  # Adjust s for smoothing
smoothed_predictions = spline(x)

# Plotting
plt.plot(x, predictions, label='Noisy Predictions', alpha=0.5)
plt.plot(x, smoothed_predictions, label='Smoothed Predictions (Spline)', color='red')
plt.legend()
plt.show()
