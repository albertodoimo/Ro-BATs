import numpy as np

import matplotlib.pyplot as plt


#%%
class AdaptiveFilter:
    def __init__(self, filter_order, mu):
        self.filter_order = filter_order
        self.mu = mu
        self.weights = np.zeros(filter_order)

    def filter(self, input_signal, reference_signal):
        n_samples = len(input_signal)
        output_signal = np.zeros(n_samples)
        error_signal = np.zeros(n_samples)

        for n in range(self.filter_order, n_samples):
            x = reference_signal[n:n-self.filter_order:-1]
            y = np.dot(self.weights, x)
            error_signal[n] = input_signal[n] - y
            self.weights += 2 * self.mu * error_signal[n] * x

        return output_signal, error_signal

# Example usage
if __name__ == "__main__":
    # Generate sample signals
    np.random.seed(0)
    n_samples = 1000
    t = np.linspace(0, 1, n_samples)
    input_signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(n_samples)
    reference_signal = 0.5 * np.random.randn(n_samples)

    # Adaptive filter parameters
    filter_order = 5
    mu = 0.01

    # Create and apply adaptive filter
    adaptive_filter = AdaptiveFilter(filter_order, mu)
    output_signal, error_signal = adaptive_filter.filter(input_signal, reference_signal)

    # Plot results
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t, input_signal, label='Input Signal')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(t, reference_signal, label='Reference Signal')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(t, error_signal, label='Error Signal (Filtered Output)')
    plt.legend()
    plt.tight_layout()
    plt.show()

#%%

# implement  the adaptive filter with the LMS algorithm
class LMSFilter(AdaptiveFilter):
    def filter(self, input_signal, reference_signal):
        n_samples = len(input_signal)
        output_signal = np.zeros(n_samples)
        error_signal = np.zeros(n_samples)

        for n in range(self.filter_order, n_samples):
            x = reference_signal[n:n-self.filter_order:-1]
            y = np.dot(self.weights, x)
            output_signal[n] = y
            error_signal[n] = input_signal[n] - y
            self.weights += 2 * self.mu * error_signal[n] * x

        return output_signal, error_signal

# Example usage of LMSFilter
if __name__ == "__main__":
    # Generate sample signals
    np.random.seed(0)
    n_samples = 1000
    t = np.linspace(0, 1, n_samples)
    input_signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(n_samples)
    reference_signal = 0.5 * np.random.randn(n_samples)

    # Adaptive filter parameters
    filter_order = 5
    mu = 0.01

    # Create and apply LMS filter
    lms_filter = LMSFilter(filter_order, mu)
    output_signal, error_signal = lms_filter.filter(input_signal, reference_signal)

    # Plot results
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t, input_signal, label='Input Signal')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(t, reference_signal, label='Reference Signal')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(t, error_signal, label='Error Signal (Filtered Output)')
    plt.legend()
    plt.tight_layout()
    plt.show()


    