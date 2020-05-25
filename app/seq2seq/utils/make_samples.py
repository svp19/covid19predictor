import numpy as np

def make_samples(window_size, output_window_size, length, features, detrended_r, detrended_c, detrended_d, detrended_a):
    # ### Making Windows
    # 6 days input -> predict next 3 days output
    samples = []
    outputs = []

    # training samples with outputs 
    # Last Indices (14 ,20) -> (21, 22, 23)
    for i in range(length-window_size-output_window_size):
        # X_train
        sample = features.iloc[i:i + window_size, :].values
        sample = np.append(sample, detrended_r[i:i + window_size].reshape(-1, 1), axis=1)
        sample = np.append(sample, detrended_c[i:i + window_size].reshape(-1, 1), axis=1)
        sample = np.append(sample, detrended_d[i:i + window_size].reshape(-1, 1), axis=1)
        sample = np.append(sample, detrended_a[i:i + window_size].reshape(-1, 1), axis=1)
        output_confirmed = detrended_c[i + window_size: i + output_window_size + window_size]
        samples.append(sample)
        outputs.append(output_confirmed)

    samples = np.asarray(samples)
    outputs = np.asarray(outputs)
    
    # Test Samples (17, 23) -> (24, 25, 26)
    test_sample = features.iloc[-window_size:, :].values
    test_sample = np.append(test_sample, detrended_r[-window_size:].reshape(-1, 1), axis=1)
    test_sample = np.append(test_sample, detrended_c[-window_size:].reshape(-1, 1), axis=1)
    test_sample = np.append(test_sample, detrended_d[-window_size:].reshape(-1, 1), axis=1)
    test_sample = np.append(test_sample, detrended_a[-window_size:].reshape(-1, 1), axis=1)
    test_samples = np.asarray([test_sample])

    # Normalize non-detrended columns in Train samples
    for sample in samples:
        for i in np.r_[0:25, 27]:
            if np.mean(sample[:, i]) != 0:
                sample[:, i] /= np.mean(sample[:, i])

    # Normalize non-detrended columns in Test sample
    for sample in test_samples:
        for i in np.r_[0:25, 27]:
            if np.mean(sample[:, i]) != 0:
                sample[:, i] /= np.mean(sample[:, i])

    return samples, outputs, test_samples
