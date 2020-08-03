from __future__ import print_function

import numpy as np

import melbank


class ExpFilter:
    """Simple exponential smoothing filter"""

    def __init__(self, configs, val=0.0, alpha_decay=0.5, alpha_rise=0.5):
        """Small rise / decay factors = more smoothing"""
        assert 0.0 < alpha_decay < 1.0, 'Invalid decay smoothing factor'
        assert 0.0 < alpha_rise < 1.0, 'Invalid rise smoothing factor'
        self.alpha_decay = alpha_decay
        self.alpha_rise = alpha_rise
        self.value = val
        self.configs = configs
        self.mel_y = None
        self.mel_x = None
        self.configs = configs

        self.create_mel_bank()

    def update(self, value):
        if isinstance(self.value, (list, np.ndarray, tuple)):
            alpha = value - self.value
            alpha[alpha > 0.0] = self.alpha_rise
            alpha[alpha <= 0.0] = self.alpha_decay
        else:
            alpha = self.alpha_rise if value > self.value else self.alpha_decay
        self.value = alpha * value + (1.0 - alpha) * self.value
        return self.value

    def create_mel_bank(self):

        mic_rate = self.configs['mic_rate']
        n_fft_bins = self.configs['n_fft_bins']
        min_frequency = self.configs['min_frequency']
        max_frequency = self.configs['max_frequency']
        samples = int(self.configs['mic_rate'] * self.configs['n_rolling_history'] / (2.0 * self.configs['fps']))

        self.mel_y, (_, self.mel_x) = melbank.compute_melmat(num_mel_bands=n_fft_bins,
                                                             freq_min=min_frequency,
                                                             freq_max=max_frequency,
                                                             num_fft_bands=samples,
                                                             sample_rate=mic_rate)
