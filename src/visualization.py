from __future__ import division
from __future__ import print_function

import time

import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

import dsp
import led
import microphone
from config import CONFIGS

_time_prev = time.time() * 1000.0
"""The previous time that the frames_per_second() function was called"""

_fps = dsp.ExpFilter(CONFIGS, val=CONFIGS['fps'], alpha_decay=0.2, alpha_rise=0.2)
"""The low-pass filter used to estimate frames-per-second"""


def frames_per_second():
    """Return the estimated frames per second

    Returns the current estimate for frames-per-second (FPS).
    FPS is estimated by measured the amount of time that has elapsed since
    this function was previously called. The FPS estimate is low-pass filtered
    to reduce noise.

    This function is intended to be called one time for every iteration of
    the program's main loop.

    Returns
    -------
    fps : float
        Estimated frames-per-second. This value is low-pass filtered
        to reduce noise.
    """
    global _time_prev, _fps
    time_now = time.time() * 1000.0
    dt = time_now - _time_prev
    _time_prev = time_now
    if dt == 0.0:
        return _fps.value
    return _fps.update(1000.0 / dt)


def memoize(function):
    """Provides a decorator for memoizing functions"""
    from functools import wraps
    memo = {}

    @wraps(function)
    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv

    return wrapper


@memoize
def _normalized_linspace(size):
    return np.linspace(0, 1, size)


def interpolate(y, new_length):
    """Intelligently resizes the array by linearly interpolating the values

    Parameters
    ----------
    y : np.array
        Array that should be resized

    new_length : int
        The length of the new interpolated array

    Returns
    -------
    z : np.array
        New array with length of new_length that contains the interpolated
        values of y.
    """
    if len(y) == new_length:
        return y
    x_old = _normalized_linspace(len(y))
    x_new = _normalized_linspace(new_length)
    z = np.interp(x_new, x_old, y)
    return z


class Visualizer:

    def __init__(self, configs):

        self.configs = configs

        self.visualization_effect = self.visualize_scroll

        n_pixels = configs['n_pixels']
        n_fft_bins = configs['n_fft_bins']

        self.r_filt = dsp.ExpFilter(CONFIGS, np.tile(0.01, n_pixels // 2),
                                    alpha_decay=0.2, alpha_rise=0.99)
        self.g_filt = dsp.ExpFilter(CONFIGS, np.tile(0.01, n_pixels // 2),
                                    alpha_decay=0.05, alpha_rise=0.3)
        self.b_filt = dsp.ExpFilter(CONFIGS, np.tile(0.01, n_pixels // 2),
                                    alpha_decay=0.1, alpha_rise=0.5)
        self.common_mode = dsp.ExpFilter(CONFIGS, np.tile(0.01, n_pixels // 2),
                                         alpha_decay=0.99, alpha_rise=0.01)
        self.p_filt = dsp.ExpFilter(CONFIGS, np.tile(1, (3, n_pixels // 2)),
                                    alpha_decay=0.1, alpha_rise=0.99)
        self.p = np.tile(1.0, (3, n_pixels // 2))
        self.gain = dsp.ExpFilter(CONFIGS, np.tile(0.01, n_fft_bins),
                                  alpha_decay=0.001, alpha_rise=0.99)
        self._prev_spectrum = np.tile(0.01, n_pixels // 2)

        ################################################

        min_volume_threshold = configs['min_volume_threshold']
        mic_rate = configs['mic_rate']
        fps = configs['fps']
        n_rolling_history = configs['n_rolling_history']

        self.fft_plot_filter = dsp.ExpFilter(CONFIGS, np.tile(1e-1, n_fft_bins),
                                             alpha_decay=0.5, alpha_rise=0.99)
        self.mel_gain = dsp.ExpFilter(CONFIGS, np.tile(1e-1, n_fft_bins),
                                      alpha_decay=0.01, alpha_rise=0.99)
        self.mel_smoothing = dsp.ExpFilter(CONFIGS, np.tile(1e-1, n_fft_bins),
                                           alpha_decay=0.5, alpha_rise=0.99)
        self.volume = dsp.ExpFilter(CONFIGS, min_volume_threshold,
                                    alpha_decay=0.02, alpha_rise=0.02)
        self.fft_window = np.hamming(int(mic_rate / fps) * n_rolling_history)
        self.prev_fps_update = time.time()

        # Number of audio samples to read every time frame
        samples_per_frame = int(mic_rate / fps)

        # Array containing the rolling audio sample window
        self.y_roll = np.random.rand(n_rolling_history, samples_per_frame) / 1e16

        self.default_dsp= dsp.ExpFilter(CONFIGS)

    def visualize_scroll(self, y):
        """Effect that originates in the center and scrolls outwards"""
        y = y ** 2.0
        self.gain.update(y)
        y /= self.gain.value
        y *= 255.0
        r = int(np.max(y[:len(y) // 3]))
        g = int(np.max(y[len(y) // 3: 2 * len(y) // 3]))
        b = int(np.max(y[2 * len(y) // 3:]))
        # Scrolling effect window
        self.p[:, 1:] = self.p[:, :-1]
        self.p *= 0.98
        self.p = gaussian_filter1d(self.p, sigma=0.2)
        # Create new color originating at the center
        self.p[0, 0] = r
        self.p[1, 0] = g
        self.p[2, 0] = b
        # Update the LED strip
        return np.concatenate((self.p[:, ::-1], self.p), axis=1)

    def visualize_energy(self, y):
        """Effect that expands from the center with increasing sound energy"""
        y = np.copy(y)
        self.gain.update(y)
        y /= self.gain.value
        # Scale by the width of the LED strip
        y *= float((self.configs['n_pixels'] // 2) - 1)
        # Map color channels according to energy in the different freq bands
        scale = 0.9
        r = int(np.mean(y[:len(y) // 3] ** scale))
        g = int(np.mean(y[len(y) // 3: 2 * len(y) // 3] ** scale))
        b = int(np.mean(y[2 * len(y) // 3:] ** scale))
        # Assign color to different frequency regions
        self.p[0, :r] = 255.0
        self.p[0, r:] = 0.0
        self.p[1, :g] = 255.0
        self.p[1, g:] = 0.0
        self.p[2, :b] = 255.0
        self.p[2, b:] = 0.0
        self.p_filt.update(self.p)
        self.p = np.round(self.p_filt.value)
        # Apply substantial blur to smooth the edges
        self.p[0, :] = gaussian_filter1d(self.p[0, :], sigma=4.0)
        self.p[1, :] = gaussian_filter1d(self.p[1, :], sigma=4.0)
        self.p[2, :] = gaussian_filter1d(self.p[2, :], sigma=4.0)
        # Set the new self.pixel value
        return np.concatenate((self.p[:, ::-1], self.p), axis=1)

    def visualize_spectrum(self, y):
        """Effect that maps the Mel filterbank frequencies onto the LED strip"""
        y = np.copy(interpolate(y, self.configs['n_pixels'] // 2))
        self.common_mode.update(y)
        diff = y - self._prev_spectrum
        self._prev_spectrum = np.copy(y)
        # Color channel mappings
        r = self.r_filt.update(y - self.common_mode.value)
        g = np.abs(diff)
        b = self.b_filt.update(np.copy(y))
        # Mirror the color channels for symmetric output
        r = np.concatenate((r[::-1], r))
        g = np.concatenate((g[::-1], g))
        b = np.concatenate((b[::-1], b))
        output = np.array([r, g, b]) * 255
        return output

    def audio_to_rgb(self, audio_samples):
        """
        Given an audio sample return the rgb values for the leds
        """
        # Normalize samples between 0 and 1
        y = audio_samples / 2.0 ** 15
        # Construct a rolling window of audio samples
        self.y_roll[:-1] = self.y_roll[1:]
        self.y_roll[-1, :] = np.copy(y)
        y_data = np.concatenate(self.y_roll, axis=0).astype(np.float32)

        vol = np.max(np.abs(y_data))
        if vol < self.configs['min_volume_threshold']:
            print('No audio input. Volume below threshold. Volume:', vol)
            output = np.tile(0, (3, self.configs['n_pixels']))
        else:
            # Transform audio input into the frequency domain
            N = len(y_data)
            N_zeros = 2 ** int(np.ceil(np.log2(N))) - N
            # Pad with zeros until the next power of two
            y_data *= self.fft_window
            y_padded = np.pad(y_data, (0, N_zeros), mode='constant')
            YS = np.abs(np.fft.rfft(y_padded)[:N // 2])
            # Construct a Mel filterbank from the FFT data
            mel = np.atleast_2d(YS).T * self.default_dsp.mel_y.T
            # Scale data to values more suitable for visualization
            # mel = np.sum(mel, axis=0)
            mel = np.sum(mel, axis=0)
            mel = mel ** 2.0
            # Gain normalization
            self.mel_gain.update(np.max(gaussian_filter1d(mel, sigma=1.0)))
            mel /= self.mel_gain.value
            mel = self.mel_smoothing.update(mel)
            # Map filterbank output onto LED strip
            output = self.visualization_effect(mel)

        return output, mel

    def microphone_update(self, audio_samples):
        """
        Update the leds and gui with the rgb values taken from the audio sample
        """

        led.pixels, mel = self.audio_to_rgb(audio_samples)
        led.update()
        if self.configs['use_gui']:
            # Plot filterbank output
            x = np.linspace(self.configs['min_frequency'], self.configs['max_frequency'], len(mel))
            mel_curve.setData(x=x, y=self.fft_plot_filter.update(mel))
            # Plot the color channels
            r_curve.setData(y=led.pixels[0])
            g_curve.setData(y=led.pixels[1])
            b_curve.setData(y=led.pixels[2])
        if self.configs['use_gui']:
            app.processEvents()

        if self.configs['display_fps']:
            fps = frames_per_second()
            if time.time() - 0.5 > self.prev_fps_update:
                self.prev_fps_update = time.time()
                print('FPS {:.0f} / {:.0f}'.format(fps, self.configs['fps']))


if __name__ == '__main__':

    vis = Visualizer(CONFIGS)
    if CONFIGS['use_gui']:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtGui

        # Create GUI window
        app = QtGui.QApplication([])
        view = pg.GraphicsView()
        layout = pg.GraphicsLayout(border=(100, 100, 100))
        view.setCentralItem(layout)
        view.show()
        view.setWindowTitle('Visualization')
        view.resize(800, 600)
        # Mel filterbank plot
        fft_plot = layout.addPlot(title='Filterbank Output', colspan=3)
        fft_plot.setRange(yRange=[-0.1, 1.2])
        fft_plot.disableAutoRange(axis=pg.ViewBox.YAxis)
        x_data = np.array(range(1, CONFIGS['n_fft_bins'] + 1))
        mel_curve = pg.PlotCurveItem()
        mel_curve.setData(x=x_data, y=x_data * 0)
        fft_plot.addItem(mel_curve)
        # Visualization plot
        layout.nextRow()
        led_plot = layout.addPlot(title='Visualization Output', colspan=3)
        led_plot.setRange(yRange=[-5, 260])
        led_plot.disableAutoRange(axis=pg.ViewBox.YAxis)
        # Pen for each of the color channel curves
        r_pen = pg.mkPen((255, 30, 30, 200), width=4)
        g_pen = pg.mkPen((30, 255, 30, 200), width=4)
        b_pen = pg.mkPen((30, 30, 255, 200), width=4)
        # Color channel curves
        r_curve = pg.PlotCurveItem(pen=r_pen)
        g_curve = pg.PlotCurveItem(pen=g_pen)
        b_curve = pg.PlotCurveItem(pen=b_pen)
        # Define x data
        x_data = np.array(range(1, CONFIGS['n_pixels'] + 1))
        r_curve.setData(x=x_data, y=x_data * 0)
        g_curve.setData(x=x_data, y=x_data * 0)
        b_curve.setData(x=x_data, y=x_data * 0)
        # Add curves to plot
        led_plot.addItem(r_curve)
        led_plot.addItem(g_curve)
        led_plot.addItem(b_curve)
        # Frequency range label
        freq_label = pg.LabelItem('')


        # Frequency slider
        def freq_slider_change(tick):
            minf = freq_slider.tickValue(0) ** 2.0 * (CONFIGS['mic_rate'] / 2.0)
            maxf = freq_slider.tickValue(1) ** 2.0 * (CONFIGS['mic_rate'] / 2.0)
            t = 'Frequency range: {:.0f} - {:.0f} Hz'.format(minf, maxf)
            freq_label.setText(t)
            CONFIGS['min_frequency'] = minf
            CONFIGS['max_frequency'] = maxf
            dsp.create_mel_bank()


        freq_slider = pg.TickSliderItem(orientation='bottom', allowAdd=False)
        freq_slider.addTick((CONFIGS['min_frequency'] / (CONFIGS['mic_rate'] / 2.0)) ** 0.5)
        freq_slider.addTick((CONFIGS['max_frequency'] / (CONFIGS['mic_rate'] / 2.0)) ** 0.5)
        freq_slider.tickMoveFinished = freq_slider_change
        freq_label.setText('Frequency range: {} - {} Hz'.format(
            CONFIGS['min_frequency'],
            CONFIGS['max_frequency']))
        # Effect selection
        active_color = '#16dbeb'
        inactive_color = '#FFFFFF'


        def energy_click(x):
            vis.visualization_effect = vis.visualize_energy
            energy_label.setText('Energy', color=active_color)
            scroll_label.setText('Scroll', color=inactive_color)
            spectrum_label.setText('Spectrum', color=inactive_color)


        def scroll_click(x):
            vis.visualization_effect = vis.visualize_scroll
            energy_label.setText('Energy', color=inactive_color)
            scroll_label.setText('Scroll', color=active_color)
            spectrum_label.setText('Spectrum', color=inactive_color)


        def spectrum_click(x):
            vis.visualization_effect = vis.visualize_spectrum
            energy_label.setText('Energy', color=inactive_color)
            scroll_label.setText('Scroll', color=inactive_color)
            spectrum_label.setText('Spectrum', color=active_color)


        # Create effect "buttons" (labels with click event)
        energy_label = pg.LabelItem('Energy')
        scroll_label = pg.LabelItem('Scroll')
        spectrum_label = pg.LabelItem('Spectrum')
        energy_label.mousePressEvent = energy_click
        scroll_label.mousePressEvent = scroll_click
        spectrum_label.mousePressEvent = spectrum_click
        energy_click(0)
        # Layout
        layout.nextRow()
        layout.addItem(freq_label, colspan=3)
        layout.nextRow()
        layout.addItem(freq_slider, colspan=3)
        layout.nextRow()
        layout.addItem(energy_label)
        layout.addItem(scroll_label)
        layout.addItem(spectrum_label)
    # Initialize LEDs
    led.update()
    # Start listening to live audio stream
    microphone.start_stream(CONFIGS, vis.microphone_update)
