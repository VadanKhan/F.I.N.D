import matplotlib.pyplot as plt
import numpy as np


def moving_average_convolve(data: np.ndarray, spread: int):
    """Calculates the moving average of the input data with a rectangle window convolution.

    This function takes a 1D NumPy array containing the data to be averaged and an integer defining the window size for
        the moving average calculation as input. It returns a 1D NumPy array containing the moving average of the input
        data.

    Args:
        data (np.ndarray): A 1D NumPy array containing the data to be averaged.
        spread (int): An integer defining the window size for the moving average calculation.

    Returns:
        np.ndarray: A 1D NumPy array containing the moving average of the input data.
    """
    kernel = np.ones(spread) / spread
    averaged_data = np.convolve(data, kernel, mode="same")
    return averaged_data


def normalise_signal(signal):
    """Normalizes a given signal so that its values fall between 0 and 1.

    This function takes a signal as input, calculates its minimum and maximum values, and uses them to rescale the
        signal so that its values fall between 0 and 1.

    Args:
        signal (array_like): The input signal to be normalized.

    Returns:
        array_like: The normalized version of the input signal with values between 0 and 1.
    """
    min_val = np.min(signal)
    max_val = np.max(signal)
    return (signal - min_val) / (max_val - min_val)


def calculate_frequencies(time, signals):
    """Calculates the frequency of each signal in the given data.

    This function takes a 1D numpy array of time values and a 2D numpy array of signals as input, where each column of
    the signals array represents a different signal, and returns a list containing the frequency of each signal.

    Args:
        time (array_like): The time values of the input data.
        signals (array_like): The input signals, where each column represents a different signal.

    Returns:
        list: A list containing the frequency of each signal.
    """

    # Calculate FFT of each signal
    fft_values = np.fft.fft(signals, axis=0)

    # Calculate power spectrum of each signal
    L = signals.shape[0]
    P2 = np.abs(fft_values / L)
    P1 = P2[: L // 2 + 1]
    P1[1:-1] = 2 * P1[1:-1]

    # Ignore 0 Hz frequency component
    P1[0] = 0

    # Find index of peak frequency for each signal
    peak_indices = np.argmax(P1, axis=0)

    # Calculate frequency values
    Fs = 1 / np.mean(np.diff(time))  # Sampling frequency
    f_values = Fs * np.arange(L // 2 + 1) / L

    fig13, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    ax1.bar(f_values, P1[:, 0], width=10)
    ax2.bar(f_values, P1[:, 1], width=10)
    ax3.bar(f_values, P1[:, 2], width=10)
    ax4.bar(f_values, P1[:, 3], width=10)

    # Calculate frequency of each signal
    frequencies = f_values[peak_indices]

    return frequencies


def align_signals(main_signal, move_signal, period, expected_phase_shift, sampling_time):
    """Aligns two given signals and calculates their mean squared error.

    This function takes a 1D numpy array of time values, two 1D numpy arrays representing the two input signals,
    the period of the signals, and the expected phase shift between them as input. It returns the aligned portions
    of the two signals and their mean squared error.

    Note:
        IT EXPECTS THE PHASE SHIFT IN UNITS OF 1 (eg. if signals differ in phase by pi/2, input 0.25)

    Args:
        main_signal (array_like): The first input signal.
        move_signal (array_like): The second input signal.
        period (float): The period of the signals.
        expected_phase_shift (float): The expected phase shift between the two signals.
        sampling_time (float): The time interval between samples.

    Returns:
        tuple: A tuple containing three elements: the aligned portion of the first signal, the aligned portion of the
            second signal, and their mean squared error.
    """

    # Shift second signal to overlap with first signal
    time_shift = round((expected_phase_shift * period) / sampling_time)
    shifted_signal = np.roll(move_signal, time_shift)

    # Calculate mean squared error between shifted signals, ignoring rolled values
    mse = np.mean((main_signal[time_shift:] - shifted_signal[time_shift:]) ** 2)

    return mse


def correct_order_checker(signal_list, threshold, period, sampling_time):
    """Checks if the given signals are in the correct order by aligning them and calculating their average absolute
        difference. Specific to the o

    This function takes a list of four signals, a 1D numpy array of time values, the period of the signals, and the
        sampling time as input. It returns a boolean value indicating whether the signals are in the correct order or
        not.

    Args:
        signal_list (list): A list containing four 1D numpy arrays representing the input signals.
        threshold (float): The threshold mean squared error for which we determine that the signals are non aligned.
        period (float): The period of the signals.
        sampling_time (float): The sampling time of the signals.

    Returns:
        bool: A boolean value indicating whether the signals are in the correct order or not.
    """
    sinP = signal_list[0]
    cosP = signal_list[1]
    sinN = signal_list[2]
    cosN = signal_list[3]

    # check sinP shifts correctly
    sinp_sinp = 0
    sinp_cosp = align_signals(cosP, sinP, period, 0.25, sampling_time)
    sinp_sinn = align_signals(sinN, sinP, period, 0.5, sampling_time)
    sinp_cosn = align_signals(cosN, sinP, period, 0.75, sampling_time)
    sinp_start_line = np.array([sinp_sinp, sinp_cosp, sinp_sinn, sinp_cosn])

    # check cosp shifts correctly
    cosp_sinp = align_signals(sinP, cosP, period, 0.75, sampling_time)
    cosp_cosp = 0
    cosp_sinn = align_signals(sinN, cosP, period, 0.25, sampling_time)
    cosp_cosn = align_signals(cosN, cosP, period, 0.5, sampling_time)
    cosp_start_line = np.array([cosp_sinp, cosp_cosp, cosp_sinn, cosp_cosn])

    # check sinn shifts correctly
    sinn_sinp = align_signals(sinP, sinN, period, 0.5, sampling_time)
    sinn_cosp = align_signals(cosP, sinN, period, 0.75, sampling_time)
    sinn_sinn = 0
    sinn_cosn = align_signals(cosN, sinN, period, 0.25, sampling_time)
    sinn_start_line = np.array([sinn_sinp, sinn_cosp, sinn_sinn, sinn_cosn])

    # check cosnn shifts correctly
    cosn_sinp = align_signals(sinP, cosN, period, 0.25, sampling_time)
    cosn_cosp = align_signals(cosP, cosN, period, 0.5, sampling_time)
    cosn_sinn = align_signals(sinN, cosN, period, 0.75, sampling_time)
    cosn_cosn = 0
    cosn_start_line = np.array([cosn_sinp, cosn_cosp, cosn_sinn, cosn_cosn])

    alignment_matrix = np.vstack((sinp_start_line, cosp_start_line, sinn_start_line, cosn_start_line))
    alignment_matrix = np.where(alignment_matrix > threshold, 1, 0)
    print("\nAlignment Matrix:\n", alignment_matrix)

    if np.all(alignment_matrix == 0):
        return True
    else:
        return False
