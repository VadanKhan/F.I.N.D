import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sql_fetch import fetch_motor_details
from sql_fetch import fetch_step_timings
from tdms_fetch import form_filepath
from tdms_fetch import get_time_series_data
from tdms_fetch import read_tdms

# %% Testing Required inputs
good_rps_test = [23918, "High_Speed", 20140]
static_rps_test = [33931, "Cogging", 32537]
order_uvw_test = [36288, "High_Speed", 31105]

eol_test_id_V = good_rps_test[0]
test_type_id_V = good_rps_test[1]
test_id_V = good_rps_test[2]

df_filepath_V = form_filepath(eol_test_id_V, test_type_id_V, test_id_V)
df_test_V = read_tdms(df_filepath_V)

# %% Hardcoded Settings
ZEROCHECKER_LOW_RMS = 0.01
SHORTCHECKER_LOW = 4.5
SHORTCHECKER_HIGH = 5.5
STATICCHECKER_TOPHAT_WIDTH = 500
STATICCHECKER_LOW_RESTVALUE = 2.4
STATICCHECKER_HIGH_RESTVALUE = 2.6
STATICCHECKER_LOW_DIFFERENTIAL_RMS = 0.05
ORDERCHECKER_SIGNAL_MSE_LOW = 1
ORDERCHECKER_POINTS = 100


# %%
def edge_filtering(step_dataframe: pd.DataFrame, time: np.ndarray):
    """Filters time values based on the start and end times of a test.

    This function takes in a DataFrame containing step information and a time array, and filters the time values to only
        include values between the start and end times of the test. The start and end times are calculated based on the
        step durations and acceleration durations in the input DataFrame. The function returns an index array containing
        the indices of the filtered time values.

    Args:
        step_dataframe (pd.DataFrame): A DataFrame containing step information, including 'Duration_ms' and
            'Accel_Time_S' columns.
        time (np.ndarray): A 1D NumPy array containing time values.

    Returns:
        np.ndarray: An index array containing the indices of the filtered time values.
    """
    step_durations: np.ndarray = step_dataframe["Duration_ms"].values / 1000
    accel_durations: np.ndarray = step_dataframe["Accel_Time_S"].values
    total_time = sum(step_durations)
    time_to_finish = sum(step_durations[:-1])
    time_to_start = step_durations[0] + accel_durations[1]
    print("\ntest length:", total_time)
    print("time when start:", time_to_start)
    print("time when finish:", time_to_finish)
    lower_bound = time_to_start
    upper_bound = time_to_finish
    filter_index_array = np.where((time >= lower_bound) & (time <= upper_bound))[0]
    return filter_index_array


def rps_prefilter(df_filepath, eol_test_id, test_type):
    """Filters RPS data based on the start and end times of a test.

    This function takes in the file path to a DataFrame containing RPS data, an EOL test ID, and a test type. The
        function filters the RPS data to only include values between the start and end times of the test, which are
        calculated based on the step durations and acceleration durations in a DataFrame fetched using the input EOL
        test ID and test type. The function returns a NumPy array containing the filtered RPS data.

    Args:
        df_filepath (str): The file path to a DataFrame containing RPS data.
        eol_test_id (int): An EOL test ID.
        test_type (str): The type of the test.

    Returns:
        np.ndarray: A NumPy array containing the filtered RPS data.
    """
    print("_" * 60, "RPS prefilter", "_" * 60)
    rps_channel_list = ["SinP", "SinN", "CosP", "CosN"]
    rps_time_list = get_time_series_data(df_filepath, rps_channel_list)

    # print(f"{rps_time_list[0]}")
    SinP_df = rps_time_list[0]
    SinN_df = rps_time_list[1]
    CosP_df = rps_time_list[2]
    CosN_df = rps_time_list[3]
    time = SinP_df.SinP_time
    time.name = "RPS_time"
    SinP_timevals_pd = SinP_df.SinP
    SinN_timevals_pd = SinN_df.SinN
    CosP_timevals_pd = CosP_df.CosP
    CosN_timevals_pd = CosN_df.CosN
    RPS_df = pd.concat([time, SinP_timevals_pd, SinN_timevals_pd, CosP_timevals_pd, CosN_timevals_pd], axis=1)
    rps_data_raw_np: np.ndarray = RPS_df.values

    time_np = rps_data_raw_np[:, 0]

    motor_type = fetch_motor_details(eol_test_id)
    step_details_df: pd.DataFrame = fetch_step_timings(motor_type, test_type)

    filtered_index_array = edge_filtering(step_details_df, time_np)
    rps_data_np = rps_data_raw_np[filtered_index_array]

    fig8, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)
    ax1.plot(rps_data_np[:, 0], rps_data_np[:, 1])
    ax2.plot(rps_data_np[:, 0], rps_data_np[:, 2])
    ax3.plot(rps_data_np[:, 0], rps_data_np[:, 3])
    ax4.plot(rps_data_np[:, 0], rps_data_np[:, 4])
    ax5.plot(rps_data_np[:, 0], rps_data_np[:, 1], label="SinP")
    ax5.plot(rps_data_np[:, 0], rps_data_np[:, 2], label="SinN")
    ax5.plot(rps_data_np[:, 0], rps_data_np[:, 3], label="CosP")
    ax5.plot(rps_data_np[:, 0], rps_data_np[:, 4], label="CosN")
    ax1.set_ylim(0, 5)
    ax2.set_ylim(0, 5)
    ax3.set_ylim(0, 5)
    ax4.set_ylim(0, 5)
    ax5.set_ylim(0, 5)
    ax5.legend()
    fig8.suptitle("Input (Selected) Signals")

    print("=" * 120, "\n")

    return rps_data_np


def rps_signal_zero_checker(rps_data: np.ndarray):
    """Checks for zero signals in the given RPS data.

    This function takes a NumPy array containing RPS data as input and returns a list of strings indicating the status
        of each sensor. If the RMS value of a sensor is less than or equal to 0.01, its status is "Zero Signal",
        otherwise it is 0.

    Args:
        rps_data (np.ndarray): A NumPy array containing RPS data.

    Returns:
        tuple: A tuple containing two elements: a list of strings indicating the status of each sensor, and a list of
        RMS values for each sensor.
    """

    print("_" * 60, "zero signal checker", "_" * 60)
    rms_values: list = []
    for column_index in range(len(rps_data[0, 1:])):
        rms = np.sqrt(np.mean(rps_data[:, column_index + 1] ** 2))
        rms_values.append(rms)
    rps_signal_sensor_status = []
    for rms_value in rms_values:
        if rms_value <= ZEROCHECKER_LOW_RMS:
            rps_signal_sensor_status.append("1")
        else:
            rps_signal_sensor_status.append("0")
    print("=" * 120, "\n")
    return rps_signal_sensor_status, rms_values


def rps_signal_5V_checker(rps_data: np.ndarray):
    """Checks for 5V signals in the given RPS data.

    This function takes a NumPy array containing RPS data as input and returns a list of strings indicating the status
        of each sensor. If the mean value of a sensor is between 4.5 and 5.5, its status is "5V Signal", otherwise it is
        0.

    Args:
        rps_data (np.ndarray): A NumPy array containing RPS data.

    Returns:
        tuple: A tuple containing two elements: a list of strings indicating the status of each sensor, and a list of
        mean values for each sensor.
    """
    print("_" * 60, "5V signal checker", "_" * 60)
    mean_values = []
    for column_index in range(len(rps_data[0, 1:])):
        mean = np.mean(rps_data[:, column_index + 1])
        mean_values.append(mean)
    rps_signal_sensor_status = []
    for mean_value in mean_values:
        if mean_value <= SHORTCHECKER_LOW and mean_value >= SHORTCHECKER_HIGH:
            rps_signal_sensor_status.append("1")
        else:
            rps_signal_sensor_status.append("0")
    print("=" * 120, "\n")
    return rps_signal_sensor_status, mean_values


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


def remove_centre_data(time: np.ndarray, eol_test_id, test_type, gap_width):
    """Removes data from the center of a given time array.

    This function takes a 1D numpy array of time values, an eol_test_id, a test_type, and a gap width as input. It
        returns two arrays containing the indices of the time values that are outside the specified gap around the
        center of the time array.

    Args:
        time (np.ndarray): The time values of the input data.
        eol_test_id (int): The eol_test_id of the motor.
        test_type (str): The type of test being performed.
        gap_width (float): The width of the gap to be removed from the center of the time array.

    Returns:
        tuple: A tuple containing two arrays: one with the indices of the time values that are below the lower bound
            of the gap, and one with the indices of the time values that are above the upper bound of the gap.
    """
    motor_type = fetch_motor_details(eol_test_id)
    step_dataframe: pd.DataFrame = fetch_step_timings(motor_type, test_type)
    step_durations: np.ndarray = step_dataframe["Duration_ms"].values / 1000
    accel_durations: np.ndarray = step_dataframe["Accel_Time_S"].values
    step_numbers = step_dataframe["Step_Number"].values
    num_steps = len(step_numbers)
    halfway_point = int(num_steps / 2)
    upto_halfway_durations = np.sum(step_durations[0:halfway_point])
    halfway = upto_halfway_durations + accel_durations[halfway_point] / 2
    print("\nHalfway time: ", halfway)
    delta = gap_width / 2
    filtered_index_array_lower = np.where(time < (halfway - delta))[0]
    filtered_index_array_higher = np.where(time > (halfway + delta))[0]
    return filtered_index_array_lower, filtered_index_array_higher


def rps_signal_static_checker(rps_data: np.ndarray, test_type):
    """Checks for static signals in the given RPS data.

    This function takes a NumPy array containing RPS data and a test_type as input and returns two lists of strings
        indicating the status of each sensor. The first list indicates whether the average value of each sensor is
        within a normal range, and the second list indicates whether the differential RMS value of each sensor is above
        a certain threshold.

    Args:
        rps_data (np.ndarray): A NumPy array containing RPS data.
        test_type (str): The type of test being performed.

    Returns:
        tuple: A tuple containing two lists of strings indicating the status of each sensor.
    """
    print("_" * 60, "static checker", "_" * 60)
    diff_arrays: list[np.ndarray] = []
    for column_index in range(len(rps_data[0, 1:])):
        diff_arr = np.diff(rps_data[:, column_index + 1])
        diff_arrays.append(diff_arr)

    fig5, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    ax1.plot(rps_data[:-1, 0], diff_arrays[0])
    ax2.plot(rps_data[:-1, 0], diff_arrays[1])
    ax3.plot(rps_data[:-1, 0], diff_arrays[2])
    ax4.plot(rps_data[:-1, 0], diff_arrays[3])
    fig5.suptitle("Differential of Signals")

    smoothed_arrays: list[np.ndarray] = []
    for column_index in range(len(rps_data[0, 1:])):
        spread_input = STATICCHECKER_TOPHAT_WIDTH  # input top-hat width
        smoothed_data = moving_average_convolve(rps_data[:, column_index + 1], spread_input)
        smoothed_arrays.append(smoothed_data)
    smoothed_data = np.column_stack(
        (rps_data[:, 0], smoothed_arrays[0], smoothed_arrays[1], smoothed_arrays[2], smoothed_arrays[3])
    )

    gap_index_lower, gap_index_higher = remove_centre_data(rps_data[:, 0], eol_test_id_V, test_type, 5)
    smoothed_gapped_data_lower = smoothed_data[gap_index_lower]
    smoothed_gapped_data_higher = smoothed_data[gap_index_higher]
    smoothed_gapped_data = np.vstack((smoothed_gapped_data_lower, smoothed_gapped_data_higher))
    smoothed_gapped_data = smoothed_gapped_data[spread_input:-spread_input]

    # fig6, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    # ax1.plot(rps_data[:, 0], smoothed_data[:, 1])
    # ax2.plot(rps_data[:, 0], smoothed_data[:, 2])
    # ax3.plot(rps_data[:, 0], smoothed_data[:, 3])
    # ax4.plot(rps_data[:, 0], smoothed_data[:, 4])
    # ax1.set_ylim(0, 5)
    # ax2.set_ylim(0, 5)
    # ax3.set_ylim(0, 5)
    # ax4.set_ylim(0, 5)
    # fig6.suptitle("Smoothed Signals")

    fig7, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    ax1.plot(smoothed_gapped_data[:, 0], smoothed_gapped_data[:, 1])
    ax2.plot(smoothed_gapped_data[:, 0], smoothed_gapped_data[:, 2])
    ax3.plot(smoothed_gapped_data[:, 0], smoothed_gapped_data[:, 3])
    ax4.plot(smoothed_gapped_data[:, 0], smoothed_gapped_data[:, 4])
    ax1.set_ylim(0, 5)
    ax2.set_ylim(0, 5)
    ax3.set_ylim(0, 5)
    ax4.set_ylim(0, 5)
    fig7.suptitle("Gapped Smooth Signals")

    non_normal_times = []
    smoothed_gapped_time = smoothed_gapped_data[:, 0]
    for column_index in range(len(smoothed_gapped_data[0, 1:])):
        non_normal_index = np.where(
            (smoothed_gapped_data[:, column_index + 1] > STATICCHECKER_HIGH_RESTVALUE)
            | (smoothed_gapped_data[:, column_index + 1] < STATICCHECKER_LOW_RESTVALUE)
        )[0]
        if non_normal_index.size <= 1000:
            non_normal_times.append(np.array([0]))
        else:
            non_normal_time = smoothed_gapped_time[non_normal_index]
            non_normal_times.append(non_normal_time)

    differential_rms_values = []
    for diff_array in diff_arrays:
        rms_diff = np.sqrt(np.mean(diff_array**2))
        differential_rms_values.append(rms_diff)

    average_status = []
    for non_normal_time in non_normal_times:
        if len(non_normal_time) > 1:
            # Handle the case where non_normal_time is an array
            average_status.append(f"Strange Rest Position from {non_normal_time[0]} to {non_normal_time[-1]}")
        elif len(non_normal_time) == 1:
            average_status.append(str(0))
        else:
            average_status.append("Averaging Error")
            return ["error", "error", "error", "error", "error"]

    differential_status = []
    for rms_diff in differential_rms_values:
        if rms_diff > STATICCHECKER_LOW_DIFFERENTIAL_RMS:
            differential_status.append(str(0))
        else:
            differential_status.append("Signal Appears Static")

    overall_results = []
    for avg, diff in zip(average_status, differential_status):
        if avg == str(0) and diff == str(0):
            overall_results.append(str(0))
        elif avg != str(0) and diff != str(0):
            overall_results.append("Static Signal")
        elif avg != str(0) and diff == str(0):
            overall_results.append("Appears Non Static but with Wrong Rest Value")
        elif avg == str(0) and diff != str(0):
            overall_results.append("Appears Static but with Good Rest Value")
        else:
            overall_results.append("Unexpected average_status and or differential_status")

    print("=" * 120, "\n")
    return overall_results, average_status, differential_status, non_normal_times, differential_rms_values


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

    return main_signal[time_shift:], shifted_signal[time_shift:], mse


def correct_order_checker(signal_list, time, period, sampling_time):
    """Checks if the given signals are in the correct order by aligning them and calculating their average absolute
        difference.

    This function takes a list of four signals, a 1D numpy array of time values, the period of the signals, and the
        sampling time as input. It returns a boolean value indicating whether the signals are in the correct order or
        not.

    Args:
        signal_list (list): A list containing four 1D numpy arrays representing the input signals.
        time (array_like): The time values of the input data.
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
    sinp_sinp = np.array([0, 0, 0])
    sinp_cosp = align_signals(cosP, sinP, period, 0.25, sampling_time)
    sinp_sinn = align_signals(sinN, sinP, period, 0.5, sampling_time)
    sinp_cosn = align_signals(cosN, sinP, period, 0.75, sampling_time)
    sinp_start_line = np.array([sinp_sinp[2], sinp_cosp[2], sinp_sinn[2], sinp_cosn[2]])

    # check cosp shifts correctly
    cosp_sinp = align_signals(sinP, cosP, period, 0.75, sampling_time)
    cosp_cosp = np.array([0, 0, 0])
    cosp_sinn = align_signals(sinN, cosP, period, 0.25, sampling_time)
    cosp_cosn = align_signals(cosN, cosP, period, 0.5, sampling_time)
    cosp_start_line = np.array([cosp_sinp[2], cosp_cosp[2], cosp_sinn[2], cosp_cosn[2]])

    # check sinn shifts correctly
    sinn_sinp = align_signals(sinP, sinN, period, 0.5, sampling_time)
    sinn_cosp = align_signals(cosP, sinN, period, 0.75, sampling_time)
    sinn_sinn = np.array([0, 0, 0])
    sinn_cosn = align_signals(cosN, sinN, period, 0.25, sampling_time)
    sinn_start_line = np.array([sinn_sinp[2], sinn_cosp[2], sinn_sinn[2], sinn_cosn[2]])

    # check cosnn shifts correctly
    cosn_sinp = align_signals(sinP, cosN, period, 0.25, sampling_time)
    cosn_cosp = align_signals(cosP, cosN, period, 0.5, sampling_time)
    cosn_sinn = align_signals(sinN, cosN, period, 0.75, sampling_time)
    cosn_cosn = np.array([0, 0, 0])
    cosn_start_line = np.array([cosn_sinp[2], cosn_cosp[2], cosn_sinn[2], cosn_cosn[2]])

    alignment_matrix = np.vstack((sinp_start_line, cosp_start_line, sinn_start_line, cosn_start_line))
    alignment_matrix = np.where(alignment_matrix > ORDERCHECKER_SIGNAL_MSE_LOW, 1, 0)
    print("\nAlignment Matrix:\n", alignment_matrix)

    if np.all(alignment_matrix == 0):
        return True
    else:
        return False


def select_step(time: np.ndarray, eol_test_id, test_type, step_input):
    """Selects a specific step from the input time array based on the given step_input.

    This function takes a 1D numpy array of time values, an eol_test_id, a test_type, and a step_input as input. It
    returns an array containing the indices of the time values that are within the specified step.

    Args:
        time (np.ndarray): The time values of the input data.
        eol_test_id (int): The eol_test_id of the motor.
        test_type (str): The type of test being performed.
        step_input (int): The step number to be selected.

    Returns:
        np.ndarray: An array containing the indices of the time values that are within the specified step.
    """
    motor_type = fetch_motor_details(eol_test_id)
    step_dataframe: pd.DataFrame = fetch_step_timings(motor_type, test_type)
    print(step_dataframe)
    step_durations: np.ndarray = step_dataframe["Duration_ms"].values / 1000
    accel_durations: np.ndarray = step_dataframe["Accel_Time_S"].values
    step_numbers: np.ndarray = step_dataframe["Step_Number"].values
    num_steps = len(step_numbers)
    step_input = step_input - 1  # reset step_input to count from 0
    if step_input + 1 > num_steps or step_input < 1:
        print(f"Invalid step input requested, number of steps: {num_steps}")
    elif step_input == 0:
        lower_bound = accel_durations[step_input]
        upper_bound = np.sum(step_durations[0 : (step_input + 1)])
    elif step_input > 0:
        lower_bound = np.sum(step_durations[0:step_input]) + accel_durations[step_input]
        upper_bound = np.sum(step_durations[0 : (step_input + 1)])
    else:
        print("Invalid step input requested, please input integer >= 1")
    print(f"\nbounds: {lower_bound}, {upper_bound}")
    filtered_index_array = np.where((time > lower_bound) & (time < upper_bound))[0]
    print(f"\nindex array {filtered_index_array}, \nlength of array {len(filtered_index_array)}")
    return filtered_index_array


def rps_order_checker(rps_data: np.ndarray, step_chosen, eol_test_id, test_type):
    """Checks the order of the given RPS signals.

    This function takes a NumPy array containing RPS data, a step_chosen, an eol_test_id, and a test_type as input and
    returns a list of integers indicating the status of each sensor and a list of strings indicating the correct order
    of the sensors. If the sensors are in the correct order, the status of each sensor is 0, otherwise it is 1.

    Args:
        rps_data (np.ndarray): A NumPy array containing RPS data.
        step_chosen (int): The step number chosen for analysis.
        eol_test_id (int): The eol_test_id of the motor.
        test_type (str): The type of test being performed.

    Returns:
        tuple: A tuple containing a list of integers indicating the status of each sensor and a list of strings
        indicating the correct order of the sensors.
    """
    print("_" * 60, "order checker", "_" * 60)
    step_index = select_step(rps_data[:, 0], eol_test_id, test_type, step_chosen)
    step_points = len(step_index)
    num_points_selected = ORDERCHECKER_POINTS
    lower_bound = step_index[0] + int(step_points / 2)
    print(lower_bound)
    upper_bound = step_index[0] + int(step_points / 2) + num_points_selected
    print(upper_bound)
    try:
        # code that may raise an IndexError
        time = rps_data[lower_bound:upper_bound, 0]
    except IndexError:
        # code to handle the error
        print(f"Index {lower_bound} to {upper_bound} is out of range for the array")
        return ["selecting step error: ", "selecting step error"]
    except Exception as e:
        print(f"Unexpected selecting step error {e}")
        return ["selecting step error: ", e]
    sampling_time = np.mean(np.diff(time))
    sinP = rps_data[lower_bound:upper_bound, 1]  # sinP
    cosP = rps_data[lower_bound:upper_bound, 3]  # cosP
    sinN = rps_data[lower_bound:upper_bound, 2]  # sinN
    cosN = rps_data[lower_bound:upper_bound, 4]  # cosN
    base_signals_list = [sinP, cosP, sinN, cosN]
    # here to easily edit the order of the input signals manually
    sinP = base_signals_list[0]
    cosP = base_signals_list[1]
    sinN = base_signals_list[2]
    cosN = base_signals_list[0]
    base_signals_list = [sinP, cosP, sinN, cosN]
    signal_names = ["1", "2", "3", "4"]
    signals = np.column_stack((sinP, sinN, cosP, cosN))

    fig12, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)
    ax1.plot(time, sinP)
    ax2.plot(time, cosP)
    ax3.plot(time, sinN)
    ax4.plot(time, cosN)
    ax5.plot(time, sinP, label="SinP")
    ax5.plot(time, cosP, label="CosP")
    ax5.plot(time, sinN, label="SinN")
    ax5.plot(time, cosN, label="CosN")
    plt.legend()

    try:
        frequencies = calculate_frequencies(time, signals)
        T = 1 / np.mean(frequencies)
    except Exception as e:
        print(f"Error calculating frequencies: {e}")
        return ["FFT error", "FFT error"]

    # main_signal, shifted_signal, error = align_signals(cosN, cosP, T, 0.5, sampling_time)
    # print("\nPlotting Error: ", error, "\n")

    # fig14, (ax1) = plt.subplots()
    # ax1.plot(time[-len(main_signal) :], main_signal, label="main")
    # ax1.plot(time[-len(shifted_signal) :], shifted_signal, label="shifted")
    # ax1.legend()

    correct_order = []
    for permutation in itertools.permutations(base_signals_list):
        if correct_order_checker(permutation, time, T, sampling_time):
            # returns true only when we have found the correct permutation of the signals
            for signal_permutation in permutation:
                """for each value in permutation, finds the matching signal in base_signals_list. It uses the index of
                that to find the corresponding signal name in the signal_names list, and so appends a name to the
                correct_order list. When it completely iterates across the permutation, the correct_order list will have
                names in the same order as the permutation."""
                correct_order.append(
                    signal_names[
                        next(
                            position_index
                            for position_index, signal_base in enumerate(base_signals_list)
                            if np.array_equal(signal_base, signal_permutation)
                        )
                    ]
                )
            break
    if len(correct_order) == 0:
        correct_order.append("Failed to find appropriate permutation")

    print("=" * 120, "\n")

    if correct_order == ["1", "2", "3", "4"]:
        rps_pinning_status = ["0", "0", "0", "0"]
    else:
        rps_pinning_status = ["1", "1", "1", "1"]
    return rps_pinning_status, correct_order


# %% Toplevel Runner
if __name__ == "__main__":
    rps_data_np_V = rps_prefilter(df_filepath_V, eol_test_id_V, test_type_id_V)
    rps_zero_status = rps_signal_zero_checker(rps_data_np_V)
    rps_short_status = rps_signal_5V_checker(rps_data_np_V)
    rps_static_status = rps_signal_static_checker(rps_data_np_V, test_type_id_V)
    rps_order_status = rps_order_checker(rps_data_np_V, 2, eol_test_id_V, test_type_id_V)  # checking with step 2
    print("_" * 60, "Results", "_" * 60)
    print(f"Zero Signal Checker: Overall Results: {rps_zero_status[0]}")
    print(f"Shorted Signal Checker: Overall Results: {rps_short_status[0]}")
    print(f"Static Checker: Overall Results: {rps_static_status[0]}")
    # print(f"Static Checker: Average Status: {rps_static_status[1]}")
    # print(f"Static Checker: Differential Status: {rps_static_status[2]}")
    # print(f"Static Checker: Non Normal Times: {rps_static_status[3]}")
    # print(f"Static Checker: Differential RMS values: {rps_static_status[4]}")
    print(f"Order Checker: Overall Results: {rps_order_status[0]}")
    print(f"Order Checker: Correct order of signals: {rps_order_status[1]}")
    print("=" * 120, "\n")
    # plt.show()
