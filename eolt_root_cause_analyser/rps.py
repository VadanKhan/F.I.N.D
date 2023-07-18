import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sql_fetch import fetch_motor_details
from sql_fetch import fetch_step_timings
from tdms_fetch import form_filename_tdms
from tdms_fetch import form_filepath
from tdms_fetch import get_time_series_data
from tdms_fetch import read_tdms

# %% Testing Required inputs
good_rps_test = [23918, "High_Speed", 20140]
static_rps_test = [33931, "Cogging", 32537]
order_rps_test = [36288, "High_Speed", 31105]

eol_test_id_V = good_rps_test[0]
test_type_id_V = good_rps_test[1]
test_id_V = good_rps_test[2]

filename_V = form_filename_tdms(eol_test_id_V, test_type_id_V, test_id_V)
df_filepath_V = form_filepath(filename_V)
df_test_V = read_tdms(df_filepath_V)

# %%
print("_" * 60, "Database Check", "_" * 60)
df = df_test_V
# df = df.reset_index()
# print(f"{df}")
columns = list(df.columns)
# print(f"{columns}")
print("=" * 120, "\n")

# %% Hardcoded Settings
ZERO_RMS_UPPER_THRESHOLD = 0.01
SHORT_THRESHOLD_HIGH = 4.5
SHORT_THRESHOLD_LOW = 5.5
AVERAGING_SPREAD = 500
NORMAL_AVERAGE_LOW = 2.4
NORMAL_AVERAGE_HIGH = 2.6
DIFFERENTIAL_RMS_LOW = 0.05


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
    print("_" * 60, "Filtering", "_" * 60)
    step_durations: np.ndarray = step_dataframe["Duration_ms"].values / 1000
    accel_durations: np.ndarray = step_dataframe["Accel_Time_S"].values
    total_time = sum(step_durations)
    time_to_finish = sum(step_durations[:-1])
    time_to_start = step_durations[0] + accel_durations[1]
    print("test length:", total_time)
    print("time when start:", time_to_start)
    print("time when finish:", time_to_finish)
    lower_bound = time_to_start
    upper_bound = time_to_finish
    filter_index_array = np.where((time >= lower_bound) & (time <= upper_bound))[0]
    return filter_index_array


def rps_prefilter(df_filepath, df_test, eol_test_id):
    """Filters RPS data based on the start and end times of a test.

    This function takes in the file path to a DataFrame containing RPS data, a DataFrame containing test information, and an EOL test ID. The function filters the RPS data to only include values between the start and end times of the test, which are calculated based on the step durations and acceleration durations in the input DataFrame. The function returns a NumPy array containing the filtered RPS data.

    Args:
        df_filepath (str): The file path to a DataFrame containing RPS data.
        df_test (pd.DataFrame): A DataFrame containing test information.
        eol_test_id (int): An EOL test ID.

    Returns:
        np.ndarray: A NumPy array containing the filtered RPS data.
    """
    print("_" * 60, "RPS prefilter", "_" * 60)
    rps_channel_list = ["SinP", "SinN", "CosP", "CosN"]
    print("_" * 60, "Get RPS time values & convert to numpy", "_" * 60)
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
    print(f"{RPS_df}\n")
    rps_data_raw_np: np.ndarray = RPS_df.values
    print(f"As a numpy array:\n{rps_data_raw_np}\n")
    print("=" * 120, "\n")

    time_np = rps_data_raw_np[:, 0]

    print("_" * 60, "fetch test details", "_" * 60)
    motor_type = fetch_motor_details(eol_test_id)
    step_details_df: pd.DataFrame = fetch_step_timings(motor_type)
    print("=" * 120, "\n")

    filtered_index_array = edge_filtering(step_details_df, time_np)
    rps_data_np = rps_data_raw_np[filtered_index_array]
    # print(rps_data_np)

    # fig, (ax1, ax2) = plt.subplots(2, 1)
    # ax1.plot(rps_data_raw_np[:, 0], rps_data_raw_np[:, 1])
    # ax2.plot(rps_data_np[:, 0], rps_data_np[:, 1])
    # fig.suptitle("SinP: Before / After prefilter")

    # fig2, (ax1, ax2) = plt.subplots(2, 1)
    # ax1.plot(rps_data_raw_np[:, 0], rps_data_raw_np[:, 2])
    # ax2.plot(rps_data_np[:, 0], rps_data_np[:, 2])
    # fig2.suptitle("SinN: Before / After prefilter")

    # fig3, (ax1, ax2) = plt.subplots(2, 1)
    # ax1.plot(rps_data_raw_np[:, 0], rps_data_raw_np[:, 3])
    # ax2.plot(rps_data_np[:, 0], rps_data_np[:, 3])
    # fig3.suptitle("CosP: Before / After prefilter")

    # fig4, (ax1, ax2) = plt.subplots(2, 1)
    # ax1.plot(rps_data_raw_np[:, 0], rps_data_raw_np[:, 4])
    # ax2.plot(rps_data_np[:, 0], rps_data_np[:, 4])
    # fig4.suptitle("CosN: Before / After prefilter")

    # fig8, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    # ax1.plot(rps_data_np[:, 0], rps_data_np[:, 1])
    # ax2.plot(rps_data_np[:, 0], rps_data_np[:, 2])
    # ax3.plot(rps_data_np[:, 0], rps_data_np[:, 3])
    # ax4.plot(rps_data_np[:, 0], rps_data_np[:, 4])
    # ax1.set_ylim(0, 5)
    # ax2.set_ylim(0, 5)
    # ax3.set_ylim(0, 5)
    # ax4.set_ylim(0, 5)
    # fig8.suptitle("Input (Selected) Signals")

    print("=" * 120, "\n")

    return rps_data_np


def rps_signal_zero_checker(rps_data: np.ndarray):
    """
    This function checks for zero signals in the given RPS data.

    :param rps_data: A numpy array containing RPS data.
    :return: A list of strings indicating the status of each sensor. "Zero Signal" if the RMS value is less than or
        equal to 0.01, otherwise 0.
    """
    print("_" * 60, "zero signal checker", "_" * 60)
    rms_values = []
    for i in range(4):
        rms = np.sqrt(np.mean(rps_data[:, i + 1] ** 2))
        rms_values.append(rms)
    rps_signal_sensor_status = []
    for i in range(len(rms_values)):
        if rms_values[i] <= ZERO_RMS_UPPER_THRESHOLD:
            rps_signal_sensor_status.append("Zero Signal")
        else:
            rps_signal_sensor_status.append(0)
    print("=" * 120, "\n")
    return rps_signal_sensor_status, rms_values


def rps_signal_5V_checker(rps_data: np.ndarray):
    """
    This function checks for 5V signals in the given RPS data.

    :param rps_data: A numpy array containing RPS data.
    :return: A list of strings indicating the status of each sensor. "5V Signal" if the RMS value is between 4.5 and
        5.5, otherwise 0.
    """
    print("_" * 60, "5V signal checker", "_" * 60)
    mean_values = []
    for i in range(4):
        mean = np.mean(rps_data[:, i + 1])
        mean_values.append(mean)
    rps_signal_sensor_status = []
    for i in range(len(mean_values)):
        if mean_values[i] <= SHORT_THRESHOLD_HIGH and mean_values[i] >= SHORT_THRESHOLD_LOW:
            rps_signal_sensor_status.append("5V Signal")
        else:
            rps_signal_sensor_status.append(0)
    print("=" * 120, "\n")
    return rps_signal_sensor_status, mean_values


def moving_average_convolve(data: np.ndarray, spread: int):
    """
    Calculates the moving average of the input data with a rectangle window convolution, size defined by the spread
        parameter.

    :param data: A 1D numpy array containing the data to be averaged.
    :param spread: An integer defining the window size for the moving average calculation.
    :return: A 1D numpy array containing the moving average of the input data.
    """
    kernel = np.ones(spread) / spread
    averaged_data = np.convolve(data, kernel, mode="same")
    return averaged_data


# def moving_average_fast(data, spread):
#     cumsum = np.cumsum(data)
#     avg_vals = np.empty(len(data))
#     for i in range(len(data)):
#         if i >= spread:
#             avg_vals[i] = (cumsum[i] - cumsum[i - spread]) / spread
#         else:
#             avg_vals[i] = cumsum[i] / (i + 1)
#         # print(len(avg_vals))
#     return avg_vals


# def worker(data, start, end, spread):
#     cumsum = np.cumsum(data[start:end])
#     avg_vals = np.empty(end - start)
#     for i in range(start, end):
#         if i >= spread:
#             avg_vals[i - start] = (cumsum[i - start] - cumsum[i - start - spread]) / spread
#         else:
#             avg_vals[i - start] = cumsum[i - start] / (i + 1)
#     return avg_vals


# def moving_average_parallel(data, spread):
#     n_workers = 4
#     chunk_size = len(data) // n_workers
#     with ProcessPoolExecutor() as executor:
#         results = executor.map(
#             worker,
#             [data] * n_workers,
#             range(0, len(data), chunk_size),
#             range(chunk_size, len(data) + chunk_size, chunk_size),
#             [spread] * n_workers,
#         )
#     return np.concatenate(list(results))


# def frequency_filtering(data: np.ndarray):
#     fft_data = np.fft.fft(data)
#     freqs = np.fft.fftfreq(len(data))
#     fig7 = plt.figure()
#     plt.stem(range(len(fft_data)), np.abs(fft_data))
#     plt.xlabel("Frequency (cycles/sample)")
#     plt.ylabel("Magnitude")
#     return fft_data


def remove_centre_data(time, eol_test_id, gap_width):
    motor_type = fetch_motor_details(eol_test_id)
    step_dataframe: pd.DataFrame = fetch_step_timings(motor_type)
    step_durations: np.ndarray = step_dataframe["Duration_ms"].values / 1000
    accel_durations: np.ndarray = step_dataframe["Accel_Time_S"].values
    upto_step_three = np.sum(step_durations[0:3])
    halfway = upto_step_three + accel_durations[3] / 2
    print("Halfway time: ", halfway)
    delta = gap_width / 2
    gap_lower = halfway - delta
    gap_higher = halfway + delta
    filtered_index_array_lower = np.where(time < (halfway - delta))[0]
    filtered_index_array_higher = np.where(time > (halfway + delta))[0]
    return filtered_index_array_lower, filtered_index_array_higher


def rps_signal_static_checker(rps_data: np.ndarray):
    print("_" * 60, "average static checker", "_" * 60)
    diff_arrays: list[np.ndarray] = []
    for i in range(4):
        diff_arr = np.diff(rps_data[:, i + 1])
        diff_arrays.append(diff_arr)

    fig5, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    ax1.plot(rps_data[:-1, 0], diff_arrays[0])
    ax2.plot(rps_data[:-1, 0], diff_arrays[1])
    ax3.plot(rps_data[:-1, 0], diff_arrays[2])
    ax4.plot(rps_data[:-1, 0], diff_arrays[3])
    fig5.suptitle("Differential of Signals")

    smoothed_arrays: list[np.ndarray] = []
    for i in range(4):
        spread_input = AVERAGING_SPREAD  # input top-hat width
        smoothed_data = moving_average_convolve(rps_data[:, i + 1], spread_input)
        smoothed_arrays.append(smoothed_data)
    smoothed_data = np.column_stack(
        (rps_data[:, 0], smoothed_arrays[0], smoothed_arrays[1], smoothed_arrays[2], smoothed_arrays[3])
    )

    gap_index_lower, gap_index_higher = remove_centre_data(rps_data[:, 0], eol_test_id_V, 5)
    smoothed_gapped_data_lower = smoothed_data[gap_index_lower]
    smoothed_gapped_data_higher = smoothed_data[gap_index_higher]
    smoothed_gapped_data = np.vstack((smoothed_gapped_data_lower, smoothed_gapped_data_higher))
    smoothed_gapped_data = smoothed_gapped_data[spread_input:-spread_input]

    print("\n Smoothed Data length: ", len(smoothed_data[:, 0]))
    print("\n Smoothed Gapped Data length: ", len(smoothed_gapped_data[:, 0]))

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
    for i in range(4):
        non_normal_index = np.where(
            (smoothed_gapped_data[:, i + 1] > NORMAL_AVERAGE_HIGH)
            | (smoothed_gapped_data[:, i + 1] < NORMAL_AVERAGE_LOW)
        )[0]
        if non_normal_index.size <= 1000:
            non_normal_times.append(0)
        else:
            non_normal_time = smoothed_gapped_time[non_normal_index]
            non_normal_times.append(non_normal_time)

    differential_rms_values = []
    for i in range(4):
        rms_diff = np.sqrt(np.mean(diff_arrays[i] ** 2))
        differential_rms_values.append(rms_diff)

    average_status = []
    for non_normal_time in non_normal_times:
        if isinstance(non_normal_time, np.ndarray):
            # Handle the case where non_normal_time is an array
            average_status.append(f"Strange Rest Position from {non_normal_time[0]} to {non_normal_time[-1]}")
        elif non_normal_time == 0:
            # Handle the case where non_normal_time is equal to 0
            average_status.append(0)
        else:
            # Handle any other cases
            print("Error: Unexpected non_normal_time")

    differential_status = []
    for rms_diff in differential_rms_values:
        if rms_diff > DIFFERENTIAL_RMS_LOW:
            differential_status.append(0)
        else:
            differential_status.append("Signal Appears Static")

    overall_results = []
    for i in range(len(differential_status)):
        if average_status[i] == 0 and differential_status[i] == 0:
            overall_results.append(0)
        elif average_status[i] != 0 and differential_status[i] != 0:
            overall_results.append("Static Signal")
        elif average_status[i] != 0 and differential_status[i] == 0:
            overall_results.append("Appears Non Static but with Wrong Rest Value")
        elif average_status[i] == 0 and differential_status[i] != 0:
            overall_results.append("Appears Static but with Good Rest Value")
        else:
            overall_results.append("Unexpected average_status and or differential_status")

    print("=" * 120, "\n")
    return overall_results, average_status, differential_status, non_normal_times, differential_rms_values


def rps_order_checker(rps_data: np.ndarray):
    print("_" * 60, "order checker", "_" * 60)
    sinP = rps_data[:, 1]
    sinN = rps_data[:, 2]
    cosP = rps_data[:, 3]
    cosN = rps_data[:, 4]
    sinP_mean = np.mean(sinP)
    sinN_mean = np.mean(sinN)
    cosP_mean = np.mean(cosP)
    cosN_mean = np.mean(cosN)
    print(f"SinP mean: {sinP_mean}")
    print(f"SinN mean: {sinN_mean}")
    print(f"CosP mean: {cosP_mean}")
    print(f"CosN mean: {cosN_mean}")

    sin_check = (sinP - sinP_mean) + (sinN - sinN_mean)
    cos_check = (cosP - cosP_mean) + (cosN - cosN_mean)
    print(f"Sin Check Mean: {np.mean(sin_check)}")
    print(f"Cos Check Mean: {np.mean(cos_check)}")

    fig9, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(rps_data[:, 0], sin_check)
    ax2.plot(rps_data[:, 0], cos_check)
    # ax1.set_ylim(-5, 5)
    # ax2.set_ylim(-5, 5)
    ax1.set_title("sinP+sinN")
    ax2.set_title("cosP+cosN")

    fig10, ax1 = plt.subplots()
    ax1.plot(rps_data[:, 0], sinP, label="sinP")
    ax1.plot(rps_data[:, 0], sinN, label="sinN")
    ax1.plot(rps_data[:, 0], cosP, label="cosP")
    ax1.plot(rps_data[:, 0], cosN, label="cosN")
    ax1.set_xlim(20, 20.01)
    ax1.legend()
    print("=" * 120, "\n")
    return 0


# %% Toplevel Runner
if __name__ == "__main__":
    rps_data_np_V = rps_prefilter(df_filepath_V, df_test_V, eol_test_id_V)
    # rps_zero_status = rps_signal_zero_checker(rps_data_np_V)
    # rps_short_status = rps_signal_5V_checker(rps_data_np_V)
    # rps_static_status = rps_signal_static_checker(rps_data_np_V)
    rps_order_status = rps_order_checker(rps_data_np_V)

    print("_" * 60, "Results", "_" * 60)
    # print(rps_zero_status)
    # print(rps_short_status)
    # print(f"Overall Results: {rps_static_status[0]}")
    # print(f"Average Status: {rps_static_status[1]}")
    # print(f"Differential Status: {rps_static_status[2]}")
    # print(f"Non Normal Times: {rps_static_status[3]}")
    # print(f"Differential RMS values: {rps_static_status[4]}")
    print("=" * 120, "\n")
    plt.show()
