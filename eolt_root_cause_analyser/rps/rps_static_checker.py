import matplotlib.pyplot as plt
import numpy as np
from eolt_root_cause_analyser.data_analysis.data_analysis import moving_average_convolve
from eolt_root_cause_analyser.fetching.data_trimming import remove_centre_data

STATICCHECKER_TOPHAT_WIDTH = 500
STATICCHECKER_LOW_RESTVALUE = 2.4
STATICCHECKER_HIGH_RESTVALUE = 2.6
STATICCHECKER_LOW_DIFFERENTIAL_RMS = 0.05


def rps_static_checker(rps_data: np.ndarray, test_type: str, eol_test_id: int):
    """Checks for static signals in the given RPS data.

    This function takes a NumPy array containing RPS data, a test_type, and an eol_test_id as input and returns two
        lists of strings indicating the status of each sensor. The first list indicates whether the average value of
        each sensor is within a normal range, and the second list indicates whether the differential RMS value of each
        sensor is above a certain threshold.

    Args:
        rps_data (np.ndarray): A NumPy array containing RPS data.
        test_type (str): The type of test being performed.
        eol_test_id (int): The ID of the end-of-line test.

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

    gap_index_lower, gap_index_higher = remove_centre_data(rps_data[:, 0], eol_test_id, test_type, 5)
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
