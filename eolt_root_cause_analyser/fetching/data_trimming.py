import numpy as np
import pandas as pd
from eolt_root_cause_analyser.fetching.sql_fetch import fetch_motor_details
from eolt_root_cause_analyser.fetching.sql_fetch import fetch_step_timings


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
