import numpy as np
import pandas as pd
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
    step_dataframe: pd.DataFrame = fetch_step_timings(eol_test_id, test_type)
    step_durations: np.ndarray = step_dataframe["Duration_ms"].values / 1000
    accel_durations: np.ndarray = step_dataframe["Accel_Time_S"].values
    step_numbers = step_dataframe["Step_Number"].values
    num_steps = len(step_numbers)
    halfway_point = int(num_steps / 2)
    upto_halfway_durations = np.sum(step_durations[0:halfway_point])
    halfway = upto_halfway_durations + accel_durations[halfway_point] / 2
    # print("\nHalfway time: ", halfway)
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
    # total_time = sum(step_durations)
    time_to_finish = sum(step_durations[:-1])
    time_to_start = step_durations[0] + accel_durations[1]
    # print("\ntest length:", total_time)
    # print("time when start:", time_to_start)
    # print("time when finish:", time_to_finish)
    lower_bound = time_to_start
    upper_bound = time_to_finish
    filter_index_array = np.where((time >= lower_bound) & (time <= upper_bound))[0]
    return filter_index_array


def select_step(step_dataframe: pd.DataFrame, time: np.ndarray, step_input):
    """Selects a specific step from the input time array based on the given step_input.

    This function takes a 1D numpy array of time values, an eol_test_id, a test_type, and a step_input as input. It
    returns an array containing the indices of the time values that are within the specified step.

    Args:
        step_dataframe (pd.DataFrame): A DataFrame containing step information, including 'Duration_ms' and
            'Accel_
            Time_S' columns.
        time (np.ndarray): The time values of the input data.
        step_input (int): The step number to be selected.

    Returns:
        np.ndarray: An array containing the indices of the time values that are within the specified step.
    """
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
    # print(f"\nbounds: {lower_bound}, {upper_bound}")
    filtered_index_array = np.where((time > lower_bound) & (time < upper_bound))[0]
    # print(f"\nindex array {filtered_index_array}, \nlength of array {len(filtered_index_array)}")
    return filtered_index_array
