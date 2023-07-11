import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sql_fetch import fetch_motor_details
from sql_fetch import fetch_step_timings
from tdms_fetch import form_filepath
from tdms_fetch import get_time_series_data
from tdms_fetch import read_tdms

# %% Testing Required inputs
test_id_V = 20140
test_type_id_V = "High_Speed"
eol_test_id_V = 23918
df_filepath_V = form_filepath("23918_High_Speed_20140.tdms")
df_test_V = read_tdms(df_filepath_V)

# %%
print("_" * 60, "Database Check", "_" * 60)
df = df_test_V
# df = df.reset_index()
# print(f"{df}")
columns = list(df.columns)
# print(f"{columns}")
print("=" * 120, "\n")


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


# %%
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
    rps_group_list = ["AI DAQ - High Speed Inc RTD"] * 4
    rps_channel_list = ["SinP", "SinN", "CosP", "CosN"]
    rps_time_list = get_time_series_data(df_filepath, rps_group_list, rps_channel_list)
    print("_" * 60, "Get RPS time values & convert to numpy", "_" * 60)
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
    print(rps_data_np)
    print("=" * 120, "\n")
    return rps_data_np


def rps_signal_zero_checker(rps_data: np.ndarray):
    print("_" * 60, "zero signal checker", "_" * 60)
    rms_values = []
    for i in range(4):
        rms = np.sqrt(np.mean(rps_data_np[:, i + 1] ** 2))
        rms_values.append(rms)
    rps_signal_sensor_status = []
    for i in range(len(rms_values)):
        if rms_values[i] <= 0.1:
            rps_signal_sensor_status.append("Zero Signal")
        else:
            rps_signal_sensor_status.append(0)
    print("=" * 120, "\n")
    return rps_signal_sensor_status


# %% Toplevel Runner
rps_data_np = rps_prefilter(df_filepath_V, df_test_V, eol_test_id_V)
rps_zero_status = rps_signal_zero_checker(rps_data_np)
