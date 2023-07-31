import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eolt_root_cause_analyser.class_setup import SpecificTest
from eolt_root_cause_analyser.fetching.data_trimming import edge_filtering
from eolt_root_cause_analyser.fetching.sql_fetch import fetch_motor_details
from eolt_root_cause_analyser.fetching.sql_fetch import fetch_step_timings
from eolt_root_cause_analyser.fetching.tdms_fetch import form_filepath
from eolt_root_cause_analyser.fetching.tdms_fetch import get_time_series_data
from eolt_root_cause_analyser.rps.rps_order_checker import rps_order_checker
from eolt_root_cause_analyser.rps.rps_short_checker import rps_short_checker
from eolt_root_cause_analyser.rps.rps_static_checker import rps_static_checker
from eolt_root_cause_analyser.rps.rps_zero_checker import rps_zero_checker


def rps_prefilter(self):
    """Filters RPS data based on the start and end times of a test.

    This function takes in the file path to a DataFrame containing RPS data, an EOL test ID, and a test type. The
        function filters the RPS data to only include values between the start and end times of the test, which are
        calculated based on the step durations and acceleration durations in a DataFrame fetched using the input EOL
        test ID and test type. The function returns a NumPy array containing the filtered RPS data.

    Args:
        self: from the SpecificTest class.

    Returns:
        np.ndarray: A NumPy array containing the filtered RPS data.
    """
    rps_channel_list = ["SinP", "SinN", "CosP", "CosN"]
    df_filepath = form_filepath(self.eol_test_id, self.test_type, self.test_id, 2)
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

    motor_type = fetch_motor_details(self.eol_test_id)
    step_details_df: pd.DataFrame = fetch_step_timings(motor_type, self.test_type)

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

    return rps_data_np


setattr(SpecificTest, "rps_prefilter", rps_prefilter)


class RpsZero(SpecificTest):
    def analyse(self):
        rps_data = self.rps_prefilter()
        return rps_zero_checker(rps_data)

    def report(self, results):
        # reporting
        return None


class RpsShort(SpecificTest):
    def analyse(self):
        rps_data = self.rps_prefilter()
        return rps_short_checker(rps_data)

    def report(self, results):
        # reporting
        return None


class RpsStatic(SpecificTest):
    def analyse(self):
        rps_data = self.rps_prefilter()
        return rps_static_checker(rps_data, self.eol_test_id, self.test_type)

    def report(self, results):
        # reporting
        return None


class RpsOrder(SpecificTest):
    def analyse(self, step_chosen):
        rps_data = self.rps_prefilter()
        return rps_order_checker(rps_data, step_chosen, self.eol_test_id, self.test_type)

    def report(self, results):
        # reporting
        return None
