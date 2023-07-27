import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eolt_root_cause_analyser.fetching.data_trimming import edge_filtering
from eolt_root_cause_analyser.fetching.sql_fetch import fetch_eol
from eolt_root_cause_analyser.fetching.sql_fetch import fetch_motor_details
from eolt_root_cause_analyser.fetching.sql_fetch import fetch_step_timings
from eolt_root_cause_analyser.fetching.tdms_fetch import form_filepath
from eolt_root_cause_analyser.fetching.tdms_fetch import get_time_series_data
from eolt_root_cause_analyser.rps.rps_order_checker import rps_order_checker
from eolt_root_cause_analyser.rps.rps_short_checker import rps_short_checker
from eolt_root_cause_analyser.rps.rps_static_checker import rps_static_checker
from eolt_root_cause_analyser.rps.rps_zero_checker import rps_zero_checker

# from eolt_root_cause_analyser.initial_plots import initial_plots

good_rps_test = [23918, "High_Speed", 20140]
static_rps_test = [33931, "Cogging", 32537]
order_uvw_test = [36288, "High_Speed", 31105]

eol_test_id_V = good_rps_test[0]
test_type_V = good_rps_test[1]
test_id_V = good_rps_test[2]


class specific_test:
    def __init__(self, eol_test_id, test_type, test_id):
        self.eol_test_id = eol_test_id
        self.test_type = test_type
        self.test_id = test_id

    def rps_prefilter(self):
        """Filters RPS data based on the start and end times of a test.

        This function takes in the file path to a DataFrame containing RPS data, an EOL test ID, and a test type. The
            function filters the RPS data to only include values between the start and end times of the test, which are
            calculated based on the step durations and acceleration durations in a DataFrame fetched using the input EOL
            test ID and test type. The function returns a NumPy array containing the filtered RPS data.

        Returns:
            np.ndarray: A NumPy array containing the filtered RPS data.
        """
        print("_" * 60, "RPS prefilter", "_" * 60)
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

        print("=" * 120, "\n")

        return rps_data_np


class rps_zero_checker_class(specific_test):
    def analyse(self):
        rps_data = self.rps_prefilter()
        return rps_zero_checker(rps_data)

    def report(self, results):
        # reporting
        return None


class rps_short_checker_class(specific_test):
    def analyse(self):
        rps_data = self.rps_prefilter()
        return rps_short_checker(rps_data)

    def report(self, results):
        # reporting
        return None


class rps_static_checker_class(specific_test):
    def analyse(self):
        rps_data = self.rps_prefilter()
        return rps_static_checker(rps_data, self.eol_test_id, self.test_type)

    def report(self, results):
        # reporting
        return None


class rps_order_checker_class(specific_test):
    def analyse(self, step_chosen):
        rps_data = self.rps_prefilter()
        return rps_order_checker(rps_data, step_chosen, self.eol_test_id, self.test_type)

    def report(self, results):
        # reporting
        return None


def logic(failure_code, test_type, test_id):
    eol_test_id = fetch_eol(test_id, test_type)

    zero_checker = rps_zero_checker_class(eol_test_id, test_type, test_id)
    results_zero_checker = zero_checker.analyse()
    zero_checker.report(results_zero_checker)

    short_checker = rps_short_checker_class(eol_test_id, test_type, test_id)
    results_short_checker = short_checker.analyse()
    short_checker.report(results_short_checker)

    static_checker = rps_static_checker_class(eol_test_id, test_type, test_id)
    results_static_checker = static_checker.analyse()
    static_checker.report(results_static_checker)

    order_checker = rps_order_checker_class(eol_test_id, test_type, test_id)
    results_order_checker = order_checker.analyse(2)
    order_checker.report(results_order_checker)

    print(f"Zero Signal Checker: Overall Results: {results_zero_checker[0]}")
    print(f"Shorted Signal Checker: Overall Results: {results_short_checker[0]}")
    print(f"Static Checker: Overall Results: {results_static_checker[0]}")
    # print(f"Static Checker: Average Status: {rps_static_status[1]}")
    # print(f"Static Checker: Differential Status: {rps_static_status[2]}")
    # print(f"Static Checker: Non Normal Times: {rps_static_status[3]}")
    # print(f"Static Checker: Differential RMS values: {rps_static_status[4]}")
    print(f"Order Checker: Overall Results: {results_order_checker[0]}")
    print(f"Order Checker: Correct order of signals: {results_order_checker[1]}")


logic("whatever", test_type_V, test_id_V)
