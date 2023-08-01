import numpy as np
import pandas as pd
from eolt_root_cause_analyser.fetching.data_trimming import edge_filtering
from eolt_root_cause_analyser.fetching.data_trimming import select_step
from eolt_root_cause_analyser.fetching.sql_fetch import fetch_step_timings
from eolt_root_cause_analyser.fetching.tdms_fetch import form_filepath
from eolt_root_cause_analyser.fetching.tdms_fetch import get_time_series_data


class FailureModeChecker:
    def __init__(self, eol_test_id, test_type, test_id, failure_code):
        self.eol_test_id = eol_test_id
        self.test_type = test_type
        self.test_id = test_id
        self.failure_code = failure_code

    def rps_prefilter(self):
        """Filters RPS data based on the start and end times of a test.

        This function takes in the file path to a DataFrame containing RPS data, an EOL test ID, and a test type. The
            function filters the RPS data to only include values between the start and end times of the test, which are
            calculated based on the step durations and acceleration durations in a DataFrame fetched using the input EOL
            test ID and test type. The function returns a NumPy array containing the filtered RPS data.

        Returns:
            np.ndarray: A NumPy array containing the filtered RPS data.
        """
        rps_channel_list = ["SinP", "CosP", "SinN", "CosN"]
        df_filepath = form_filepath(self.eol_test_id, self.test_type, self.test_id, 2)
        rps_time_list = get_time_series_data(df_filepath, rps_channel_list)

        # print(f"{rps_time_list[0]}")
        SinP_df = rps_time_list[0]
        CosP_df = rps_time_list[1]
        SinN_df = rps_time_list[2]
        CosN_df = rps_time_list[3]
        time = SinP_df.SinP_time
        time.name = "RPS_time"
        SinP_onlyvals_pd = SinP_df.SinP
        CosP_onlyvals_pd = CosP_df.CosP
        SinN_onlyvals_pd = SinN_df.SinN
        CosN_onlyvals_pd = CosN_df.CosN
        RPS_df = pd.concat([time, SinP_onlyvals_pd, CosP_onlyvals_pd, SinN_onlyvals_pd, CosN_onlyvals_pd], axis=1)
        rps_data_raw_np: np.ndarray = RPS_df.values

        time_np = rps_data_raw_np[:, 0]

        step_details_df: pd.DataFrame = fetch_step_timings(self.eol_test_id, self.test_type)

        failurecode_step = int(str(self.failure_code)[1])
        step_numbers = step_details_df["Step_Number"].values
        num_steps = len(step_numbers)

        if failurecode_step > 1 and failurecode_step < num_steps:
            # if error not in the first or last step, we output an array that filters out first and last step data.
            filtered_index_array = edge_filtering(step_details_df, time_np)
            rps_data_np = rps_data_raw_np[filtered_index_array]

            # fig8, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)
            # ax1.plot(rps_data_np[:, 0], rps_data_np[:, 1])
            # ax2.plot(rps_data_np[:, 0], rps_data_np[:, 2])
            # ax3.plot(rps_data_np[:, 0], rps_data_np[:, 3])
            # ax4.plot(rps_data_np[:, 0], rps_data_np[:, 4])
            # ax5.plot(rps_data_np[:, 0], rps_data_np[:, 1], label="SinP")
            # ax5.plot(rps_data_np[:, 0], rps_data_np[:, 2], label="SinN")
            # ax5.plot(rps_data_np[:, 0], rps_data_np[:, 3], label="CosP")
            # ax5.plot(rps_data_np[:, 0], rps_data_np[:, 4], label="CosN")
            # ax1.set_ylim(0, 5)
            # ax2.set_ylim(0, 5)
            # ax3.set_ylim(0, 5)
            # ax4.set_ylim(0, 5)
            # ax5.set_ylim(0, 5)
            # ax5.legend()
            # fig8.suptitle("Input (Selected) Signals")

            return rps_data_np

        elif failurecode_step == 1 or failurecode_step == num_steps:
            # in the case of error in the first or last step, we just output the data from that step.
            filtered_index_array = select_step(step_details_df, time, failurecode_step)
            rps_data_np = rps_data_raw_np[filtered_index_array]
            return rps_data_np

        else:
            print("Error: Failurecode Step not matching with Step Data")
            return 1
