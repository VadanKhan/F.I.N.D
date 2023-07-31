import itertools

import numpy as np
from eolt_root_cause_analyser.class_setup import FailureModeChecker
from eolt_root_cause_analyser.data_processing.data_processing import calculate_frequencies
from eolt_root_cause_analyser.data_processing.data_processing import correct_order_checker
from eolt_root_cause_analyser.fetching.sql_fetch import select_step

# import matplotlib.pyplot as plt


ORDERCHECKER_POINTS = 200
ORDERCHECKER_SIGNAL_MSE_LOW = 1.0


class RpsOrder(FailureModeChecker):
    def _rps_order_checker(self, rps_data: np.ndarray, step_chosen, eol_test_id, test_type):
        """Checks the order of the given RPS signals.

        This function takes a NumPy array containing RPS data, a step_chosen, an eol_test_id, and a test_type as input
            and returns a list of integers indicating the status of each sensor and a list of strings indicating the
            correct order of the sensors. If the sensors are in the correct order, the status of each sensor is 0,
            otherwise it is 1.

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
        # print(lower_bound)
        upper_bound = step_index[0] + int(step_points / 2) + num_points_selected
        # print(upper_bound)
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
        sinP = rps_data[lower_bound:upper_bound, 1]  # sinP from column 1
        cosP = rps_data[lower_bound:upper_bound, 2]  # cosP from column 3
        sinN = rps_data[lower_bound:upper_bound, 3]  # sinN from column 2
        cosN = rps_data[lower_bound:upper_bound, 4]  # cosN from column 4
        base_signals_list = [sinP, cosP, sinN, cosN]
        signal_names = ["1", "2", "3", "4"]
        signals = np.column_stack((sinP, sinN, cosP, cosN))

        # fig12, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)
        # ax1.plot(time, sinP)
        # ax2.plot(time, cosP)
        # ax3.plot(time, sinN)
        # ax4.plot(time, cosN)
        # ax5.plot(time, sinP, label="SinP")
        # ax5.plot(time, cosP, label="CosP")
        # ax5.plot(time, sinN, label="SinN")
        # ax5.plot(time, cosN, label="CosN")
        # plt.legend()

        try:
            frequencies = calculate_frequencies(time, signals)
            T = 1 / np.mean(frequencies)
        except Exception as e:
            print(f"Error calculating frequencies: {e}")
            return ["FFT error", "FFT error"]

        # main_signal, shifted_signal, error = align_signals(cosN, cosP, T, 0.5, sampling_time)
        # print("\nPlotting Error: ", error, "\n")

        signals_dict = dict(zip(signal_names, base_signals_list))
        correct_order = []
        for permutation in itertools.permutations(signals_dict.items()):
            if correct_order_checker(
                [signal for name, signal in permutation], ORDERCHECKER_SIGNAL_MSE_LOW, T, sampling_time
            ):
                for name, signal in permutation:
                    correct_order.append(name)
                break

        print("=" * 120, "\n")

        rps_pinning_status = ["1" if a != b else "0" for a, b in zip(correct_order, ["1", "2", "3", "4"])]
        return rps_pinning_status, correct_order

    def analyse(self, step_chosen):
        rps_data = self.rps_prefilter()
        return self._rps_order_checker(rps_data, step_chosen, self.eol_test_id, self.test_type)

    def report(self, results):
        # reporting
        return None
