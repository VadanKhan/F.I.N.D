import numpy as np

ZEROCHECKER_LOW_RMS = 0.01


def rps_zero_checker(rps_data: np.ndarray):
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
