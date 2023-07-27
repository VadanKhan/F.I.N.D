import numpy as np

SHORTCHECKER_LOW = 4.5
SHORTCHECKER_HIGH = 5.5


def rps_short_checker(rps_data: np.ndarray):
    """Checks for 5V signals in the given RPS data.

    This function takes a NumPy array containing RPS data as input and returns a list of strings indicating the status
        of each sensor. If the mean value of a sensor is between 4.5 and 5.5, its status is "5V Signal", otherwise it is
        0.

    Args:
        rps_data (np.ndarray): A NumPy array containing RPS data.

    Returns:
        tuple: A tuple containing two elements: a list of strings indicating the status of each sensor, and a list of
        mean values for each sensor.
    """
    print("_" * 60, "5V signal checker", "_" * 60)
    mean_values = []
    for column_index in range(len(rps_data[0, 1:])):
        mean = np.mean(rps_data[:, column_index + 1])
        mean_values.append(mean)
    rps_signal_sensor_status = []
    for mean_value in mean_values:
        if mean_value <= SHORTCHECKER_LOW and mean_value >= SHORTCHECKER_HIGH:
            rps_signal_sensor_status.append("1")
        else:
            rps_signal_sensor_status.append("0")
    print("=" * 120, "\n")
    return rps_signal_sensor_status, mean_values
