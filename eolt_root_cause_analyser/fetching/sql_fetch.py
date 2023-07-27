import numpy as np
import pandas as pd
import pyodbc


def eolt_connect():
    """
    Creates the initial connection to the EOLT database to be used by the pandas SQL queries

    Returns:
        connection (pyodbc.Connection): Initialises connection to EOLT database

    """

    connection = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        r"Server=10.30.0.2\EOLTESTER;"
        "Database=EOL Management;"
        "UID=EOLQuery;"
        "PWD=Yasa1234;"
        "APP=Python"
        "Trusted_Connection=yes;"
    )
    return connection


def fetch_eol(test_id, test_type_id):
    """Fetches the EOL_Test_ID from the database table Test_{test_type_id} where Test_ID is equal to the given test_id.

    Args:
        test_id (int): The ID of the test to fetch the EOL_Test_ID for.
        test_type_id (int): The ID of the test type to fetch the EOL_Test_ID from.

    Returns:
        int or Error: The fetched EOL_Test_ID or an error object if an error occurred.
    """
    connection = eolt_connect()
    eol_test_id = pd.read_sql_query(f"SELECT EOL_Test_ID from Test_{test_type_id} WHERE Test_ID={test_id}", connection)
    eol_test_id_value = eol_test_id.iloc[0, 0]
    # print(eol_test_id)

    connection.close()
    return eol_test_id_value


def fetch_motor_details(eol_test_id: int):
    """Fetches the motor type from the EOLTest database table for a given EOL_Test_ID.

    Args:
        eol_test_id (int): The ID of the EOL test to fetch the motor type for.

    Returns:
        int or Error: The fetched motor type or an error object if an error occurred.
    """
    connection = eolt_connect()
    motor_type_db = pd.read_sql_query(f"SELECT Motor_Type FROM EOLTest WHERE EOL_Test_ID={eol_test_id}", connection)
    motor_type = motor_type_db.iloc[0, 0]
    # print(eol_test_id)

    connection.close()
    return motor_type


def fetch_step_timings(motor_type, test_type):
    """Fetches step timings from the database for the given motor type and test type.

    This function takes a motor type and a test type as input and returns a DataFrame containing the step number,
        duration, and acceleration time for each step in the test.

    Args:
        motor_type (str): The type of the motor to fetch step timings for.
        test_type (str): The type of the test to fetch step timings for.

    Returns:
        pd.DataFrame: A DataFrame containing the step number, duration, and acceleration time for each step in the test.
    """
    connection = eolt_connect()
    motor_type_df = pd.read_sql_query(
        f"""Select Step_Number, Duration_ms, Accel_Time_S
        FROM StepDescription_{test_type}
        INNER JOIN DriveCycleStep_{test_type} ON Step_ID = Step_Description_ID
        INNER JOIN DriveCycle_{test_type} ON Cycle_ID = Drive_Cycle_ID
        INNER JOIN MotorTypes ON Drive_Cycle_ID = {test_type}_Cycle_ID
        WHERE Motor_Type_ID='{motor_type}' """,
        connection,
    )

    connection.close()
    return motor_type_df


def select_step(time: np.ndarray, eol_test_id, test_type, step_input):
    """Selects a specific step from the input time array based on the given step_input.

    This function takes a 1D numpy array of time values, an eol_test_id, a test_type, and a step_input as input. It
    returns an array containing the indices of the time values that are within the specified step.

    Args:
        time (np.ndarray): The time values of the input data.
        eol_test_id (int): The eol_test_id of the motor.
        test_type (str): The type of test being performed.
        step_input (int): The step number to be selected.

    Returns:
        np.ndarray: An array containing the indices of the time values that are within the specified step.
    """
    motor_type = fetch_motor_details(eol_test_id)
    step_dataframe: pd.DataFrame = fetch_step_timings(motor_type, test_type)
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
