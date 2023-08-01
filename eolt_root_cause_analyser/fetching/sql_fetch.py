import warnings

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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eol_test_id = pd.read_sql_query(
            f"SELECT EOL_Test_ID from Test_{test_type_id} WHERE Test_ID={test_id}", connection
        )
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        motor_type_db = pd.read_sql_query(f"SELECT Motor_Type FROM EOLTest WHERE EOL_Test_ID={eol_test_id}", connection)
    motor_type = motor_type_db.iloc[0, 0]
    # print(eol_test_id)

    connection.close()
    return motor_type


def fetch_step_timings(eol_test_id, test_type):
    """Fetches step timings from the database for the given motor type and test type.

    This function takes a motor type and a test type as input and returns a DataFrame containing the step number,
        duration, and acceleration time for each step in the test.

    Args:
        eol_test_id (int): The ID of the EOL test to fetch the motor type for.
        test_type (str): The type of the test to fetch step timings for.

    Returns:
        pd.DataFrame: A DataFrame containing the step number, duration, and acceleration time for each step in the test.
    """
    connection = eolt_connect()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        motor_type_db = pd.read_sql_query(f"SELECT Motor_Type FROM EOLTest WHERE EOL_Test_ID={eol_test_id}", connection)
        motor_type = motor_type_db.iloc[0, 0]
        step_timings_df = pd.read_sql_query(
            f"""Select Step_Number, Duration_ms, Accel_Time_S
            FROM StepDescription_{test_type}
            INNER JOIN DriveCycleStep_{test_type} ON Step_ID = Step_Description_ID
            INNER JOIN DriveCycle_{test_type} ON Cycle_ID = Drive_Cycle_ID
            INNER JOIN MotorTypes ON Drive_Cycle_ID = {test_type}_Cycle_ID
            WHERE Motor_Type_ID='{motor_type}' """,
            connection,
        )

    connection.close()
    return step_timings_df


def fetch_past_failure_code(eol_test_id, test_type, test_id):
    """Fetches failure codes from the database for the given test type, test id and EOL test id.

    This function takes a test type, test id and EOL test id as input and returns a DataFrame containing the failure
        codes.

    Args:
        test_type (str): The type of the test to fetch failure codes for.
        test_id (int): The id of the test to fetch failure codes for.
        eol_test_id (int): The EOL test id to fetch failure codes for.

    Returns:
        pd.DataFrame: A DataFrame containing the failure codes.
    """
    connection = eolt_connect()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        failure_code_df = pd.read_sql_query(
            f"SELECT Failure_Codes FROM dbo.Test_{test_type} WHERE Test_ID={test_id} AND EOL_Test_ID={eol_test_id}",
            connection,
        )
        failure_code = failure_code_df.iloc[0, 0]

    connection.close()
    return failure_code
