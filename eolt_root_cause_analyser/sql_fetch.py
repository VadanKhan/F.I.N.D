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
        r"Server=CLASSIFIED;"
        "Database=CLASSIFIED;"
        "UID=CLASSIFIED;"
        "PWD=CLASSIFIED;"
        "APP=CLASSIFIED"
        "Trusted_Connection=CLASSIFIED;"
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
    print(f"\nReceived EOL Test ID: {eol_test_id_value}\n")
    # print(eol_test_id)

    connection.close()
    return eol_test_id_value


def fetch_motor_details(eol_test_id):
    """Fetches the EOL_Test_ID from the database table Test_{test_type_id} where Test_ID is equal to the given test_id.

    Args:
        test_id (int): The ID of the test to fetch the EOL_Test_ID for.
        test_type_id (int): The ID of the test type to fetch the EOL_Test_ID from.

    Returns:
        int or Error: The fetched EOL_Test_ID or an error object if an error occurred.
    """
    connection = eolt_connect()
    motor_type_db = pd.read_sql_query(f"SELECT Motor_Type FROM EOLTest WHERE EOL_Test_ID={eol_test_id}", connection)
    motor_type = motor_type_db.iloc[0, 0]
    print(f"\nReceived Motor Type: {motor_type}\n")
    # print(eol_test_id)

    connection.close()
    return motor_type


def fetch_step_timings(motor_type):
    """Fetches the EOL_Test_ID from the database table Test_{test_type_id} where Test_ID is equal to the given test_id.

    Args:
        test_id (int): The ID of the test to fetch the EOL_Test_ID for.
        test_type_id (int): The ID of the test type to fetch the EOL_Test_ID from.

    Returns:
        int or Error: The fetched EOL_Test_ID or an error object if an error occurred.
    """
    connection = eolt_connect()
    motor_type_df = pd.read_sql_query(
        f"""Select Step_Number, Duration_ms, Accel_Time_S
        FROM StepDescription_High_Speed
        INNER JOIN DriveCycleStep_High_Speed ON Step_ID = Step_Description_ID
        INNER JOIN DriveCycle_High_Speed ON Cycle_ID = Drive_Cycle_ID
        INNER JOIN MotorTypes ON Drive_Cycle_ID = High_Speed_Cycle_ID
        WHERE Motor_Type_ID='{motor_type}' """,
        connection,
    )
    # motor_type = motor_type_db.iloc[0, 0]
    print(f"\nReceived Dataframe:\n {motor_type_df}")
    # print(eol_test_id)

    connection.close()
    return motor_type_df
