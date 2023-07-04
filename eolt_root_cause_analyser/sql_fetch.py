import pandas as pd
import pyodbc
from mysql.connector import Error


def eolt_connect() -> pyodbc.Connection:
    """
    Creates the initial connection to the EOLT database to be used by the pandas SQL queries

    Returns:
        connection (pyodbc.Connection): Initialises connection to EOLT database

    """
    try:
        connection = pyodbc.connect(
            "DRIVER=SQL Server;"
            r"SERVER=0150-D\EOLTESTER;"
            "UID=EOLQuery;"
            "PWD=Yasa1234;"
            "APP=Python;"
            "Trusted_Connection=yes;"
            "Database=EOL Management"
        )
    except Error as err:
        print(f"Error: '{err}'")
    return connection


def fetch_eol(test_id, test_type_id):
    """Fetches the EOL_Test_ID from the database table Test_{test_type_id} where Test_ID is equal to the given test_id.

    Args:
        test_id (int): The ID of the test to fetch the EOL_Test_ID for.
        test_type_id (int): The ID of the test type to fetch the EOL_Test_ID from.

    Returns:
        int or Error: The fetched EOL_Test_ID or an error object if an error occurred.
    """
    try:
        connection = eolt_connect()
        eol_test_id = pd.read_sql_query(
            f"SELECT EOL_Test_ID from Test_{test_type_id} WHERE Test_ID={test_id}", connection
        )
        eol_test_id_value = eol_test_id.iloc[0, 0]
        print(f"\nReceived EOL Test ID: {eol_test_id_value}\n")
        # print(eol_test_id)

        connection.close()
    except Error as err:
        print(f"Error: '{err}'")
        eol_test_id_value = err
    return eol_test_id_value


# dummy_test_id = 393
# dummy_test_type = "High_Speed"
# dummy_eol_test_id = fetch(dummy_test_id, dummy_test_type)
# print(dummy_eol_test_id)
