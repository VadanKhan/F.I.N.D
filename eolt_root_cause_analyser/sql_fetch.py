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
    print(f"\nReceived EOL Test ID: {eol_test_id_value}\n")
    # print(eol_test_id)

    connection.close()
    return eol_test_id_value
