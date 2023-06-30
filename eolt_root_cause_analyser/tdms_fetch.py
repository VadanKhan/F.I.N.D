import warnings
from pathlib import Path

import pandas as pd
import yasa_file_io.tdms


def form_filename_tdms(test_id, test_type_id, eol_test_id):
    """Forms a filename for the EOL TDMS files using the given test_id, test_type_id, and eol_test_id.

    Args:
        test_id (int): The ID of the test.
        test_type_id (int): The ID of the test type.
        eol_test_id (int): The EOL_Test_ID.

    Returns:
        str: The formed filename for the TDMS file.
    """
    filename = f"{eol_test_id}_{test_type_id}_{test_id}.tdms"
    return filename


def read_tdms(filename):
    """Reads a EOL TDMS file and returns its contents as a Pandas DataFrame.
    NOTE: THIS STEP HAS A LARGE RUNTIME ~20 SECONDS if connecting to QNAP

    Args:
        filename (str): The name of the TDMS file to read.

    Returns:
        DataFrame: A Pandas DataFrame containing the data from the TDMS file.
    """
    try:
        df = yasa_file_io.tdms.read_tdms_as_dataframe(
            Path(rf"C:\Users\Vadan.Khan\Documents\Project\Sample TDMS files\{filename}"),
            channel_map={"Ambient_new": "Ambient"},
            extract_all=True,
            fuzzy_matching=True,
            drop_duplicates=False,
            fuzzy_match_score=50.0,
            search_terms_as_keys=False,
        )
        return df
    except Exception:
        warnings.warn("Accessing TDMS Failed", UserWarning)
        return None
