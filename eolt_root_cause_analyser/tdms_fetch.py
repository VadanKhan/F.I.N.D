from pathlib import Path

import pandas as pd
import yasa_file_io.tdms


def form_filename_tdms(test_id, test_type_id, eol_test_id):
    filename = f"{eol_test_id}_{test_type_id}_{test_id}.tdms"
    return filename


def read_tdms(filename):
    df = yasa_file_io.tdms.read_tdms_as_dataframe(
        Path(rf"\\QNAP-463\eol\{filename}"),
        channel_map={"Ambient_new": "Ambient"},
        extract_all=True,
        fuzzy_matching=True,
        drop_duplicates=False,
        fuzzy_match_score=50.0,
        search_terms_as_keys=False,
    )
    return df
