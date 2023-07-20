from eolt_root_cause_analyser.sql_fetch import fetch_eol
from eolt_root_cause_analyser.tdms_fetch import form_filepath
from eolt_root_cause_analyser.tdms_fetch import read_tdms

# from eolt_root_cause_analyser.initial_plots import initial_plots


def logic(failure_code, test_id, test_type_id):
    eol_test_id = fetch_eol(test_id, test_type_id)
    tdms_filepath = form_filepath(eol_test_id, test_id, test_type_id)
    df = read_tdms(tdms_filepath)
    # print(df)
    # initial_plots(df)
    return df


# logic(20070, 20140, "High_Speed")
