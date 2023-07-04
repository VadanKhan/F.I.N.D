from eolt_root_cause_analyser.sql_fetch import fetch_eol
from eolt_root_cause_analyser.tdms_fetch import form_filename_tdms
from eolt_root_cause_analyser.tdms_fetch import read_tdms

# from eolt_root_cause_analyser.initial_plots import initial_plots

ASSUMED_SAMPLING_PERIOD = 0.00391129140625


def logic(failure_code, test_id, test_type_id):
    eol_test_id = fetch_eol(test_id, test_type_id)
    tdmsname = form_filename_tdms(test_id, test_type_id, eol_test_id)
    df = read_tdms(tdmsname)
    # print(df)
    # initial_plots(df)
    return 0


logic(20070, 20140, "High_Speed")
