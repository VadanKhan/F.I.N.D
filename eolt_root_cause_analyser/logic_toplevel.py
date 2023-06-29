from eolt_root_cause_analyser.sql_fetch import fetch_eol
from eolt_root_cause_analyser.tdms_fetch import form_filename_tdms
from eolt_root_cause_analyser.tdms_fetch import read_tdms


def logic(failure_code, test_id, test_type_id):
    eol_test_id = fetch_eol(test_id, test_type_id)
    print(f"Received EOL Test ID: {eol_test_id}")
    tdmsname = form_filename_tdms(test_id, test_type_id, eol_test_id)
    data = read_tdms(tdmsname)
    print(data)
    return 0


logic(20070, 20140, "High_Speed")
