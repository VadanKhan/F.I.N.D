from pathlib import Path

import pandas as pd
import yasa_file_io.tdms


def form_filename(test_id, test_type_id, eol_test_id):
    filename = f"{eol_test_id}_{test_type_id}_{test_id}"
    return filename


ans = form_filename(20140, "High_Speed", 23918)
print(ans)
