from sql_fetch import fetch_eol


def logic(failure_code, test_id, test_type_id):
    eol_test_id = fetch_eol(test_id, test_type_id)
    print(f"Received EOL Test ID: {eol_test_id}")
    return 0


logic(20070, 20140, "High_Speed")
