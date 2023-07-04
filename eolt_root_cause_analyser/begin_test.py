def begin_test(failure_code, test_id, test_type_id):
    """Simple program that prints out the input information as you call this cmd interface"""
    print(f"received inputs: failure_code={failure_code}, test_id={test_id}, test_type_id={test_type_id}")
    return failure_code, test_id, test_type_id
