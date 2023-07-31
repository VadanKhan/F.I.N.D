from eolt_root_cause_analyser.fetching.sql_fetch import fetch_eol
from eolt_root_cause_analyser.results import results
from eolt_root_cause_analyser.rps.rps_order_checker import RpsOrder
from eolt_root_cause_analyser.rps.rps_short_checker import RpsShort
from eolt_root_cause_analyser.rps.rps_static_checker import RpsStatic
from eolt_root_cause_analyser.rps.rps_zero_checker import RpsZero

# from eolt_root_cause_analyser.initial_plots import initial_plots

good_rps_test = [23918, "High_Speed", 20140]
static_rps_test = [33931, "Cogging", 32537]
order_uvw_test = [36288, "High_Speed", 31105]

eol_test_id_V = good_rps_test[0]
test_type_V = good_rps_test[1]
test_id_V = good_rps_test[2]

SAVING_PATH = 0  # set to 0 if not wanting to call remote saving


def logic(failure_code, test_type, test_id):
    eol_test_id = fetch_eol(test_id, test_type)

    zero_checker = RpsZero(eol_test_id, test_type, test_id)
    results_zero_checker = zero_checker.analyse()
    zero_checker.report(results_zero_checker)

    short_checker = RpsShort(eol_test_id, test_type, test_id)
    results_short_checker = short_checker.analyse()
    short_checker.report(results_short_checker)

    static_checker = RpsStatic(eol_test_id, test_type, test_id)
    results_static_checker = static_checker.analyse()
    static_checker.report(results_static_checker)

    order_checker = RpsOrder(eol_test_id, test_type, test_id)
    results_order_checker = order_checker.analyse(2)
    order_checker.report(results_order_checker)

    print("_" * 60, "Analysis Results", "_" * 60)
    print(f"Zero Signal Checker: Overall Results: {results_zero_checker[0]}")
    print(f"Shorted Signal Checker: Overall Results: {results_short_checker[0]}")
    print(f"Static Checker: Overall Results: {results_static_checker[0]}")
    # print(f"Static Checker: Average Status: {rps_static_status[1]}")
    # print(f"Static Checker: Differential Status: {rps_static_status[2]}")
    # print(f"Static Checker: Non Normal Times: {rps_static_status[3]}")
    # print(f"Static Checker: Differential RMS values: {rps_static_status[4]}")
    print(f"Order Checker: Overall Results: {results_order_checker[0]}")
    print(f"Order Checker: Correct order of signals: {results_order_checker[1]}")
    status_dict = {
        "Zero Signal Checker": results_zero_checker[0],
        "Shorted Signal Checker": results_short_checker[0],
        "Static Checker": results_static_checker[0],
        "Order Checker": results_order_checker[0],
    }
    results(status_dict, eol_test_id, test_type, test_id, SAVING_PATH)

    print("\n")


# to run code:
logic("whatever", test_type_V, test_id_V)
