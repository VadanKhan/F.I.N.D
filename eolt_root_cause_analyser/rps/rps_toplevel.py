good_rps_test = [23918, "High_Speed", 20140]
static_rps_test = [33931, "Cogging", 32537]
order_uvw_test = [36288, "High_Speed", 31105]

eol_test_id_V = good_rps_test[0]
test_type_V = good_rps_test[1]
test_id_V = good_rps_test[2]


# %% Toplevel Runner
# if __name__ == "__main__":
#     zero_checker = rps_zero_checker_class(eol_test_id_V, test_type_V, test_id_V)
#     results_zero_checker = zero_checker.analyse()
#     zero_checker.report(results_zero_checker)

#     short_checker = rps_short_checker_class(eol_test_id_V, test_type_V, test_id_V)
#     results_short_checker = short_checker.analyse()
#     short_checker.report(results_short_checker)

#     static_checker = rps_static_checker_class(eol_test_id_V, test_type_V, test_id_V)
#     results_static_checker = static_checker.analyse()
#     static_checker.report(results_static_checker)

#     order_checker = rps_order_checker_class(eol_test_id_V, test_type_V, test_id_V)
#     results_order_checker = order_checker.analyse(2)
#     order_checker.report(results_order_checker)

#     print(f"Zero Signal Checker: Overall Results: {results_zero_checker[0]}")
#     print(f"Shorted Signal Checker: Overall Results: {results_short_checker[0]}")
#     print(f"Static Checker: Overall Results: {results_static_checker[0]}")
#     # print(f"Static Checker: Average Status: {rps_static_status[1]}")
#     # print(f"Static Checker: Differential Status: {rps_static_status[2]}")
#     # print(f"Static Checker: Non Normal Times: {rps_static_status[3]}")
#     # print(f"Static Checker: Differential RMS values: {rps_static_status[4]}")
#     print(f"Order Checker: Overall Results: {results_order_checker[0]}")
#     print(f"Order Checker: Correct order of signals: {results_order_checker[1]}")
