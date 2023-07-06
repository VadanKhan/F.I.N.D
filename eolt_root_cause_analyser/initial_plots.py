import matplotlib.pyplot as plt
import pandas as pd
from tdms_fetch import form_filepath
from tdms_fetch import get_time_series_data
from tdms_fetch import read_tdms

df_filepath = form_filepath("23918_High_Speed_20140.tdms")
df_test = read_tdms(df_filepath)
RPS_time_list = get_time_series_data(df_filepath, ["AI DAQ - High Speed Inc RTD"], ["SinP", "SinN", "CosP", "CosN"])

# time_dfs = get_time_series_data()


# %%
def initial_plots(df):
    # df = df.reset_index()
    # # index_subset = df[(df['index'] >= 650) & (df['index'] <= 650.1 )]
    # index_subset2 = df[(df["index"] >= 200000) & (df["index"] <= 200100)]
    # index_subset3 = df[(df["index"] >= 200000) & (df["index"] <= 200100)]
    # index_subset4 = df[(df["index"] >= 200000) & (df["index"] <= 201000)]
    # index_subset5 = df[(df["index"] >= 200000) & (df["index"] <= 200100)]
    # # df.plot("index", y=["U", "V", "W"])
    # index_subset2.plot("index", y=["U", "V", "W"])
    # df.plot("index", y=["Speed_original"])
    # df.plot("index", y=["Torque"])
    # # index_subset4.plot("index", y=["Microphone"])
    plt.show()


# %%
df = df_test
# df = df.reset_index()
print(f"{df}\n")
columns = list(df.columns)
print(columns)

# SinP = RPS_time_list[0]
# SinN = RPS_time_list[1]
# CosP = RPS_time_list[2]
# CosN = RPS_time_list[3]
# print(SinP)
# Sin
# SinP.plot("SinP_Time", y=["SinP"])

# sinN = df["AI DAQ - High Speed Inc RTD/SinN"]
# print(sinN)
# index_subset = df[(df['index'] >= 650) & (df['index'] <= 650.1 )]
# index_subset2 = df[(df["index"] >= 200000) & (df["index"] <= 200100)]
# index_subset3 = df[(df["index"] >= 200000) & (df["index"] <= 201000)]
# index_subset4 = df[(df["index"] >= 266000) & (df["index"] <= 285000)]
# index_subset5 = df[(df["index"] >= 200000) & (df["index"] <= 200100)]
# df.plot("index", y=["U", "V", "W"])

# index_subset2.plot("index", y=["U", "V", "W"])
# df.plot("index", y=["Speed_original"])
# df.plot("index", y=["Torque"])
# index_subset4.plot("index", y=["Microphone"])
# df.plot("index", y=["Rotor_1_original"])
# index_subset4.plot("index", y=["SinP", "SinN", "CosP", "CosN"])
# index_subset4.plot("index", y=["U", "V", "W"])
plt.show()
