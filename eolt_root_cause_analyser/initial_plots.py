import matplotlib.pyplot as plt
import pandas as pd
from tdms_fetch import read_tdms

df_test = read_tdms("23918_High_Speed_20140.tdms")


def initial_plots(df):
    df = df.reset_index()
    # index_subset = df[(df['index'] >= 650) & (df['index'] <= 650.1 )]
    index_subset2 = df[(df["index"] >= 200000) & (df["index"] <= 200100)]
    index_subset3 = df[(df["index"] >= 200000) & (df["index"] <= 200100)]
    index_subset4 = df[(df["index"] >= 200000) & (df["index"] <= 201000)]
    index_subset5 = df[(df["index"] >= 200000) & (df["index"] <= 200100)]
    # df.plot("index", y=["U", "V", "W"])
    index_subset2.plot("index", y=["U", "V", "W"])
    df.plot("index", y=["Speed_original"])
    df.plot("index", y=["Torque"])
    # index_subset4.plot("index", y=["Microphone"])
    plt.show()


df = df_test
df = df.reset_index()
columns = list(df.columns)
print(columns)
# index_subset = df[(df['index'] >= 650) & (df['index'] <= 650.1 )]
index_subset2 = df[(df["index"] >= 200000) & (df["index"] <= 200100)]
index_subset3 = df[(df["index"] >= 200000) & (df["index"] <= 201000)]
index_subset4 = df[(df["index"] >= 266000) & (df["index"] <= 285000)]
index_subset5 = df[(df["index"] >= 200000) & (df["index"] <= 200100)]
# df.plot("index", y=["U", "V", "W"])

# index_subset2.plot("index", y=["U", "V", "W"])
# df.plot("index", y=["Speed_original"])
# df.plot("index", y=["Torque"])
# index_subset4.plot("index", y=["Microphone"])
# df.plot("index", y=["Rotor_1_original"])
index_subset4.plot("index", y=["SinP", "SinN", "CosP", "CosN"])
index_subset4.plot("index", y=["U", "V", "W"])
plt.show()
