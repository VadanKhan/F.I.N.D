import re
import sys
from pathlib import Path
from typing import Optional
from warnings import warn

import pandas as pd
import thefuzz.process as fuzz
from nptdms import TdmsFile


def _fuzzy_match_dictionary(tdms_columns: list, channel_map: dict, fuzzy_match_score: float) -> dict:  # type: ignore
    """Creates a new dictionary that fuzzy matches the key to the column.

    Args:
        tdms_columns (list): all the columns in the dataframe
        channel_map (dict): dictionary containing the channel map
        fuzzy_match_score (float): threshold the fuzzy score should be above

    Returns:
        new_channel_map (dict): new channel map, key=fuzzy matched column, val=name to change col to
    """
    new_channel_map = {}
    for key, val in channel_map.items():
        key_match, fuzzy_score = fuzz.extractOne(key, tdms_columns)
        new_channel_map[key_match] = val
        if fuzzy_score <= fuzzy_match_score:
            warn(
                f"fuzzy matched column scored {fuzzy_score}%, below threshold of {fuzzy_match_score}%. {key} mapped to"
                f"{key_match}",
                stacklevel=2,
            )

    return new_channel_map


def tdms_mapping_warning(tdms_columns, channel_map):
    """Prints out warnings if a channel key does not match a column exactly.

    Args:
        tdms_columns (list): all the columns in the dataframe
        channel_map (dict): dictionary of channel maps

    """
    for key, val in channel_map.items():
        if key in tdms_columns:
            continue
        else:
            warn(f"channel key, {key}, did not match a column in the dataframe. {tdms_columns}")


def _apply_channel_map(
    tdms_file: pd.DataFrame,
    channel_map: dict,
    fuzzy_matching: bool,
    drop_duplicates: bool,
    extract_all: Optional[bool] = False,
    fuzzy_match_score: Optional[float] = 60.0,
) -> pd.DataFrame:
    """Applies the channel map to the dataframe.

    Args:
        tdms_file (pd.DataFrame): tdms file as a dataframe
        channel_map (dict): dictionary containing the channel map
        fuzzy_matching (bool): whether to use fuzzy matching
        drop_duplicates (bool): whether to drop duplicates
        extract_all (bool, optional): whether to extract all channels. Defaults to False.
        fuzzy_match_score (float, optional): fuzzy score threshold, if fuzzy_match=True

    Returns:
        pd.DataFrame: tdms file with the columns renamed according to the channel map
    """
    # remove the groups from the columns names for convenience. Use regex to find the text between the delimiting chars
    tdms_file.columns = [re.search(r"'\/'(.*?)'", column).group(1) for column in tdms_file.columns]  # type: ignore

    # new channel_map - keys were fuzzy matched to a column name and a new dictionary was generated
    if fuzzy_matching:
        channel_map = _fuzzy_match_dictionary(tdms_file.columns, channel_map, fuzzy_match_score)  # type: ignore

    tdms_mapping_warning(tdms_file.columns, channel_map)  # writes warnings if dictionary keys dont match column name

    if extract_all:
        renamed_tdms = tdms_file.rename(columns=channel_map, inplace=False)
        tdms_file = tdms_file.join(renamed_tdms, how="outer", lsuffix="_original", sort=False)

    else:
        tdms_file.rename(columns=channel_map, inplace=True)
        tdms_file.drop(labels=tdms_file.columns.difference(channel_map.values()), axis=1, inplace=True)  # type: ignore

    if drop_duplicates:
        tdms_file = tdms_file.loc[:, ~tdms_file.columns.duplicated()].copy()
    return tdms_file


def form_filepath(eol_test_id, test_type_id, test_id, levels_up) -> Path:
    """
    Forms a file path for a TDMS file.

    This function takes an EOL test ID, a test type ID, a test ID, and the number of levels up as arguments and returns
        a Path object representing the full path to the TDMS file. The path is constructed by joining a filename formed
        from the input arguments with a predefined base directory. If the desired directory is in two levels above the
        calling directory, input 2 in levels_up.

    Args:
        eol_test_id (int): The EOL test ID.
        test_type_id (int): The test type ID.
        test_id (int): The test ID.
        levels_up (int): The number of levels up from the parent directory of the script file.

    Returns:
        Path: A Path object representing the full path to the TDMS file.
    """

    # form filename
    filename = f"{eol_test_id}_{test_type_id}_{test_id}.tdms"

    # Get the parent directory of the script file

    parent_dir = Path(__file__).parent

    # Get up 2 levels in the parent directory
    two_levels_up = parent_dir.parents[levels_up - 1]

    # Define the relative path to the target file
    relative_path = rf"Sample TDMS files\{filename}"

    # Join the parent directory with the relative path
    tdms_file_path = two_levels_up / relative_path

    # Check if the target directory exists
    if not (tdms_file_path.exists()):
        print("Target TDMS directory not found")

    # tdms_file_path = Path(rf"C:\Users\Vadan.Khan\Documents\Project\Sample TDMS files\{filename}")

    return tdms_file_path


def read_tdms(tdms_file_path):
    """
    Reads data from a TDMS file and returns it as a pandas DataFrame.

    This function takes the path to a TDMS file as an argument, reads the data from the file, and converts it into a
        pandas DataFrame. The column names in the DataFrame are renamed according to a predefined channel map, using
        fuzzy matching if necessary. Duplicate columns are removed, and all columns are extracted.

    Args:
        tdms_file_path: The path to the TDMS file to read.

    Returns:
        A pandas DataFrame containing the data from the TDMS file.
    """
    tdms_file = TdmsFile.read(tdms_file_path)

    # Converting TDMS data to a pandas DataFrame
    df = tdms_file.as_dataframe()
    tdms_df = _apply_channel_map(
        df,
        channel_map={"Ambient_new": "Ambient"},
        fuzzy_matching=True,
        drop_duplicates=True,
        extract_all=True,
        fuzzy_match_score=60,
    )
    # Creating a DataFrame with columns of unequal length
    # df = pd.DataFrame({k: pd.Series(v) for k, v in data.items()})
    return tdms_df


def get_time_series_data(tdms_filepath, channel_names: list[str]) -> list[pd.DataFrame]:
    """
    Reads data from a TDMS file and the wanted channels as a list of pandas DataFrames.

    This function takes the path to a TDMS file and a list of channel names as arguments.
    It opens the TDMS file and reads the data for the specified channels, searching in each group for the channel names.
    It returns a list ofof pandas DataFrames, where each DataFrame contains the time and channel data for one of the
    specified channels.

    Args:
        tdms_filepath: The path to the TDMS file to read.
        channel_names: A list of channel names to read data from.

    Returns:
        A list of pandas DataFrames containing the time and channel data for the specified channels.

    """
    tdms_data = []
    try:
        with TdmsFile.open(tdms_filepath) as tdms_file:
            for channel_name in channel_names:
                for group_name in tdms_file.groups():
                    group_name = str(group_name)  # converts to string
                    group_name = group_name.split("'")[1]  # trims to only extract the name between ''
                    group = tdms_file[group_name]
                    if channel_name in group:
                        channel = group[channel_name]  # Specifies the channel to look at
                        time_data = channel.time_track()  # Gets the time values for the channel
                        channel_data = channel[:]  # Gets raw data from the channel into an array
                        d = {
                            (channel_name + "_time"): time_data,
                            channel_name: channel_data,
                        }
                        df = pd.DataFrame(data=d)
                        tdms_data.append(df)
                        break
                else:
                    print(f"Channel '{channel}' not found in any group\n")

    except Exception:
        print("WARNING: Error Looking for Time Channel")
        print(sys.exc_info())
    return tdms_data
