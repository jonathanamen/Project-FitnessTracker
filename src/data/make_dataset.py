import pandas as pd
from glob import glob


# --------------------------------------------------------------
# JA: Read single CSV file -------------------------------------
# --------------------------------------------------------------
# single_file_acc = pd.read_csv(
#     "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
# )

# single_file_gyr = pd.read_csv(
#     "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
# )


# --------------------------------------------------------------
# JA: List all data in data/raw/MetaMotion ---------------------
# --------------------------------------------------------------
files = glob("../../data/raw/MetaMotion/*.csv")
len(files)


# --------------------------------------------------------------
# JA: Extract features from filename ---------------------------
# --------------------------------------------------------------
#JA: this is a discovery code block
data_path = "../../data/raw/MetaMotion\\"
f = files[2]

participant = f.split("-")[0].replace(data_path, "")
label = f.split("-")[1]
category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

# JA: create dataframe
df = pd.read_csv(f)
# JA: add columns with data extracted from above
df["participant"] = participant
df["label"] = label
df["category"] = category


# --------------------------------------------------------------
# JA: Read all files -------------------------------------------
# --------------------------------------------------------------
df_acc = pd.DataFrame()
df_gyr = pd.DataFrame()

set_acc = 1
set_gyr = 1

data_path = "../../data/raw/MetaMotion\\"

# JA: for all files, loop
for f in files:

    # JA: get attributes
    participant = f.split("-")[0].replace(data_path, "")
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

    # print(participant)
    # print(label)
    # print(category)

    # JA: add data to dataframe
    df = pd.read_csv(f)

    # JA: add attribute categories to df
    df["participant"] = participant
    df["label"] = label
    df["category"] = category

    # JA: split acc and gyr data
    if "Acc" in f:
        # JA: first add the iteration
        df["set"] = set_acc
        set_acc += 1
        # JA: then append data
        df_acc = pd.concat([df_acc, df])
    else:
        # JA: first add the iteration
        df["set"] = set_gyr
        set_gyr += 1
        # JA: then append data
        df_gyr = pd.concat([df_gyr, df])

# JA: get dataset (file) 1 of the acc dataframe
# df_acc[df_acc["set"] == 1]


# --------------------------------------------------------------
# JA: Working with datetimes -----------------------------------
# --------------------------------------------------------------
# JA: get datatypes of dataframe
df_acc.info()

# JA: convert int64 of unix time to datetime
pd.to_datetime(df["epoch (ms)"], unit="ms")
# JA: get time col to see the diff between unix time and recorded time (prob DLT)
# df["time (01:00)"]
# JA: convert object to datetime, so we can get month etc
pd.to_datetime(df["time (01:00)"])

# JA: create indexes for df (primary key)
df_acc.index = pd.to_datetime(df_acc["epoch (ms)"], unit="ms")
df_gyr.index = pd.to_datetime(df_gyr["epoch (ms)"], unit="ms")

# JA: now that we have time and an index, delete other time cols
del df_acc["epoch (ms)"]
del df_acc["time (01:00)"]
del df_acc["elapsed (s)"]

del df_gyr["epoch (ms)"]
del df_gyr["time (01:00)"]
del df_gyr["elapsed (s)"]


# --------------------------------------------------------------
# JA: Turn into function (package all the code so it can be ran
# JA: together with ease) --------------------------------------
# --------------------------------------------------------------
def read_data_from_files(files):
    df_acc = pd.DataFrame()
    df_gyr = pd.DataFrame()

    set_acc = 1
    set_gyr = 1

    data_path = "../../data/raw/MetaMotion\\"

    # JA: for all files, loop
    for f in files:

        # JA: get attributes
        participant = f.split("-")[0].replace(data_path, "")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

        # JA: add data to dataframe
        df = pd.read_csv(f)

        # JA: add attribute categories to df
        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        # JA: split acc and gyr data
        if "Acc" in f:
            # JA: first add the iteration
            df["set"] = set_acc
            set_acc += 1
            # JA: then append data
            df_acc = pd.concat([df_acc, df])
        else:
            # JA: first add the iteration
            df["set"] = set_gyr
            set_gyr += 1
            # JA: then append data
            df_gyr = pd.concat([df_gyr, df])

    # JA: create indexes for df (primary key)
    df_acc.index = pd.to_datetime(df_acc["epoch (ms)"], unit="ms")
    df_gyr.index = pd.to_datetime(df_gyr["epoch (ms)"], unit="ms")

    # JA: now that we have time and an index, delete other time cols
    del df_acc["epoch (ms)"]
    del df_acc["time (01:00)"]
    del df_acc["elapsed (s)"]

    del df_gyr["epoch (ms)"]
    del df_gyr["time (01:00)"]
    del df_gyr["elapsed (s)"]

    return df_acc, df_gyr


# JA: now read flies and process into dataframes
files = glob("../../data/raw/MetaMotion/*.csv")

df_acc, df_gyr = read_data_from_files(files)


# --------------------------------------------------------------
# JA: Merging datasets -----------------------------------------
# --------------------------------------------------------------
# JA: concat dfs based on columns (0 is rows)
# JA: using iloc to select up to the 3rd column for acc data
df_merge = pd.concat([df_acc.iloc[:, :3], df_gyr], axis=1)

# JA: rename cols for readability
df_merge.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]


# --------------------------------------------------------------
# JA: Resample data (frequency conversion) ---------------------
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

# JA: use offset alias to resample time from ms to 200ms, so we have
# JA: in every record for both acc and gyr - we're taking the
# JA: mean of any records with multiple entries
df_merge[:1000].resample(rule="200ms").mean()

# JA: but then we lose the attributes, so let's change it
# JA: first save a dictionary of col names and what we want the grouping
# JA: to use as an aggregate function
sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last",
}

# JA: see what it looks like resampled with the dictionary - this
# JA: doesn't actually save the resample - that happens below
# JA: when we do concat and drop nulls
df_merge[:1000].resample(rule="200ms").apply(sampling)
# df_merge[df_merge["set"] == 3]

# JA: if you run the above on the whole dataframe pandas will fill
# JA: time gaps for all the time from 1st entry to last, which
# JA: would create a HUGE dataframe - no bueno - lets remedy that
# JA: split the dataframe into days (1 df for each day) and drop
# JA: nulls, which will be the timestamps with no data (no workouts
# JA: were being done during that time)
days = [g for n, g in df_merge.groupby(pd.Grouper(freq="D"))]
# days[0]

# JA: loop over days, resampling and dropping nulls for each dataframe
# JA: in days
df_resampled = pd.concat(
    [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
)

df_resampled.info()

# JA: change datatype of set from float to int
df_resampled["set"] = df_resampled["set"].astype("int")

# --------------------------------------------------------------
# JA: Export dataset -------------------------------------------
# --------------------------------------------------------------
# JA: this is a great export that is readable by python
# JA: can also export to csv, excel, etc
df_resampled.to_pickle("../../data/interim/data_processed_01.pkl")
df_resampled.to_csv("../../data/interim/data_processed_csv_01.csv")


# --------------------------------------------------------------
# JA: function for merging and resampling ----------------------
# --------------------------------------------------------------
def merge_resample_2_df(df_1, df_2):
    # JA: concat dfs based on columns (0 is rows)
    # JA: using iloc to select up to the 3rd column for acc data
    df_merge = pd.concat([df_1.iloc[:, :3], df_2], axis=1)

    # JA: rename cols for readability
    df_merge.columns = [
        "acc_x",
        "acc_y",
        "acc_z",
        "gyr_x",
        "gyr_y",
        "gyr_z",
        "participant",
        "label",
        "category",
        "set",
    ]

    # JA: first save a dictionary of col names and what we want the grouping
    # JA: to use as an aggregate function
    sampling = {
        "acc_x": "mean",
        "acc_y": "mean",
        "acc_z": "mean",
        "gyr_x": "mean",
        "gyr_y": "mean",
        "gyr_z": "mean",
        "participant": "last",
        "label": "last",
        "category": "last",
        "set": "last",
    }

    # JA: split the dataframe into days (1 df for each day) and drop
    # JA: nulls, which will be the timestamps with no data (no workouts
    # JA: were being done during that time)
    days = [g for n, g in df_merge.groupby(pd.Grouper(freq="D"))]

    # JA: loop over days, resampling and dropping nulls for each dataframe
    # JA: in days
    df_resampled = pd.concat(
        [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
    )

    # JA: change datatype of set from float to int
    df_resampled["set"] = df_resampled["set"].astype("int")

    return df_resampled


# JA: merge, resample, and export
df_resampled = merge_resample_2_df(df_acc, df_gyr)
df_resampled.to_pickle("../../data/interim/data_processed_01.pkl")
df_resampled.to_csv("../../data/interim/data_processed_csv_01.csv")
