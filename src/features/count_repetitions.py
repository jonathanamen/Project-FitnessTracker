import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None

# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
#JA: start with original data processed instead of data features file
df = pd.read_pickle("../../data/interim/data_processed_01.pkl")
#JA: remove rest rows from df
df = df[df["label"] != "rest"]

#JA: calculate sum of squares to remove direction
# JA: we are going to combine 3 gyro or acc readings which them makes it impervious
# to direction or orientation - It's simply a magnitude of movement
# r = square root of (square the readings and add that together)

#JA: do the math and add features to df
acc_r = df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2
gyr_r = df["gyr_x"] ** 2 + df["gyr_y"] ** 2 + df["gyr_z"] ** 2
df["acc_r"] = np.sqrt(acc_r)
df["gyr_r"] = np.sqrt(gyr_r)


# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------
#JA: split data into each exercise to make it easier to analyze data
df_bench = df[df["label"] == "bench"]
df_squat = df[df["label"] == "squat"]
df_row = df[df["label"] == "row"]
df_ohp = df[df["label"] == "ohp"]
df_dead = df[df["label"] == "dead"]


# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------
#JA: plot data to see patters to help us count reps
df_plot = df_bench

#JA: look at each measure to see if we can find something that makes reps obvious
df_plot[df_plot["set"] == df_plot["set"].unique()[0]]["acc_x"].plot()
df_plot[df_plot["set"] == df_plot["set"].unique()[0]]["acc_y"].plot()
df_plot[df_plot["set"] == df_plot["set"].unique()[0]]["acc_z"].plot()
df_plot[df_plot["set"] == df_plot["set"].unique()[0]]["acc_r"].plot()

df_plot[df_plot["set"] == df_plot["set"].unique()[0]]["gyr_x"].plot()
df_plot[df_plot["set"] == df_plot["set"].unique()[0]]["gyr_y"].plot()
df_plot[df_plot["set"] == df_plot["set"].unique()[0]]["gyr_z"].plot()
df_plot[df_plot["set"] == df_plot["set"].unique()[0]]["gyr_r"].plot()


# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------
#JA: lets remove some noise to see if it makes it easier to see reps, using lowpass
fs = 1000 / 200
LowPass = LowPassFilter()


# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------
df_set_bench = df_bench[df_bench["set"] == df_bench["set"].unique()[0]]
df_set_squat = df_squat[df_squat["set"] == df_squat["set"].unique()[0]]
df_set_row = df_row[df_row["set"] == df_row["set"].unique()[0]]
df_set_ohp = df_ohp[df_ohp["set"] == df_ohp["set"].unique()[0]]
df_set_dead = df_dead[df_dead["set"] == df_dead["set"].unique()[0]]

df_set_bench["acc_r"].plot()



#JA: lets smooth it out
column = "acc_r"
#JA: 0.4 is best - acc_r is best
LowPass.low_pass_filter(
    df_set_bench, col=column, sampling_frequency=fs, cutoff_frequency=0.4, order=5
)[column + "_lowpass"].plot()

LowPass.low_pass_filter(
    df_set_bench, col=column, sampling_frequency=fs, cutoff_frequency=0.9, order=5
)[column + "_lowpass"].plot()

column = "acc_y"
LowPass.low_pass_filter(
    df_set_bench, col=column, sampling_frequency=fs, cutoff_frequency=0.4, order=5
)[column + "_lowpass"].plot()

#JA: try other workouts
column = "acc_r"
LowPass.low_pass_filter(
    df_set_row, col=column, sampling_frequency=fs, cutoff_frequency=0.8, order=5
)[column + "_lowpass"].plot()

df_set_row["acc_r"].plot()


# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------
#JA: lets get a function from scipy that will show us max and min in the wave to count reps
#JA: get lowpass first
column = "acc_r"
df_lowpass = LowPass.low_pass_filter(
    df_set_bench, col=column, sampling_frequency=fs, cutoff_frequency=0.4, order=5
)
indexes_greater = argrelextrema(df_lowpass[column + "_lowpass"].values, np.greater)
#JA: now identify where the index aligns with the dataframe
df_index_hits = df_lowpass.iloc[indexes_greater]

def count_reps(dataset, cutoff=0.4, order=10, column="acc_r"):
    df_data = LowPass.low_pass_filter(
        dataset, col=column, sampling_frequency=fs, cutoff_frequency=cutoff, order=order
    )
    
    indexes_greater = argrelextrema(df_data[column + "_lowpass"].values, np.greater)
    #JA: now identify where the index aligns with the dataframe
    peaks = df_data.iloc[indexes_greater]
    
    fig, ax = plt.subplots()
    plt.plot(df_data[f"{column}_lowpass"])
    plt.plot(peaks[f"{column}_lowpass"], "o", color="red")
    ax.set_ylabel(f"{column}_lowpass")
    df_exercise = df_data["label"].iloc[0].title()
    df_category = df_data["category"].iloc[0].title()
    plt.title(f"{df_category} {df_exercise}: {len(peaks)} Reps")
    plt.show()
    
    return len(peaks)

#JA: now lets try it out
count_reps(df_set_bench, cutoff=0.4)
count_reps(df_set_squat, cutoff=0.35) #JA: this data looks messed up
count_reps(df_set_row, cutoff=0.65) #, column="gyr_x")
count_reps(df_set_dead, cutoff=0.5)
#JA: would need to tweak this a lot because it doesn't work well



# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------
#JA: we're going to tell the dataframe how many reps each set should have
df["reps"] = df["category"].apply(lambda x: 5 if x == "heavy" else 10)
#JA: now lets group the dataframes
#JA: grouping by 3 cols (get distinct) then aggregate (max)
#JA: this is what each set should count for reps, assuming no erroneous data or
# other tweaks we still need to make
df_rep = df.groupby(["label", "category", "set"])["reps"].max().reset_index()
#JA: creating a col for the counted reps
df_rep["reps_counted"] = 0

#JA: now lets create a loop for the dataframe and count the reps
for s in df["set"].unique():
    subset = df[df["set"] == s]
    
    column = "acc_r"
    
    #JA: set cuttoff based on exercise type (label)
    if subset["label"].iloc[0] == "bench":
        cutoff = 0.4
    if subset["label"].iloc[0] == "squat":
        cutoff = 0.35
    if subset["label"].iloc[0] == "row":
        cutoff = 0.65
    if subset["label"].iloc[0] == "dead":
        cutoff = 0.5
    if subset["label"].iloc[0] == "ohp":
        cutoff = 0.35
    
    #JA: count reps for this set    
    reps = count_reps(subset, cutoff=cutoff, column=column)
    
    #JA: add overwrite reps_counted with how many reps were counted
    df_rep.loc[df_rep["set"] == s, "reps_counted"] = reps
    
df_rep



# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------
#JA: capture error
error = mean_absolute_error(df_rep["reps"], df_rep["reps_counted"]).round(2)
df_rep.groupby(["label", "category"])["reps", "reps_counted"].mean().plot.bar()