import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/data_processed_01.pkl")


# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------
# JA: isolate set 1 into df
df_set = df[df["set"] == 1]

# JA: use pyplot to view a graph of data
plt.plot(df_set["acc_y"])

# JA: take time off of the index and use integer only
# JA: this can show the number of samples
plt.plot(df_set["acc_y"].reset_index(drop=True))

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------
# JA: get distinct values
df["label"].unique()

# JA: loop over distinct values - split up each exercise type
# JA: (label) into it's own df - This will show all data for
# JA: every participant and time in 1 display (1 display for
# #JA: each exerise type)
for label in df["label"].unique():
    # JA: get next exercise type data into its own df
    subset = df[df["label"] == label]

    # JA: display the data, but only first 2 rows
    # display(subset.head(2))

    # JA: create a figure we can edit later
    fig, ax = plt.subplots()
    plt.plot(subset["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()

# JA: lets see only the first 100 samples instead
# JA: ah now we can see the patterns in each exercise
for label in df["label"].unique():
    # JA: get next exercise type data into its own df
    subset = df[df["label"] == label]

    # JA: create a figure we can edit later
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()

# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------
# JA: tricks to tweak plots to make patterns easier to recognize
# JA: once you run these the adjustments persist until changed again
mpl.style.use("seaborn-v0_8-deep")  # JA: style
mpl.rcParams["figure.figsize"] = (20, 5)  # JA: wider
mpl.rcParams["figure.dpi"] = 100  # JA: good resolution for reports


# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------
# JA: here's a different way of getting a subset of the df
# JA: let's get particip A's squats
# JA: (if you don't remove timestamps (reset_index), the plot will
# JA: show all time in between workouts, basically all the time the
# JA: person was going about their daily life)
df_cat = df.query("label == 'squat'").query("participant == 'A'").reset_index()

# JA: now a grouped plot, splitting category into 1 plot for ea
# JA: distinct
df_cat.groupby(["category"])["acc_y"].plot()

# JA: now let's style it
fig, ax = plt.subplots()
# JA: this splits the plotted data into it's categories
df_cat.groupby(["category"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------
# JA: I want to compare patterns for all participants' heavy squats
df_cat = df.query("label == 'squat'").query("category == 'heavy'").reset_index()

# JA: now let's style it
fig, ax = plt.subplots()
# JA: this splits the plotted data into it's participants
df_cat.groupby(["participant"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()

# JA: Now I want to see participant E's sets to see if they are similar
df_cat = (
    df.query("label == 'squat'")
    .query("category == 'heavy'")
    .query("participant == 'E'")
    .reset_index()
)

# JA: now let's style it
fig, ax = plt.subplots()
# JA: this splits the plotted data into it's participants
df_cat.groupby(["set"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()

# JA: back to the training...
# JA: get bench press, sort by participant
df_par = df.query("label == 'bench'").sort_values("participant").reset_index()

# JA: now let's style it
fig, ax = plt.subplots()
# JA: this splits the plotted data into it's participants
df_par.groupby(["participant"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()


# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------
label = "squat"
participant = "A"
# JA: this is called an fstring, a way to put a variable inside of a string
df_all_axis = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index()
)

fig, ax = plt.subplots()
df_all_axis[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()


# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------
# JA: first get unique list of labels and participants
labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        df_all_axis = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )

        # JA: this dataset doesn't have data for all combinations
        # JA: of exercise and participant, so if it's blank don't plot
        if len(df_all_axis) > 0:

            fig, ax = plt.subplots()
            df_all_axis[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
            ax.set_ylabel("acc_y")
            ax.set_xlabel("samples")
            plt.title(f"{label} ({participant})".title())
            plt.legend()

# JA: now do this for gyro
# JA: first get unique list of labels and participants
labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        df_all_axis = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )

        # JA: this dataset doesn't have data for all combinations
        # JA: of exercise and participant, so if it's blank don't plot
        if len(df_all_axis) > 0:

            fig, ax = plt.subplots()
            df_all_axis[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax)
            ax.set_ylabel("gyr_y")
            ax.set_xlabel("samples")
            plt.title(f"{label} ({participant})".title())
            plt.legend()

# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------
# JA: combine acc and gyr in same plot for one exer and partic
label = "row"
participant = "A"
df_comb_plot = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index()
)

# JA: you can look up the parameters avail for subplots by
# JA: holding control and clicking to launch documentation
# JA: with this we want 2 graphs one on top of another
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
df_comb_plot[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
df_comb_plot[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

# JA: styling
ax[0].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
)
ax[1].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
)
ax[1].set_xlabel("samples")


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------
# JA: first get unique list of labels and participants
labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        df_comb_plot = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )

        # JA: this dataset doesn't have data for all combinations
        # JA: of exercise and participant, so if it's blank don't plot
        if len(df_comb_plot) > 0:
            # JA: with this we want 2 graphs one on top of another
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
            df_comb_plot[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
            df_comb_plot[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

            # JA: styling
            ax[0].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                fancybox=True,
                shadow=True,
            )
            ax[1].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                fancybox=True,
                shadow=True,
            )
            ax[1].set_xlabel("samples")

            # JA: save figures
            plt.savefig(f"../../reports/figures/{label.title()} ({participant}).png")
            plt.show()
