import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import *
from sklearn.cluster import KMeans


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/02_data_outliers_rem_chauvenets.pkl")

#JA: I think I might not have all the excersizes
df["label"].unique()

# JA: get first 6 col names (this autoskips the index col name)
columns_predictor = list(df.columns[:6])

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------
# JA: see that there is missing data in each predictor column
df.info()

# JA: observe missing data
df_subset = df[df["set"] == 35]["gyr_y"].plot()

# JA: we can either amputate records with missing data... but we may cut off a lot of data
# JA: or we can impute instead of amputate, which means calculate what it would be
# JA: if it was there - We will use straight line to connect end and beginning pointss
for col in columns_predictor:
    df[col] = df[col].interpolate()

df.info()


# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------
# JA: how long does a repetition take - we use this to filter out the noise
# JA: look at these waves - see their patterns
df[df["set"] == 25]["acc_y"].plot()
df[df["set"] == 50]["acc_y"].plot()

# JA: -1 is the last frame of the dataframe and 0 is the first
# JA: this is the time diff between first and last rep
duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]
duration.seconds

for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]

    duration = stop - start
    # JA: add duration time diff as new col in the df - col name duration
    df.loc[(df["set"] == s), "duration"] = duration.seconds

# JA: find average duration of each set of type of set (heavy medium etc)
df_duration = df.groupby(["category"])["duration"].mean()

# JA: average duration of heavy [0] vs medium [1] set (using number of reps in teaching data)
df_duration.iloc[0] / 5
df_duration.iloc[1] / 10


# --------------------------------------------------------------
# Butterworth lowpass filter (frequency reduction)
# --------------------------------------------------------------
# JA: this filter basically resamples the data around common movement in the data
#JA: it smooths out a wave form
#JA: mostly used to smooth out jerky movements

df_lowpass = df.copy()
LowPass = LowPassFilter()

freq_sampling = 1000 / 200  # JA: 1 sec div by how many ms (our data is every 200ms)
# JA: the higher this number, the less we resample the data - higher jagged lower smoother
#JA: manipulate this number until you get good results
freq_cutoff = 1.20

df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", freq_sampling, freq_cutoff, order=5)

subset = df_lowpass[df_lowpass["set"] == 45]
print(subset["label"][0])

# JA: visualize the orig col vs the new lowpass col we just created
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

# JA: loop for all predictor cols
for col in columns_predictor:
    # JA: not adding cols here - overwriting cols
    # JA: to be clear, first generate lowpass col - then overwrite - then del lowpass col
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, freq_sampling, freq_cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]
    
    
"""JA: this is where I'm a bit unsure... why are we smoothing out original data? Wouldn't
that make it possible to over-smooth a movement and then the model will think an
exercise happened when it didn't? Also we will have to smooth out all incoming
data, which again means we might lose too much data and the model will fail to predict
correctly. (Later in this series this guy tests against the smoothed out data. He gets
99.5% accuracy, but that's against smoothed data. Not sure I completely agree here.)"""


# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------
# JA: used in machine learning to reduce the complexity of data by
# JA: transforming the data into a new set of variables called principal component
# JA: basically merges variables into most relevant data points, reducing complexity
# and noise
df_pca = df_lowpass.copy()

PCA = PrincipalComponentAnalysis()

#JA: determine the optimal amount of principal components - get each col's variance
pc_values = PCA.determine_pc_explained_variance(df_pca, columns_predictor)

# JA: the optimal component number is chosen as the number of components that capture
# JA: the most variance with the least amount of components (cols). The elbow, which
# JA: we are about to caluclate, is the change at which removing another component
# JA: reduces the variance by a more significant amount
plt.figure(figsize=(10, 10))
plt.plot(range(1, len(columns_predictor) + 1), pc_values)
plt.xlabel("principal component number")
plt.ylabel("explained variance")
plt.show()
# JA: and this graph shows the elbow is at 3 - drops off after 3

# JA: so now we will combine our 6 value (predictor) cols into 3 - We will check
# later to make sure the 3 cols predict better than 6
df_pca = PCA.apply_pca(df_pca, columns_predictor, 3)

subset = df_pca[df_pca["set"] == 35]
subset[["pca_1", "pca_2", "pca_3"]].plot()


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------
# JA: we are going to combine 3 gyro or acc readings which them makes it impervious
# to direction or orientation - It's simply a magnitude of movement
# r = square root of (square the readings and add that together)
df_squared = df_pca.copy()

#JA: do the math and add features to df
acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

subset = df_squared[df_squared["set"] == 14]
subset[["acc_r", "gyr_r"]].plot(subplots=True)


# --------------------------------------------------------------
# Temporal abstraction (frequency reduction)
# --------------------------------------------------------------
#JA: rolling averages from pandas
df_tempabs = df_squared.copy()

NumAbs = NumericalAbstraction()

# JA: add columns to our predictor array
columns_predictor = list(df_tempabs.columns[:6]) + ["acc_r", "gyr_r"]

# JA: define window size (size of how far to look back in data when
# doing calculation)
# JA: remember when you set your window size you have to throw away your
# first x records due to the fact the first x records are actually pulling
# data from a different type of excersize, making them invalid

# JA: lets create a window size - we want to have a window size of 1 sec, which
# means we need 5 data points since they are 200ms each
ws = int(1000 / 200)

# JA: for each record calculate the mean and standard dev based on window size
# JA: this adds 16 cols (8 cols, mean and std)
for col in columns_predictor:
    df_tempabs = NumAbs.abstract_numerical(df_tempabs, [col], ws, "mean")
    df_tempabs = NumAbs.abstract_numerical(df_tempabs, [col], ws, "std")

# JA: in order to fix the part where we have to throw away the first x data
# points, we will break this df up into sets, so we only loop over each set
# without using data from previous sets
df_tempabs_list = []

for s in df_tempabs["set"].unique():
    # JA: here we gather the next set to work with from the total df
    subset = df_tempabs[df_tempabs["set"] == s].copy()

    for col in columns_predictor:
        subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], ws, "std")

    # JA: gather each subset into an array df
    df_tempabs_list.append(subset)

# JA: now pivot df_tempabs_list
df_tempabs = pd.concat(df_tempabs_list)

# JA: looking at the results of one set
subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
subset[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()

df_tempabs.info()


# --------------------------------------------------------------
# Discrete Fourier Transformation (DFT) (Freq Complexity Reduction)
# --------------------------------------------------------------
# JA: decompose orig signal into it's component frequencies (into its different waves)
df_freq = df_tempabs.copy().reset_index()

FreqAbs = FourierTransformation()

#JA: sampling rate and window size
sr = int(1000 / 200)
ws = int(2000 / 200) #JA: average rate of a rep, which was about 1 sec

# JA: split this wave into component freq
df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], ws, sr)
df_freq.info()

# JA: look at a set to see what it looks like, with a few of the new cols
subset = df_freq[df_freq["set"] == 35]
subset[["acc_y"]].plot()
subset[
    [
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_pse",
        "acc_y_freq_1.5_Hz_ws_10",
        "acc_y_freq_2.5_Hz_ws_10",
    ]
].plot()

# JA: now loop over all cols
df_freq_list = []

for s in df_freq["set"].unique():
    # JA: this might take a while, so this lets you know where the loop is at
    # during processing (this doesnt really work because to do this you'd have
    # to do it in seq order)
    print(f"Applying Fourier transformations to set {s}")
    # JA: here we gather the next set to work with from the total df
    subset = df_freq[df_freq["set"] == s].copy().reset_index(drop=True)
    # JA: this function acutally takes a list of columns, so we don't have to break them out by hand
    subset = FreqAbs.abstract_frequency(subset, columns_predictor, ws, sr)

    # JA: gather each subset into an array df
    df_freq_list.append(subset)

# JA: now pivot df_freq_list
# JA: reset index to orig
df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------
# JA: we want to avoid overlapping windows because we do not want to CREATE
# highly correlated data. This can cause overfitting. We will allow for a
# certain percentage of overlap and remove the rest of the data.
df_overlap = df_freq.dropna()

# JA: getting rid of 50% of the data is recommended - lots of data loss but more accurate
# JA: this iloc says I want every other row please
# JA: this reduces created correlation
df_overlap.iloc[::2]

# --------------------------------------------------------------
# Kmeans Clustering (Removing complexity)
# --------------------------------------------------------------
# JA: kmeans clustering
# JA: k is the total clusters we want, and this will cluster our data around
# the central points of our data (unsupervised) - we will use the elbow method
# to figure out how many clusters we should use
df_clust = df_overlap.copy()

columns_cluster = ["acc_y", "acc_x", "acc_z"]
#JA: this is the number of clusters we will test, 2 clusters to 10 clusters
k_range = range(2, 10)
intertias = []  # JA: results

for k in k_range:
    # JA: get subset then produce a kmeans
    subset = df_clust[columns_cluster]
    kmeans_result = KMeans(n_clusters=k, n_init=20, random_state=0)
    # JA: then get fit predict
    #JA: train the model (fit) and then do a prediction, so we know which k is best
    cluster_labels = kmeans_result.fit_predict(subset)
    # JA: store the inertias so we can use elbow
    #JA: inertia is how accurate the fit was
    intertias.append(kmeans_result.inertia_)

# JA: lets look at it to determine elbow
plt.figure(figsize=(10, 10))
plt.plot(k_range, intertias)
plt.xlabel("k")
plt.ylabel("Sum of squared distances")
plt.show()

# JA: now that we know the elbow is 5, lets grab that data
kmeans_result = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_clust[columns_cluster]
# JA: add kmeans as a new col in the df
df_clust["cluster"] = kmeans_result.fit_predict(subset)

# JA: plot clusters
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in df_clust["cluster"].unique():
    subset = df_clust[df_clust["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
df_clust.to_pickle("../../data/interim/03_data_features.pkl")
