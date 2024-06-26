import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

df = pd.read_pickle("../../data/interim/03_data_features.pkl")


# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------
# JA: get rid of these columns
df_train = df.drop(["participant", "category", "set"], axis=1)

# JA: splitting the dataset into the x and y
x = df_train.drop("label", axis=1)
y = df_train["label"]

# JA: train our data using train_test_split
# JA: need to set the same random for both
# we're training with 75% (0.25) of the dataset and testing with 25%
# We're also using stratify to make sure there is an equal distribution of our exersize types
# JA: train test split comes with a shuffle to try and
# have a sample of each type of data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42, stratify=y
)

# JA: this graph shows us our test and train datapoints, to make sure
# we have enough data from all exercises
fig, ax = plt.subplots(figsize=(10, 5))
df_train["label"].value_counts().plot(
    kind="bar", ax=ax, color="lightblue", label="Total"
)

y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")
y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
plt.legend()
plt.show()


# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------
# JA: split features to check if features actually help with model training
features_basic = df_train.columns[:6] #JA: 6 - Tot 6
#JA: skip 2 - tot 6 (8)
features_pca = ["pca_1", "pca_2", "pca_3"] #JA: 3 more - tot 9 (11)
features_square = ["acc_r", "gyr_r"] #JA: add 2 - tot 11 (13)
# JA: each column name in df_train, put name into f, then check if
# _temp_ exists in the name, if so put it into the list
features_time = [f for f in df_train.columns if "_temp_" in f] #JA: add 16 - 27 (29)
features_freq = [f for f in df_train.columns if ("_freq" in f) or ("_pse" in f)]
features_clust = ["cluster"]

#JA: column number check
df_cols = [f for f in df_train.columns] #JA: list of all cols

print("basic:", len(features_basic))
print("square:", len(features_square))
print("pca:", len(features_pca))
print("time:", len(features_time))
print("freq:", len(features_freq))
print("clust:", len(features_clust))

feat_col_num = (len(features_basic)
    + len(features_square)
    + len(features_pca)
    + len(features_time)
    + len(features_freq)
    + len(features_clust))

print("total cols in features:", feat_col_num) #JA: 100 cols
len(df_train.columns) #JA: 102 cols

#JA: a set will remove duplicate entries, but a list will not
# so here we want to use set to concat and remove dupes but then
# turn it back into a list
#JA: we will use these feature sets to iterate over the model to see the differences
feature_set_1 = list(set(features_basic))
feature_set_2 = list(set(feature_set_1 + features_square + features_pca))
feature_set_3 = list(set(feature_set_2 + features_time))
feature_set_4 = list(set(feature_set_3 + features_freq + features_clust))
#JA: somehow duration label got into the feature set - see if you can find it
feature_set_4.__contains__("Duration") #JA: hmm I don't see it


# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------
#JA: find the best performing feature
learner = ClassificationAlgorithms()

#JA: this function will test all features to find the best, then go back
# and find the next best, etc, and will return these features in order of accuracy
features_max = 10
features_selected, features_ordered, scores_ordered = learner.forward_selection(features_max, x_train, y_train)

#JA: plot to view results
plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, features_max + 1, 1), scores_ordered)
plt.xlabel("Number of features")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1, features_max + 1, 1))
plt.show()

#JA: store features so we don't lose them
features_selected = [
    'acc_y_freq_0.0_Hz_ws_10',
    'duration',
    'acc_x_freq_0.0_Hz_ws_10',
    'acc_z_temp_mean_ws_5',
    'gyr_r_temp_mean_ws_5',
    'gyr_z_freq_1.0_Hz_ws_10',
    'gyr_x_max_freq',
    'acc_z_pse',
    'gyr_r_freq_2.0_Hz_ws_10',
    'gyr_x_freq_2.5_Hz_ws_10'
]



# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------
#JA: find the best hyper parameters for my model
feature_sets_possible = [
    feature_set_1
    ,feature_set_2
    ,feature_set_3
    ,feature_set_4
    ,features_selected
]

feature_names = [
    "Feature Set 1"
    ,"Feature Set 2"
    ,"Feature Set 3"
    ,"Feature Set 4"
    ,"Selected Features"
]

iterations = 1
df_score = pd.DataFrame()

#JA: training 5 models with our features - stored in df
for i, f in zip(range(len(feature_sets_possible)), feature_names):
    print("Feature set:", i)
    selected_train_X = x_train[feature_sets_possible[i]]
    selected_test_X = x_test[feature_sets_possible[i]]

    # First run non deterministic classifiers to average their score.
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            y_train,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(y_test, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    df_score = pd.concat([df_score, new_scores])



# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------
df_score.sort_values(by="accuracy", ascending=False)

plt.figure(figsize=(10, 10))
sns.barplot(x="model", y="accuracy", hue="feature_set", data=df_score)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0.7,1)
plt.legend(loc="lower right")
plt.show()



# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------
#JA: using random forest, feature set 4
#JA: lets train it on a confusion matrix
#JA: get random forest guts
(
class_train_y,
class_test_y,
class_train_prob_y,
class_test_prob_y,
) = learner.random_forest(
x_train[feature_set_4], y_train, x_test[feature_set_4], gridsearch=True
)

#JA: calculate accuracy
accuracy = accuracy_score(y_test,class_test_y)

#JA: create confusion matrix
classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

#JA: graph it or whatever
# create confusion matrix for cm
#JA: this shows were wrong predictions occur
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()



# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------
#JA: now subtract participant A from the training data, so that we provide
# only 4 of the participants to the training and see if it can predict a
# person it's never seen before
#JA: split
df_participants = df.drop(["set", "category"], axis=1)
x_train = df_participants[df_participants["participant"] != "A"].drop("label",axis=1)
y_train = df_participants[df_participants["participant"] != "A"]["label"]

x_test = df_participants[df_participants["participant"] == "A"].drop("label",axis=1)
y_test = df_participants[df_participants["participant"] == "A"]["label"]

#JA: drop participant - best practice
x_train = x_train.drop(["participant"], axis=1)
x_test = x_test.drop(["participant"], axis=1)

#JA: make sure we have enough data in each excersize type
fig, ax = plt.subplots(figsize=(10, 5))
df_train["label"].value_counts().plot(
    kind="bar", ax=ax, color="lightblue", label="Total"
)

y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")
y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
plt.legend()
plt.show()



# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------
#JA: using random forest, features selected
#JA: lets train it on a confusion matrix
#JA: get random forest guts
(
class_train_y,
class_test_y,
class_train_prob_y,
class_test_prob_y,
) = learner.feedforward_neural_network(
x_train[features_selected], y_train, x_test[features_selected], gridsearch=True
)

#JA: calculate accuracy
accuracy = accuracy_score(y_test,class_test_y)

#JA: create confusion matrix
classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

#JA: graph it or whatever
# create confusion matrix for cm
#JA: this shows were wrong predictions occur
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()



# --------------------------------------------------------------
# Try a simpler model with the selected features
# --------------------------------------------------------------
#JA: using NN
#JA: lets train it on a confusion matrix
#JA: get random forest guts
(
class_train_y,
class_test_y,
class_train_prob_y,
class_test_prob_y,
) = learner.random_forest(
x_train[feature_set_4], y_train, x_test[feature_set_4], gridsearch=False
)

#JA: calculate accuracy
accuracy = accuracy_score(y_test,class_test_y)

#JA: create confusion matrix
classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

#JA: graph it or whatever
# create confusion matrix for cm
#JA: this shows were wrong predictions occur
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()