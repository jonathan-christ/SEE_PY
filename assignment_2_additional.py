import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

"""
ADDITIONAL TASK:

✅ Create a callable function in each step. 
✅ Check if these callable function has the same result 
(and insights) with your Jupyter Notebook.
 
"""


def import_data() -> pd.DataFrame:
    """
    Imports data from a CSV file into a pandas DataFrame.

    Reads the 'med_events.csv' file, assigns column names, and converts 
    the 'eksd' column to datetime format. Returns a cleaned DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with columns ['pnr', 'eksd', 'perday', 
        'ATC', 'dur_original'] where 'eksd' is converted to datetime.
    """

    tidy = pd.read_csv("med_events.csv")
    # add columns
    tidy.columns = ["pnr", "eksd", "perday", "ATC", "dur_original"]
    tidy["eksd"] = pd.to_datetime(tidy["eksd"], errors="coerce")

    return tidy


def filter_data(df, atc_code):
    """
    Filters the given DataFrame to only include rows with the given ATC code.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        atc_code (str): ATC code to filter by.

    Returns:
        pd.DataFrame: Filtered DataFrame, a copy of the original.
    """
    return df[df["ATC"] == atc_code].copy()


def prepare_data(df):
    """
    Prepares the given DataFrame for SEE analysis.

    Sorts the DataFrame by ['pnr', 'eksd'] in ascending order, adds a 'prev_eksd'
    column which is the previous 'eksd' value for each patient, and drops any rows
    with missing 'prev_eksd' values.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Prepared DataFrame, a copy of the original.
    """
    df = df.sort_values(by=["pnr", "eksd"])
    df["prev_eksd"] = df.groupby("pnr")["eksd"].shift(1)
    return df.dropna(subset=["prev_eksd"])


def sample_one_per_patient(df):
    """
    Samples one random row per patient from the DataFrame.

    Sets a random seed for reproducibility, groups the DataFrame by 'pnr',
    and samples one row per group (patient). Returns the resulting DataFrame
    with reset index.

    Parameters:
        df (pd.DataFrame): Input DataFrame with patient data.

    Returns:
        pd.DataFrame: DataFrame with one sampled row per patient.
    """

    np.random.seed(1234)
    return df.groupby("pnr").apply(lambda x: x.sample(1)).reset_index(drop=True)


def compute_event_intervals(df):
    """
    Computes the event intervals for each patient in the given DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame with patient data.

    Returns:
        pd.DataFrame: DataFrame with 'event.interval' column, filtered to only
        include rows with event intervals greater than 0.
    """
    df["event.interval"] = (df["eksd"] - df["prev_eksd"]).dt.days.astype(float)
    return df[df["event.interval"] > 0]


def compute_ecdf(df):
    """
    Computes the empirical cumulative distribution function (ECDF) for event intervals.

    Sorts the 'event.interval' column in ascending order and computes the ECDF values.
    Returns a DataFrame containing the sorted event intervals ('x') and their corresponding
    ECDF values ('y').

    Parameters:
        df (pd.DataFrame): Input DataFrame containing 'event.interval' column.

    Returns:
        pd.DataFrame: DataFrame with columns 'x' (sorted event intervals) and 'y' (ECDF values).
    """

    x = np.sort(df["event.interval"])
    y = np.arange(1, len(x) + 1) / len(x)
    return pd.DataFrame({"x": x, "y": y})


def plot_ecdf(df, x, y):
    """
    Plots the ECDF of the event intervals and the 80% ECDF.

    Parameters:
        df (pd.DataFrame): DataFrame with 'x' (sorted event intervals) and 'y' (ECDF values) columns.
        x (ndarray): Sorted event intervals.
        y (ndarray): ECDF values.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(df["x"], df["y"], label="80% ECDF")
    plt.title("80% ECDF")
    plt.subplot(1, 2, 2)
    plt.plot(x, y, label="100% ECDF")
    plt.title("100% ECDF")
    plt.show()


def plot_density(df):
    """
    Plots the density of the log(event interval) column in the given DataFrame using seaborn's kdeplot.

    Parameters:
        df (pd.DataFrame): DataFrame containing the 'event.interval' column.

    Returns:
        None
    """
    sns.kdeplot(np.log(df["event.interval"]), label="Log(event interval)")
    plt.title("Log(event interval)")
    plt.show()


def cluster_data(df, algorithm):
    """
    Performs clustering on the given DataFrame using either K-Means or DBSCAN.

    Parameters:
        df (pd.DataFrame): DataFrame containing the 'x' column.
        algorithm (str): Algorithm to use, either 'kmeans' or 'dbscan'.

    Returns:
        pd.DataFrame: DataFrame with the added 'cluster' column.

    Raises:
        ValueError: If `algorithm` is not 'kmeans' or 'dbscan'.
    """
    data = StandardScaler().fit_transform(df[["x"]])
    if algorithm == "kmeans":
        scores = [
            silhouette_score(data, KMeans(n, random_state=1234).fit_predict(data))
            for n in range(2, 11)
        ]
        best_n = np.argmax(scores) + 2
        df["cluster"] = KMeans(n_clusters=best_n, random_state=1234).fit_predict(data)
    elif algorithm == "dbscan":
        df["cluster"] = DBSCAN(eps=0.5, min_samples=5).fit_predict(data)
    else:
        raise ValueError("Invalid algorithm")
    return df


def assign_clusters(event_df, cluster_df):
    """
    Assigns the cluster labels to the original event data based on the cluster intervals.

    Parameters:
        event_df (pd.DataFrame): DataFrame containing the original event data.
        cluster_df (pd.DataFrame): DataFrame containing the cluster intervals.

    Returns:
        pd.DataFrame: DataFrame with the added 'Final_cluster' column, containing the cluster label.
    """
    summary = (
        cluster_df.groupby("cluster")["x"].agg(["min", "max", "median"]).reset_index()
    )
    event_df = pd.merge(event_df, summary, how="cross")
    event_df["Final_cluster"] = np.where(
        (event_df["event.interval"] >= event_df["min"])
        & (event_df["event.interval"] <= event_df["max"]),
        event_df["cluster"],
        np.nan,
    )
    return event_df.dropna(subset=["Final_cluster"])


def see(arg1, tidy: pd.DataFrame, algorithm="kmeans"):

    """
    Runs the SEE algorithm on the given DataFrame.

    Parameters:
        arg1 (str): ATC code to filter the data by.
        tidy (pd.DataFrame): Tidy DataFrame containing the data.
        algorithm (str): Algorithm to use, either 'kmeans' or 'dbscan'.

    Returns:
        pd.DataFrame: DataFrame with the added 'cluster' and 'median' columns.
    """
    C09CA01 = filter_data(tidy, arg1)
    Drug_see_p1 = compute_event_intervals(sample_one_per_patient(prepare_data(C09CA01)))
    dfper = compute_ecdf(Drug_see_p1)
    dfper = dfper[dfper["y"] <= 0.8]
    plot_ecdf(
        dfper,
        np.sort(Drug_see_p1["event.interval"]),
        np.arange(1, len(Drug_see_p1) + 1) / len(Drug_see_p1),
    )
    plot_density(Drug_see_p1)
    dfper = cluster_data(dfper, algorithm)
    results = assign_clusters(Drug_see_p1, dfper)

    # Assign cluster median to each patient
    # If a patient doesn't have a cluster, assign the median of the most frequent cluster
    t1_cluster = results["cluster"].value_counts().idxmax()
    t1_median = results[results["cluster"] == t1_cluster]["median"].iloc[0]

    Drug_see_p1 = pd.merge(
        Drug_see_p1, results[["pnr", "median", "cluster"]], on="pnr", how="left"
    )
    Drug_see_p1["median"].fillna(t1_median, inplace=True)
    Drug_see_p1["test"] = Drug_see_p1["event.interval"] - Drug_see_p1["median"]

    Drug_see_p0 = pd.merge(
        C09CA01, Drug_see_p1[["pnr", "median", "cluster"]], on="pnr", how="left"
    )
    Drug_see_p0["median"].fillna(t1_median, inplace=True)

    return Drug_see_p0


def see_assumption(arg1: pd.DataFrame):
    arg1 = arg1.sort_values(by=["pnr", "eksd"])
    arg1["prev_eksd"] = arg1.groupby("pnr")["eksd"].shift(1)

    Drug_see2 = arg1.copy()
    Drug_see2["p_number"] = Drug_see2.groupby("pnr").cumcount() + 1
    Drug_see2 = Drug_see2[Drug_see2["p_number"] >= 2]
    Drug_see2 = Drug_see2[["pnr", "eksd", "prev_eksd", "p_number"]]
    Drug_see2["Duration"] = (Drug_see2["eksd"] - Drug_see2["prev_eksd"]).dt.days

    # Boxplot with medians
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="p_number", y="Duration", data=Drug_see2)

    medians_of_medians = Drug_see2.groupby("pnr")["Duration"].median().reset_index()
    plt.axhline(y=medians_of_medians["Duration"].median(), color="red", linestyle="--")
    plt.title("Duration by Prescription Number")
    plt.show()


"""
PROGRAM RUN
"""

tidy = import_data()

# Run kmeans
result_kmeans_a = see("medA", tidy, algorithm="kmeans")
result_kmeans_b = see("medB", tidy, algorithm="kmeans")

print("K-Means Results:")
print(result_kmeans_a.head())
print(result_kmeans_b.head())

see_assumption(result_kmeans_a)
see_assumption(result_kmeans_b)

# Run DBSCAN
result_dbscan_a = see("medA", tidy, algorithm="dbscan")
result_dbscan_b = see("medB", tidy, algorithm="dbscan")

print("\nDBSCAN Results:")
print(result_dbscan_a.head())
print(result_dbscan_b.head())

see_assumption(result_dbscan_a)
see_assumption(result_dbscan_b)
