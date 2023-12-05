# Extract the results of all runs for a specified dataset and compile them into one file / dataframe.

# - iterate over all folders in the specified path
# - for each folder, obtain data just like in the .csv, and put it along the "3rd" dimension of a pandas dataframe
# - save the pandas dataframe as a pickle file
# - save the pandas dataframe as an .xlsx

import os
import argparse
from termcolor import colored
from natsort import natsorted, natsort_keygen
from timeit import default_timer as timer
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path
import concurrent.futures

def write_xlsx_summary_from_df(df, path):
    df.index.name = "run"
    df1 = df.groupby("run", sort=False).max()
    df_best_val_acc_mic = df.sort_values(['run', 'Validation/accuracy-micro'], ascending=False).groupby(['run']).first().sort_values(
            by="run", key=natsort_keygen())
    df_best_val_acc_mac = df.sort_values(['run', 'Validation/accuracy-macro'], ascending=False).groupby(['run']).first().sort_values(
            by="run", key=natsort_keygen())
    df_best_val_jacc_mac = df.sort_values(['run', 'Validation/jaccard'], ascending=False).groupby(['run']).first().sort_values(
            by="run", key=natsort_keygen())
    df_best_val_f1_mac = df.sort_values(['run', 'Validation/f1-macro'], ascending=False).groupby(['run']).first().sort_values(
            by="run", key=natsort_keygen())
    with pd.ExcelWriter(path) as writer:
        df1.to_excel(writer, index=True, sheet_name="max_values")
        df_best_val_jacc_mac.to_excel(writer, index=True, sheet_name="best_val_epochs_macro_jaccard")
        df_best_val_acc_mac.to_excel(writer, index=True, sheet_name="best_val_epochs_macro_acc")
        df_best_val_f1_mac.to_excel(writer, index=True, sheet_name="best_val_epochs_macro_f1")
        df_best_val_acc_mic.to_excel(writer, index=True, sheet_name="best_val_epochs_micro_acc")
        for index in natsorted(set(df.index)):
            df[df.index == index].to_excel(writer, index=False, sheet_name=index)


def tflog_to_pandas(path):
    path = path
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            if tag in ["hp_metric","train_loss_step"]:
                continue
            if "grad_2.0_norm" in tag:
                continue
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])

    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()

    # additionally add the class metrics for each epoch, from the class folder:
    # obviously really naive way of parsing the data...
    path+="/class-metrics/"
    epoch_nrs = []
    metrics = []
    for p in Path(path).glob("*.csv"):
        filename = p.name.split(".")[0]
        l = filename.split("_")
        # metric, train/val, epochXXX
        epoch_str = l[2].rstrip('0123456789')
        epoch_nr = l[2][len(epoch_str):]
        epoch_nrs.append(int(epoch_nr))
        metrics.append(l[0])
    metrics = set(metrics) # remove duplicates

    # TODO all of the following is hacky
    epochs = max(epoch_nrs) - min(epoch_nrs) + 1
    step_size_per_epoch = runlog_data[runlog_data.metric == "Train/jaccard"].iloc[0]["step"] # really ugly way to get the step size
    for s in ["train","val"]:
        for m in (metrics):
            for e in range(epochs):
                vs = list(np.genfromtxt(path+m+"_"+s+"_epoch"+str(e)+".csv"))
                for idx,v in enumerate(vs):
                    if s == "train":
                        metric_name = "Train/"+m
                    elif s == "val":
                        metric_name = "Validation/"+m
                    step = e * (step_size_per_epoch+1) + step_size_per_epoch
                    r = {"metric": [(metric_name+"-"+str(idx))], "value": v, "step": step}
                    r = pd.DataFrame(r)
                    runlog_data = pd.concat([runlog_data, r])

    df = pd.pivot_table(runlog_data, values="value", columns="metric", index="step", sort=False)
    # for some reason without ignoring the epoch tag there are duplicate entries with NaN values,
    # so for now just remove those
    df = df.dropna(axis=0, how="any")
    print(df)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tool to extract the metrics of multiple training runs for one dataset and compile them into one data frame.')
    parser.add_argument('-p','--path', help='Path to the folder containing the runs for one dataset (e.g. lightning_logs dir).', required=True)
    parser.add_argument('-n','--name', help='Name for the output files, e.g. the name of the dataset used.', required=True)
    parser.add_argument('-o','--output_path', help='Path to folder for output files', required=True)
    parser.add_argument('-l','--load_pkl_path', help='Path to an existing pickled DF for this dataset to use as cache; to avoid reloading runs already existing in that DF.', required=False)
    args = vars(parser.parse_args())

    # TODO convert all of the paths into proper paths to avoid broken paths

    path = args['path']
    load_pkl_path = args['load_pkl_path']
    dfs = []
    folders = natsorted(os.listdir(path))

    # load saved pickle and list existing versions
    loaded_run_names = set()
    if load_pkl_path is not None:
        print(f"Loading pickled DF as cache: {load_pkl_path}")
        loaded_df = pd.read_pickle(load_pkl_path)
        loaded_run_names = natsorted(set(loaded_df.index))
        print(f"Loaded {len(loaded_run_names)} runs:")
        print(*loaded_run_names, sep="\n")

    # TODO better way of checking if run wasn't empty
    run_paths = [path+"/"+r for r in folders if os.path.isfile(path+"/"+r+"/class-metrics/jaccard-class_val_epoch1.csv")]
    # filter out paths based on run names which are already in loaded_run_names
    run_paths = [p for p in run_paths if os.path.basename(os.path.normpath(p)) not in loaded_run_names]
    print(f"Found {len(folders)} new run folders ({len(folders)-len(run_paths)} are empty or already loaded):")
    run_names = [os.path.basename(os.path.normpath(p)) for p in run_paths]
    print(*run_paths, sep="\n")
    if not run_paths:
        if load_pkl_path is not None:
            print(f"All runs are already in loaded pkl, exiting...")
            quit()
        else:
            print(f"No runs in specified folder, exiting...")
            quit()

    print("Extracting runs, may take a few minutes...")
    start = timer()
    with concurrent.futures.ProcessPoolExecutor() as ex:
        for _, df in zip(run_paths, ex.map(tflog_to_pandas, run_paths)):
            dfs.append(df)
    df = pd.concat(dfs, keys=run_names)
    df = df.reset_index(level=[1])
    if load_pkl_path is not None:
        df = pd.concat([loaded_df, df])
    end = timer()
    print(df)
    print(f"...took {end-start} seconds.")

    pickle_path = args['output_path']+"/"+args['name']+'_results_DF.pkl'
    xlsx_path = args['output_path']+"/"+args['name']+'_results_summary.xlsx'
    print(colored(f"Writing DataFrame summary to .xlsx: {xlsx_path}", "green"))
    write_xlsx_summary_from_df(df, xlsx_path)
    print(colored(f"Writing pickled DataFrame to: {pickle_path}", "green"))
    df.to_pickle(pickle_path)



