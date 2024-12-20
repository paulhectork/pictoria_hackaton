"""
generate a dataset as expected by keras
(1 root folder / 1 subfolder per label / all files for this label)

input contains images and a metadata table with columns `path,label`.
create a folder for each label and move each image to the propoer label-folder

this cli has 2 modes: `dummy` and `real`

- dummy mode:
    at first, we did not yet have the images and metadata table, so we generated
    fake files with fake classes, build a metadata table from that, and then moved
    images to fit the structure expected by Keras.
- real mode:
    we have real data (metadata tables and images) and so we build the datasets from
    real data.
"""

import pandas as pd
import typing as t
import argparse
import shutil
import os

from tqdm import tqdm

# ***********************************************************************

ROOT = os.path.abspath(os.path.dirname(__file__))

# ***********************************************************************

def make_dummy_input_dataset():
    """
    create a dummy input dataset to test the process
    """
    import random
    import string
    from uuid import uuid4

    # HELPERS
    # write a fake file to `fp`
    def dummy_file_writer(fp):
        letters = string.ascii_lowercase
        contents = "".join(random.choice(letters) for i in range(5000))
        with open(fp, mode="w") as fh:
            fh.write(contents)
    # input folder name: data/reorder_dummy_dataset_in/
    dummy_dataset_name = "reorder_dummy_dataset_in"
    dummy_dataset_path = os.path.join(ROOT, "data", dummy_dataset_name)
    # generates a path to subfolders
    dummy_dataset_subfolder = (
        lambda dir: os.path.join(dummy_dataset_path, dir))
    # generate a path to a single file
    dummy_dataset_file = (
        lambda dir, fn: os.path.join(dummy_dataset_subfolder(dir), fn))
    # 5 fake directories to add files to
    source_dirs = [ str(uuid4()) for i in range(5) ]

    # WRITE THE FILES
    # delete the input if it exists
    # create 500 files with random text in `dummy_dataset_name`
    if os.path.exists(dummy_dataset_path):
        shutil.rmtree(dummy_dataset_path)
    for d in source_dirs:
        subfolder = dummy_dataset_subfolder(d)
        if not os.path.isdir(subfolder):
            os.makedirs(subfolder)
        for i in range(500):
            filepath = dummy_dataset_file(d, f"{str(uuid4())}.txt")
            dummy_file_writer(filepath)

    # BUILD THE DATAFRAME
    filepaths = []  # relative path from `os.curdir` to each dummy file created
    for (root, dir, files) in os.walk(dummy_dataset_path):
        for f in files:
            filepaths.append(os.path.join(root, f))
    # { className: weight }
    classes = { "labelA": 0.3, "labelB": 0.2, "labelC": 0.1, "labelD": 0.4 }
    # { className: [1st filename with className,last filename with className] }
    classes_range = {}
    step = 0
    for classname, weight in classes.items():
        step_start = step #step_start = step + 1
        step += int( round(len(filepaths) * weight, 0) )
        classes_range[classname] = [step_start, step]

    match_class = lambda idx: [ classname
                                for classname, range_ in classes_range.items()
                                if idx < range_[1] and idx >= range_[0] ]
    # build the dataframe
    df = pd.DataFrame.from_records([ (f, match_class(i))
                                     for i, f in enumerate(filepaths) ]
                                   )
    df = df.rename(columns={ 0: "path", 1: "label" })
    df.label = df.label.apply(lambda x: x[0])

    return df


def make_dirs(df:pd.DataFrame, dataset_path) -> None:
    """create 1 dir in `DATASET` for each dir in `DF`"""
    dir_list = df["label"].drop_duplicates().to_list()  # type:ignore

    for d in dir_list:
        # `makedirs` is creates all directories in the path so we don't need to create the dataset
        if not os.path.isdir(os.path.join(dataset_path, d)):
            os.makedirs(os.path.join(dataset_path, d))
    return


def copy_files(df:pd.DataFrame, dataset_path) -> None:
    """copy the files to the `dataset_path` (full path to dataset directory)"""
    # create output path
    df["filename"] = df.path.apply(lambda p: os.path.basename(p))  # type:ignore
    df["output_path"] = df.apply(lambda row: os.path.join(dataset_path, row.label, row.filename), axis=1)  # type:ignore

    # move files to output
    mover = lambda infile, outfile: shutil.copy2(infile, outfile)
    tqdm.pandas(desc="copying files to dataset")
    df.progress_apply(lambda row: mover(row.path, row.output_path), axis=1)  # type:ignore

    return


def process(df, output_dataset_name:str):
    # process
    output_dataset_path = os.path.join(ROOT, "data", output_dataset_name)
    if os.path.exists(output_dataset_path):
        shutil.rmtree(output_dataset_path)
    make_dirs(df, output_dataset_name)  # type:ignore
    copy_files(df, output_dataset_name)  # type:ignore


# ***********************************************************************


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate a dataset as expected by keras')
    parser.add_argument("-m", "--mode", choices=["dummy", "real"], required=True, help="if `dummy`, generate fake data to test everything works. if `real`, work on real data")
    args = parser.parse_args()

    if args.mode == "test":
        df = make_dummy_input_dataset()
        dataset_name = "dataset_dummy"
        process(df, dataset_name)

    else:
        # you might need to tweak your filepaths
        df_train = pd.read_csv(os.path.join(ROOT, "data", "data_entrainement_final.csv"), sep=";")
        df_train = df_train.rename(columns={ "path": "path", "Type de document[multi_tags]": "label" })
        df_train = df_train.loc[df_train.label.notna()]
        dataset_train_name = "dataset_train"
        process(df_train, dataset_train_name)

        df_valid = pd.read_csv(os.path.join(ROOT, "data", "data_test_final.csv"), sep=",")
        df_valid = df_valid.rename(columns={ "path": "path", "type de document[tag]": "label" })
        print(df_valid.columns)
        df_valid = df_valid.loc[df_valid.label.notna()]
        dataset_valid_name = "dataset_valid"
        process(df_valid, dataset_valid_name)




