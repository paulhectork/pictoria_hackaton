"""
input contains images and a metadata table with columns `path,label`.
create a folder for each label and move each image to the propoer label-folder
"""

import pandas as pd
import typing as t
import argparse
import shutil
import os

from tqdm import tqdm

# ***********************************************************************

ROOT = os.path.abspath(os.path.dirname(__file__))
DATASET = os.path.abspath(os.path.join(ROOT, "data", "dataset"))

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


def make_dirs(df=pd.DataFrame) -> None:
    """create 1 dir in `DATASET` for each dir in `DF`"""
    dir_list = df["label"].drop_duplicates().to_list()  # type:ignore

    for d in dir_list:
        # `makedirs` is creates all directories in the path so we don't need to create DATASET
        if not os.path.isdir(os.path.join(DATASET, d)):
            os.makedirs(os.path.join(DATASET, d))
    return


def copy_files(df=pd.DataFrame) -> None:
    """copy the files to the DATASET"""
    # create output path
    df["filename"] = df.path.apply(lambda p: os.path.basename(p))  # type:ignore
    df["output_path"] = df.apply(lambda row: os.path.join(DATASET, row.label, row.filename), axis=1)  # type:ignore

    # move files to output
    mover = lambda infile, outfile: shutil.copy2(infile, outfile)
    tqdm.pandas(desc="copying files to DATASET")
    df.progress_apply(lambda row: mover(row.path, row.output_path), axis=1)

    return

# ***********************************************************************


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", choices=["test", "normal"], required=True)
    args = parser.parse_args()

    if args.mode == "test":
        df = make_dummy_input_dataset()
    else:
        df = None  # todo

    # process
    if os.path.exists(DATASET):
        shutil.rmtree(DATASET)
    make_dirs(df)  # type:ignore
    copy_files(df)  # type:ignore



