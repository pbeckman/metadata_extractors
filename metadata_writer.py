import os
import json
import traceback
import re
from metadata_util import extract_metadata


def write_metadata(metadata_file_name, list_file_name, new_file_list=False, overwrite=False,
                   pass_fail=False, lda_preamble=True, null_inference=False):
    """Write the metadata from a list of files to JSON.

        :param metadata_file_name: (str) name of file for writing metadata JSON
        :param list_file_name: (str) name of file that stores list of full file paths
        :param new_file_list: (bool) whether to created a new file list
        :param overwrite: (bool) whether to overwrite the existing metadata document
        :param pass_fail: (bool) whether to extract metadata using pass_fail
        :param lda_preamble: (bool) whether to record LDA-generated topics for columnar preambles
        :param null_inference: (bool) whether to perform null_inference on columnar files"""

    if new_file_list:
        with open(list_file_name, "w") as lf:
            make_file_list("test_files", lf)

    with open(list_file_name, "r") as lf:
        restart_file = 0

        if os.path.isfile(metadata_file_name) and not overwrite:
            try:
                with open("restart.txt", "r") as rf:
                    restart_file = int(rf.read().split(",")[0].strip())
                mf = open(metadata_file_name, "a")
            except ValueError:
                mf = open(metadata_file_name, "w")
                mf.write('{"files":[')
        else:
            mf = open(metadata_file_name, "w")
            mf.write('{"files":[')

        write_file_metadata(lf.readlines(), restart_file, mf, "restart.txt",
                            pass_fail=pass_fail, lda_preamble=lda_preamble, null_inference=null_inference)
        mf.seek(-2, 1)
        mf.write(']}')

    mf.close()


def write_file_metadata(files, start_number, metadata_file, restart_file_name,
                        pass_fail=False, lda_preamble=True, null_inference=False):
    """Write the metadata from a single file to JSON.

        :param files: (list(str)) list of full paths to files.
        :param start_number: (int) which file to start with, used mainly to restart
        :param metadata_file: (file) file object open for writing metadata JSON
        :param restart_file_name: (str) name of file that stores last processed file as a restart point
        :param pass_fail: (bool) whether to extract metadata using pass_fail
        :param lda_preamble: (bool) whether to record LDA-generated topics for columnar preambles
        :param null_inference: (bool) whether to perform null_inference on columnar files"""

    for file_number in range(start_number, len(files)):
        full_path = files[file_number]
        path, file_name = full_path.strip().rsplit("/", 1)
        path += "/"

        with open(restart_file_name, "w") as rf:
            rf.write(str(file_number) + ',' + full_path)

        print "extracting metadata from: {}".format(path + file_name)
        try:
            metadata = extract_metadata(file_name, path, pass_fail=pass_fail,
                                        lda_preamble=lda_preamble, null_inference=null_inference)
            metadata_file.write(json.dumps(metadata) + ",\n")
        except (OverflowError, UnicodeDecodeError, MemoryError, TypeError) as e:
            with open("errors.log", "a") as error_file:
                error_file.write(
                    "{} :: {}\n{}\n\n".format(full_path, str(e), traceback.format_exc()))


def make_file_list(dir, list_file=None):
    """Return and/or write list of full paths to files in a given directory.

        :param dir: (str) directory name. It's easiest to use the full path.
        :param list_file: (file) file object open for writing. If None, no file is written.
        :returns: (list) list of full paths"""

    all_files = []
    for root, dirs, files in os.walk(dir):
        for file_name in files:
            if file_name[0] != ".":
                all_files.append(os.path.join(root, file_name))

    if list_file is not None:
        for file_name in all_files:
            list_file.write(file_name + "\n")

    return all_files


def write_cols_to_csv(metadata_file, csv_writer, num_files=None):
    """Write all valid columns from a metadata JSON file to csv. Used to create training sets for ML models.

        :param metadata_file: (file) metadata file object opened for reading
        :param csv_writer: (csv.writer) csv writer object
        :param num_files: (int) number of files from which to write columns to csv. If None, all files used"""

    csv_writer.writerow([
        "path", "file", "column",
        "min_1", "min_diff_1", "min_2", "min_diff_1", "min_3",
        "max_1", "max_diff_1", "max_2", "max_diff_1", "max_3",
        "avg",
        "null"
    ])

    files = json.load(metadata_file)["files"]
    valid_files = []
    for item in files:
        if "columnar" in item["system"]["extractors"] \
                and not any([re.match("__\d*__", key) for key in item["columns"].keys()]):
            valid_files.append(item)

    if num_files is not None:
        to_write = [int(len(valid_files) / float(num_files) * j) for j in range(0, num_files)]
        valid_files = [valid_files[i] for i in to_write]

    for item in valid_files:
        write_dict_to_row(item, csv_writer)


def write_dict_to_row(metadata, csv_writer):
    """Write a metadata dict to csv.

        :param metadata: (dict) standard formatted metadata dictionary
        :param csv_writer: (csv.writer) csv writer object"""

    cols = metadata["columns"].keys()
    for col in cols:
        col_agg = metadata["columns"][col]
        csv_writer.writerow([
            metadata["system"]["path"], metadata["system"]["file"], col.lower(),

            col_agg["min"][0] if "min" in col_agg.keys() and len(col_agg["min"]) > 0 else None,
            col_agg["min"][1] - col_agg["min"][0] if "min" in col_agg.keys() and len(col_agg["min"]) > 1 else None,
            col_agg["min"][1] if "min" in col_agg.keys() and len(col_agg["min"]) > 1 else None,
            col_agg["min"][2] - col_agg["min"][1] if "min" in col_agg.keys() and len(col_agg["min"]) > 2 else None,
            col_agg["min"][2] if "min" in col_agg.keys() and len(col_agg["min"]) > 2 else None,

            col_agg["max"][0] if "max" in col_agg.keys() and len(col_agg["max"]) > 0 else None,
            col_agg["max"][0] - col_agg["max"][1] if "max" in col_agg.keys() and len(col_agg["max"]) > 1 else None,
            col_agg["max"][1] if "max" in col_agg.keys() and len(col_agg["max"]) > 1 else None,
            col_agg["max"][1] - col_agg["max"][2] if "max" in col_agg.keys() and len(col_agg["max"]) > 2 else None,
            col_agg["max"][2] if "max" in col_agg.keys() and len(col_agg["max"]) > 2 else None,

            col_agg["avg"] if "avg" in col_agg.keys() else None,

            None  # space for null value to be recorded by hand
        ])
