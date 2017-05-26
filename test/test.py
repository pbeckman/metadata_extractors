import json
import csv
import os
import re
import pickle as pkl
from metadata_extractors.metadata_util import extract_metadata
from metadata_extractors.metadata_collector import make_file_list, write_metadata
from metadata_extractors.metadata_refiner import make_filesystem_graph, topic_mixture


def display_metadata(file_name, path, pass_fail=False):
    print """
    ----------------------------
    {}
    ----------------------------
    """.format(file_name)
    print json.dumps(extract_metadata(file_name, path, pass_fail=pass_fail), sort_keys=True, indent=4, separators=(',', ': '))


def test_metadata_extraction(pass_fail=False):
    for f in [
        "no_headers.csv",
        "some_netcdf.nc",
        "single_header.csv",
        "multiple_headers.csv",
        "single_header.txt",
        "preamble.exc.csv",
        "preamble.dat",
        "preamble.c32",
        "test.pdf",
        "test.zip",
        "excel.xls",
        "image.jpg",
        "structured.xml",
        "readme.txt",
        "readme2.txt"
    ]:
        display_metadata(f, "test_files/", pass_fail=pass_fail)
        raw_input()


def write_test_metadata(metadata_file_name, new_file_list=False, overwrite=False):
    if new_file_list:
        with open("test_file_list.txt", "w") as lf:
            make_file_list("test_files", lf)

    with open("test_file_list.txt", "r") as lf:
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

        write_metadata(lf.readlines(), restart_file, mf, "restart.txt", pass_fail=False)
        mf.seek(-2, 1)
        mf.write(']}')

    mf.close()


def write_test_graph(graph_file_name):
    with open(graph_file_name, "wb") as graph_file:
        make_filesystem_graph("test_files", graph_file=graph_file)


def test_topic_mixture(graph_file_name, metadata_file_name):
    with open(graph_file_name, "rb") as graph_file, open(metadata_file_name, "r") as metadata_file:
        metadata = json.load(metadata_file)["files"]
        G = pkl.load(graph_file)
        print topic_mixture("test_files/no_headers.csv", metadata, G)


def write_dict_to_csv(metadata, csv_writer):
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


def write_cols_to_csv(metadata_file, csv_writer, num_rows=None):
    files = json.load(metadata_file)["files"]
    valid_files = []
    for item in files:
        if "columnar" in item["system"]["extractors"] \
                and not any([re.match("__\d*__", key) for key in item["columns"].keys()]):
            valid_files.append(item)

    if num_rows is not None:
        to_write = [int(len(valid_files)/float(num_rows)*j) for j in range(0, num_rows)]
        valid_files = [valid_files[i] for i in to_write]

    for item in valid_files:
        write_dict_to_csv(item, csv_writer)


def make_test_col_csv(num_rows=None):
    csv_writer = csv.writer(open("../ML_models/header_inference_model/header_training_data.csv", "w"))
    csv_writer.writerow([
        "path", "file", "column",
        "min_1", "min_diff_1", "min_2", "min_diff_1", "min_3",
        "max_1", "max_diff_1", "max_2", "max_diff_1", "max_3",
        "avg",
        # "null"
    ])
    with open("metadata_5-15.json", "r") as mf:
        write_cols_to_csv(mf, csv_writer, num_rows=num_rows)

write_test_metadata("test_metadata.json", new_file_list=True, overwrite=True)
# write_test_graph("test_graph.pkl")
# make_test_col_csv()
# test_metadata_extraction()
# test_topic_mixture("test_graph.pkl", "test_metadata.json")


