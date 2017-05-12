import json
import csv
import os
from metadata_extractors.metadata_util import extract_metadata
from metadata_extractors.local_metadata_collector import write_file_list, write_metadata, write_cols_to_csv


def display_metadata(file_name, path, pass_fail=False):
    print """
    ----------------------------
    {}
    ----------------------------
    """.format(file_name)
    print json.dumps(extract_metadata(file_name, path, pass_fail), sort_keys=True, indent=4, separators=(',', ': '))


def test_metadata_extraction(pass_fail=False):
    for f in [
        # "no_headers.csv",
        # "some_netcdf.nc",
        # "single_header.csv",
        # "multiple_headers.csv",
        # "single_header.txt",
        "preamble.exc.csv",
        # "preamble.dat",
        # "preamble.c32",
        # "test.pdf",
        # "test.zip",
        # "excel.xls",
        # "image.jpg",
        # "structured.xml",
        # "readme.txt",
        # "readme2.txt"
    ]:
        display_metadata(f, "test_files/", pass_fail=pass_fail)
        raw_input()


def write_test_metadata():
    with open("file_list.txt", "w") as lf:
        write_file_list("/home/tskluzac/pub8", lf)

    with open("file_list.txt", "r") as rf:

        if os.path.isfile("test_metadata.json"):
            mf = open("metadata_5-12.json", "a")
        else:
            mf = open("metadata_5-12.json", "w")
            mf.write('{"files":[')

        write_metadata(rf.readlines(), 0, mf, "restart.txt", pass_fail=False)
        mf.seek(-1, 1)
        mf.write(']}')

    mf.close()


def make_test_col_csv():
    csv_writer = csv.writer(open("test_cols.csv", "w"))
    csv_writer.writerow([
        "path", "file", "column",
        "min_1", "min_diff_1", "min_2", "min_diff_1", "min_3",
        "max_1", "max_diff_1", "max_2", "max_diff_1", "max_3",
        "avg",
        "null"
    ])
    with open("test_metadata.json", "r") as mf:
        write_cols_to_csv(mf, csv_writer)

write_test_metadata()
# make_test_col_csv()
# test_metadata_extraction()


