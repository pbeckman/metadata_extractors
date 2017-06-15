import json
import csv
import os
import re
import pickle as pkl
from metadata_extractors.metadata_util import extract_metadata
from metadata_extractors.metadata_writer import make_file_list, write_metadata
from metadata_extractors.metadata_refiner import make_filesystem_graph, topic_mixture, refine_metadata


def display_metadata(file_name, path, pass_fail=False):
    print """
    ----------------------------
    {}
    ----------------------------
    """.format(file_name)
    metadata = json.dumps(extract_metadata(file_name, path, pass_fail=pass_fail),
                          sort_keys=True, indent=4, separators=(',', ': '))
    print metadata


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


def write_test_graph(graph_file_name):
    with open(graph_file_name, "wb") as graph_file:
        make_filesystem_graph("test_files", graph_file=graph_file)


def test_topic_mixture(graph_file_name, metadata_file_name):
    with open(graph_file_name, "rb") as graph_file, open(metadata_file_name, "r") as metadata_file:
        metadata = json.load(metadata_file)["files"]
        G = pkl.load(graph_file)
        print topic_mixture("test_files/no_headers.csv", metadata, G)


def write_test_metadata():
    write_metadata("test_metadata.json", "test_file_list.txt",
                   new_file_list=True, overwrite=True,
                   pass_fail=False, lda_preamble=True, null_inference=False)


def refine_test_metadata():
    refine_metadata("test_metadata.json", "test_metadata-refined.json", "test_graph.pkl",
                    lda_preamble=False, null_inference=False)


def pipeline_test():
    write_test_metadata()
    write_test_graph("test_graph.pkl")
    refine_test_metadata()

pipeline_test()

# make_test_col_csv()
# test_metadata_extraction()
# test_topic_mixture("test_graph.pkl", "test_metadata.json")
# refine_metadata("metadata_5-15.json", "metadata_5-15_refined.json", "pub8_graph.pkl",
#                 lda_preamble=False, null_inference=False)


