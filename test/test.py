import json
from metadata_extractors.metadata_util import extract_metadata, extract_topic


def display_metadata(file_name, path, pass_fail=False):
    print """
    ----------------------------
    {}
    ----------------------------
    """.format(path + file_name)
    print json.dumps(extract_metadata(file_name, path, pass_fail), sort_keys=True, indent=4, separators=(',', ': '))


def test_metadata_extraction(pass_fail=False):
    for f in [
        "no_headers.csv",
        "some_netcdf.nc",
        "single_header.csv",
        "readme.txt",
        "multiple_headers.csv",
        "single_header.txt",
        "preamble.exc.csv",
        "preamble.dat",
        "preamble.c32",
        "structured.xml",
        "test.pdf",
        "test.zip",
        "excel.xls",
        "image.jpg"
    ]:
        display_metadata(f, "test_files/", pass_fail=pass_fail)
        raw_input()

# test_metadata_extraction(pass_fail=False)

with open("test_files/readme.txt", "r") as f:
    print extract_topic(f)
