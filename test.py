import json
from metadata_util import extract_metadata


def display_metadata(file_name, path, pass_fail=False):
    print """
    ----------------------------
    {}
    ----------------------------
    """.format(path + file_name)
    print json.dumps(extract_metadata(file_name, path, pass_fail), sort_keys=True, indent=4, separators=(',', ': '))


def test_metadata_extraction(pass_fail=False):
    display_metadata("no_headers.csv", "test_files/", pass_fail=pass_fail)
    # display_metadata("some_netcdf.nc", "test_files/", pass_fail=pass_fail)
    # display_metadata("single_header.csv", "test_files/", pass_fail=pass_fail)
    # display_metadata("readme.txt", "test_files/", pass_fail=pass_fail)
    # display_metadata("multiple_headers.csv", "test_files/", pass_fail=pass_fail)
    # display_metadata("single_header.txt", "test_files/", pass_fail=pass_fail)
    # display_metadata("preamble.exc.csv", "test_files/", pass_fail=pass_fail)
    # display_metadata("preamble.dat", "test_files/", pass_fail=pass_fail)
    # display_metadata("preamble.c32", "test_files/", pass_fail=pass_fail)

test_metadata_extraction()