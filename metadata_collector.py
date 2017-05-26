import os
import json
import traceback
from metadata_util import extract_metadata


def make_file_list(dir, list_file=None):
    all_files = []
    for root, dirs, files in os.walk(dir):
        for file_name in files:
            if file_name[0] != ".":
                all_files.append(os.path.join(root, file_name))

    if list_file is not None:
        for file_name in all_files:
            list_file.write(file_name + "\n")

    return all_files


def write_metadata(files, start_number, metadata_file, restart_file, pass_fail=False):
    for file_number in range(start_number, len(files)):
        full_path = files[file_number]
        path, file_name = full_path.strip().rsplit("/", 1)
        path += "/"

        with open(restart_file, "w") as rf:
            rf.write(str(file_number) + ',' + full_path)

        print "extracting metadata from: {}".format(path + file_name)
        try:
            metadata = extract_metadata(file_name, path, pass_fail=pass_fail)
            metadata_file.write(json.dumps(metadata) + ",\n")
        except (OverflowError, UnicodeDecodeError, MemoryError, TypeError) as e:
            with open("errors.log", "a") as error_file:
                error_file.write(
                    "{} :: {}\n{}\n\n".format(full_path, str(e), traceback.format_exc()))
