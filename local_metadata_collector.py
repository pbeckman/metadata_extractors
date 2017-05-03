import os
import json
import traceback
from metadata_util import extract_metadata


def write_file_list(dir, list_file):
    for root, dirs, files in os.walk(dir):
        for file_name in files:
            list_file.write(os.path.join(dir, file_name)+"\n")


def write_metadata(files, start_number, metadata_file, restart_file):
    for file_number in range(start_number, len(files)):
        full_path = files[file_number]
        path, file_name = full_path.strip().rsplit("/", 1)
        path += "/"

        with open(restart_file, "w") as rf:
            rf.write(str(file_number) + ',' + full_path)

        metadata = {}
        try:
            metadata = extract_metadata(file_name, path)
        except (UnicodeDecodeError, MemoryError, TypeError) as e:
            with open("errors.log", "a") as error_file:
                error_file.write(
                    "{} :: {}\n{}\n\n".format(full_path, str(e), traceback.format_exc()))

        metadata_file.write(json.dumps(metadata) + ",\n")

# with open("file_list.txt", "w") as lf:
#     write_file_list("test_files", lf)

with open("test_metadata.json", "a") as mf, open("file_list.txt", "r") as rf:
    write_metadata(rf.readlines(), 0, mf, "restart.txt")
