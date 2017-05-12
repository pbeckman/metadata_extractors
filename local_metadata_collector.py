import os
import json
import traceback
from metadata_util import extract_metadata


def write_file_list(dir, list_file):
    for root, dirs, files in os.walk(dir):
        for file_name in files:
            if file_name[0] != ".":
                list_file.write(os.path.join(dir, file_name) + "\n")


def write_metadata(files, start_number, metadata_file, restart_file, pass_fail=False):
    for file_number in range(start_number, len(files)):
        full_path = files[file_number]
        path, file_name = full_path.strip().rsplit("/", 1)
        path += "/"

        with open(restart_file, "w") as rf:
            rf.write(str(file_number) + ',' + full_path)

        print "extracting metadata from: {}".format(path + file_name)
        metadata = {}
        try:
            metadata = extract_metadata(file_name, path, pass_fail=pass_fail)
        except (UnicodeDecodeError, MemoryError, TypeError) as e:
            with open("errors.log", "a") as error_file:
                error_file.write(
                    "{} :: {}\n{}\n\n".format(full_path, str(e), traceback.format_exc()))

        metadata_file.write(json.dumps(metadata) + ",\n")


def write_dict_to_csv(metadata, csv_writer):
    cols = metadata["columns"].keys()
    for col in cols:
        col_agg = metadata["columns"][col]
        csv_writer.writerow([
            metadata["system"]["path"], metadata["system"]["file"], col,

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


def write_cols_to_csv(metadata_file, csv_writer):
    metadata = json.load(metadata_file)["files"]
    for item in metadata:
        if "columnar" in item["system"]["extractors"]:
            write_dict_to_csv(item, csv_writer)
