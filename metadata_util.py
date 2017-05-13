import json
import numpy
import os
import re
import magic
import pickle as pkl
from netCDF4 import Dataset
from decimal import Decimal
from hashlib import sha256
from heapq import nsmallest, nlargest
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk.tokenize import RegexpTokenizer

# load necessary LDA resources
dictionary = corpora.Dictionary.load("../lda_model/climate_abstracts.dict")
lda_model = LdaModel.load("../lda_model/climate_abstracts.lda")

# load null inference model
with open(os.path.abspath("../null_inference_model/ni_model.pkl")) as model_file:
    ni_model = pkl.load(model_file)
# maximum distance of value from null to still be considered null
NULL_EPSILON = 1


class ExtractionFailed(Exception):
    """Basic error to throw when an extractor fails"""


class ExtractionPassed(Exception):
    """Indicator to throw when extractor passes for fast file classification"""


def extract_metadata(file_name, path, pass_fail=False):
    """Create metadata JSON from file.

        :param file_name: (str) file name
        :param path: (str) absolute or relative path to file
        :param pass_fail: (bool) whether to exit after ascertaining file class
        :returns: (dict) metadata dictionary"""

    with open(path + file_name, 'rU') as file_handle:

        extension = file_name.split('.', 1)[1] if '.' in file_name else "no extension"
        mime = magic.Magic(mime=True)
        mime_type = mime.from_file(path + file_name)
        metadata = {
            "system": {
                "file": file_name,
                "path": path,
                "extension": extension,
                "mime_type": mime_type,
                "size": os.path.getsize(path + file_name),
                "checksum": sha256(file_handle.read()).hexdigest(),
                "extractors": []
            }
        }

        # checksum puts cursor at end of file - reset to beginning for metadata extraction
        file_handle.seek(0)

        text_frac_num = 0.2

        if "nc" in extension.lower():
            try:
                metadata.update(extract_netcdf_metadata(file_handle, pass_fail=pass_fail))
                metadata["system"]["extractors"].append("netcdf")
            except ExtractionPassed:
                metadata["system"]["extractors"].append("netcdf")
                pass
            except ExtractionFailed:
                # not a netCDF file
                pass
        elif any([i in mime_type for i in ["text", "csv", "xml"]]):
            try:
                metadata.update(extract_columnar_metadata(file_handle, pass_fail=pass_fail, null_inference=False))
                metadata["system"]["extractors"].append("columnar")
            except ExtractionPassed:
                metadata["system"]["extractors"].append("columnar")
                pass
            except ExtractionFailed:
                # not a columnar file
                # check if this file is a usable abstract-like file
                if frac_numeric(file_handle) < text_frac_num:
                    try:
                        metadata.update(extract_topic(file_handle, pass_fail=pass_fail))
                        metadata["system"]["extractors"].append("lda")
                    except ExtractionPassed:
                        metadata["system"]["extractors"].append("lda")
                    except ExtractionFailed:
                        pass

    return metadata


def extract_netcdf_metadata(file_handle, pass_fail=False):
    """Create netcdf metadata JSON from file.

        :param file_handle: (str) file
        :param pass_fail: (bool) whether to exit after ascertaining file class
        :returns: (dict) metadata dictionary"""

    try:
        dataset = Dataset(os.path.realpath(file_handle.name))
    except IOError:
        raise ExtractionFailed

    if pass_fail:
        raise ExtractionPassed

    metadata = {
        "file_format": dataset.file_format,
    }
    if len(dataset.ncattrs()) > 0:
        metadata["global_attributes"] = {}
    for attr in dataset.ncattrs():
        metadata["global_attributes"][attr] = dataset.getncattr(attr)

    dims = dataset.dimensions
    if len(dims) > 0:
        metadata["dimensions"] = {}
    for dim in dims:
        metadata["dimensions"][dim] = {
            "size": len(dataset.dimensions[dim])
        }
        add_ncattr_metadata(dataset, dim, "dimensions", metadata)

    vars = dataset.variables
    if len(vars) > 0:
        metadata["variables"] = {}
    for var in vars:
        if var not in dims:
            metadata["variables"][var] = {
                "dimensions": dataset.variables[var].dimensions,
                "size": dataset.variables[var].size
            }
        add_ncattr_metadata(dataset, var, "variables", metadata)

    # cast all numpy types to native python types via dumps, then back to dict via loads
    return json.loads(json.dumps(metadata, cls=NumpyDecoder))


def add_ncattr_metadata(dataset, name, dim_or_var, metadata):
    """Get attributes from a netCDF variable or dimension.

        :param dataset: (netCDF4.Dataset) dataset from which to extract metadata
        :param name: (str) name of attribute
        :param dim_or_var: ("dimensions" | "variables") metadata key for attribute info
        :param metadata: (dict) dictionary to add this attribute info to"""

    try:
        metadata[dim_or_var][name]["type"] = dataset.variables[name].dtype
        for attr in dataset.variables[name].ncattrs():
            metadata[dim_or_var][name][attr] = dataset.variables[name].getncattr(attr)
    # some variables have no attributes
    except KeyError:
        pass


class NumpyDecoder(json.JSONEncoder):
    """Serializer used to convert numpy types to normal json serializable types.
    Since netCDF4 produces numpy types, this is necessary for compatibility with
    other metadata scrapers like the csv, which returns a python dict"""

    def default(self, obj):
        if isinstance(obj, numpy.generic):
            return numpy.asscalar(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, numpy.dtype):
            return str(obj)
        else:
            return super(NumpyDecoder, self).default(obj)


def extract_columnar_metadata(file_handle, pass_fail=False, collect_preamble=False, null_inference=False):
    """Get metadata from column-formatted file.

            :param file_handle: (file) open file
            :param pass_fail: (bool) whether to exit after ascertaining file class
            :param collect_preamble: (bool) whether to collect the free-text preamble at the start of the file
            :param null_inference: (bool) whether to use the null inference model to remove nulls
            :returns: (dict) ascertained metadata
            :raises: (ExtractionFailed) if the file cannot be read as a columnar file"""

    try:
        return _extract_columnar_metadata(
            file_handle, ",",
            pass_fail=pass_fail, collect_preamble=collect_preamble, null_inference=null_inference
        )
    except ExtractionFailed:
        try:
            return _extract_columnar_metadata(
                file_handle, "\t",
                pass_fail=pass_fail, collect_preamble=collect_preamble, null_inference=null_inference
            )
        except ExtractionFailed:
            return _extract_columnar_metadata(
                file_handle, " ",
                pass_fail=pass_fail, collect_preamble=collect_preamble, null_inference=null_inference
            )


def _extract_columnar_metadata(file_handle, delimiter, pass_fail=False, collect_preamble=False,
                               null_inference=False, nulls=None):
    """helper method for extract_columnar_metadata that uses a specific delimiter."""

    reverse_reader = ReverseReader(file_handle, delimiter=delimiter)

    # base dictionary in which to store all the metadata
    metadata = {"columns": {}}

    # minimum number of rows to be considered an extractable table
    min_rows = 5
    # number of rows used to generate aggregates for the null inference model
    ni_rows = 100
    # number of rows to skip at the end of the file before reading
    end_rows = 5
    # size of extracted free-text preamble in characters
    preamble_size = 1000

    headers = []
    col_types = []
    col_aliases = []
    num_rows = 0
    # used to check if all rows are the same length, if not, this is not a valid columnar file
    row_length = 0
    is_first_row = True
    fully_parsed = True

    # save the last `end_rows` rows to try to parse them later
    # if there are less than `end_rows` rows, you must catch the StopIteration exception
    last_rows = []
    try:
        last_rows = [reverse_reader.next() for i in range(0, end_rows)]
    except StopIteration:
        pass

    # now we try to extract a table from the remaining n-`end_rows` rows
    for row in reverse_reader:
        # if row is not the same length as previous row, raise an error showing this is not a valid columnar file
        if not is_first_row and row_length != len(row):
            # tables are not worth extracting if under this row threshold
            if num_rows < min_rows:
                raise ExtractionFailed
            else:
                # show that extract failed before we reached the beginning of the file
                fully_parsed = False
                break
        # update row length for next check
        row_length = len(row)

        if is_first_row:
            # make column aliases so that we can create aggregates even for unlabelled columns
            col_aliases = ["__{}__".format(i) for i in range(0, row_length)]
            # type check the first row to decide which aggregates to use
            col_types = ["num" if is_number(field) else "str" for field in row]
            is_first_row = False

        # if the row is a header row, add all its fields to the headers list
        if is_header_row(row):
            # tables are likely not representative of the file if under this row threshold, don't extract metadata
            if num_rows < min_rows:
                raise ExtractionFailed
            # set the column aliases to the most recent header row if they are unique
            # we do this because the most accurate headers usually appear first in the file after the preamble
            if len(set(row)) == len(row):
                for i in range(0, len(row)):
                    metadata["columns"][row[i]] = metadata["columns"].pop(col_aliases[i])
                col_aliases = row

            for header in row:
                if header != "":
                    headers.append(header)

        else:  # is a row of values
            num_rows += 1
            if not pass_fail:
                add_row_to_aggregates(metadata, row, col_aliases, col_types, nulls=nulls)

        if pass_fail and num_rows >= min_rows:
            raise ExtractionPassed

        # we've taken enough rows to use aggregates for null inference
        if null_inference and num_rows >= ni_rows:
            add_final_aggregates(metadata, col_aliases, col_types, num_rows)
            return _extract_columnar_metadata(file_handle, delimiter, pass_fail=pass_fail,
                                              collect_preamble=collect_preamble,
                                              null_inference=False,
                                              nulls=inferred_nulls(metadata))

    # extraction passed but there are too few rows
    if num_rows < min_rows:
        raise ExtractionFailed

    # add the originally skipped rows into the aggregates
    for row in last_rows:
        if len(row) == row_length:
            add_row_to_aggregates(metadata, row, col_aliases, col_types, nulls=nulls)

    add_final_aggregates(metadata, col_aliases, col_types, num_rows)

    # add header list to metadata
    if len(headers) > 0:
        metadata["headers"] = list(set(headers))

    # we've parsed the whole table, now do null inference
    if null_inference:
        return _extract_columnar_metadata(file_handle, delimiter, pass_fail=pass_fail,
                                          collect_preamble=collect_preamble,
                                          null_inference=False,
                                          nulls=inferred_nulls(metadata))

    # extract free-text preamble, which may contain headers
    if collect_preamble and not fully_parsed:
        # number of characters in file before last un-parse-able row
        file_handle.seek(reverse_reader.prev_position)
        remaining_chars = file_handle.tell() - 1
        # go to start of preamble
        if remaining_chars >= preamble_size:
            file_handle.seek(-preamble_size, 1)
        else:
            file_handle.seek(0)
        preamble = ""
        # do this `<=` method instead of passing a numerical length argument to read()
        # in order to avoid multi-byte character encoding difficulties
        while file_handle.tell() <= reverse_reader.prev_position:
            preamble += file_handle.read(1)
        # add preamble to the metadata
        if len(preamble) > 0:
            metadata["preamble"] = preamble

    # remove empty string aggregates that were placeholders in null inference
    for key in metadata["columns"].keys():
        if metadata["columns"][key] == {}:
            metadata["columns"].pop(key)

    return metadata


def add_row_to_aggregates(metadata, row, col_aliases, col_types, nulls=None):
    """Adds row data to aggregates.

        :param metadata: (dict) metadata dictionary to add to
        :param row: (list(str)) row of strings to add
        :param col_aliases: (list(str)) list of headers
        :param nulls: (list(num)) list giving the null value to avoid for each column
        :param col_types: (list("num" | "str")) list of header types"""

    for i in range(0, len(row)):
        value = row[i]
        col_alias = col_aliases[i]
        col_type = col_types[i]
        is_first_row = col_alias not in metadata["columns"].keys()

        if col_type == "num":
            # start off the metadata if this is the first row of values
            if is_first_row:
                metadata["columns"][col_alias] = {
                    "min": [float("inf"), float("inf"), float("inf")],
                    "max": [None, None, None],
                    "total": 0
                }
            # cast the field to a number to do numerical aggregates
            # the try except is used to pass over textual and blank space nulls on which type coercion will fail
            try:
                value = float(value)
                if float(value) == int(value):
                    value = int(value)
            except ValueError:
                # skips adding to aggregates
                continue

            if nulls is not None:
                null = nulls[i]
                # if value is (close enough to) null, don't add it to the aggregates
                # 0 is returned by the model if there is no null value
                if null != 0 and abs(value - null) < NULL_EPSILON:
                    continue

            # add row data to existing aggregates
            mins = list(set(metadata["columns"][col_alias]["min"] + [value]))
            maxes = list(set(metadata["columns"][col_alias]["max"] + [value]))
            metadata["columns"][col_alias]["min"] = nsmallest(3, mins)
            metadata["columns"][col_alias]["max"] = nlargest(3, maxes)
            metadata["columns"][col_alias]["total"] += value

        elif col_type == "str":
            if is_first_row:
                metadata["columns"][col_alias] = {}
            # TODO: add string-specific field aggregates?
            pass


def add_final_aggregates(metadata, col_aliases, col_types, num_rows):
    """Adds row data to aggregates.

        :param metadata: (dict) metadata dictionary to add to
        :param col_aliases: (list(str)) list of headers
        :param col_types: (list("num" | "str")) list of header types
        :param num_rows: (int) number of value rows"""

    # calculate averages for numerical columns if aggregates were taken,
    # (which only happens when there is a single row of headers)
    for i in range(0, len(col_aliases)):
        col_alias = col_aliases[i]

        # if metadata["columns"][col_alias] == {}:
        #     metadata["columns"].pop(col_alias)

        if col_types[i] == "num":
            metadata["columns"][col_alias]["max"] = [val for val in metadata["columns"][col_alias]["max"]
                                                     if val is not None]
            metadata["columns"][col_alias]["min"] = [val for val in metadata["columns"][col_alias]["min"]
                                                     if val != float("inf")]

            metadata["columns"][col_alias]["avg"] = round(
                float(metadata["columns"][col_alias]["total"]) / num_rows,
                max_precision(metadata["columns"][col_alias]["min"] + metadata["columns"][col_alias]["max"])
            ) if len(metadata["columns"][col_alias]["min"]) > 0 else None
            metadata["columns"][col_alias].pop("total")


def max_precision(nums):
    """Determine the maximum precision of a list of floating point numbers.

        :param nums: (list(float)) list of numbers
        :return: (int) number of decimal places precision"""
    return max([abs(Decimal(str(num)).as_tuple().exponent) for num in nums])


class ReverseReader:
    """Reads column-formatted files in reverse as lists of fields.

        :param file_handle: (file) open file
        :param delimiter: (string) delimiting character """

    def __init__(self, file_handle, delimiter=","):
        self.fh = file_handle
        self.fh.seek(0, os.SEEK_END)
        self.delimiter = delimiter
        self.position = self.fh.tell()
        self.prev_position = self.fh.tell()

    @staticmethod
    def fields(line, delim):
        # if space-delimited, do not keep whitespace fields, otherwise do
        fields = [field.strip() for field in re.split(delim if delim != " " else "\\s", line)]
        if delim in [" ", "\t", "\n"]:
            fields = filter(lambda f: f != "", fields)
        return fields

    def next(self):
        line = ''
        if self.position <= 0:
            raise StopIteration
        self.prev_position = self.position
        while self.position >= 0:
            self.fh.seek(self.position)
            next_char = self.fh.read(1)
            if next_char in ['\n', '\r']:
                self.position -= 1
                if len(line) > 1:
                    return self.fields(line[::-1], self.delimiter)
            else:
                line += next_char
                self.position -= 1
        return self.fields(line[::-1], self.delimiter)

    def __iter__(self):
        return self


def is_header_row(row):
    """Determine if row is a header row by checking that it contains no fields that are
    only numeric.

        :param row: (list(str)) list of fields in row
        :returns: (bool) whether row is a header row"""

    for field in row:
        if is_number(field):
            return False
    return True


def is_number(field):
    """Determine if a string is a number by attempting to cast to it a float.

        :param field: (str) field
        :returns: (bool) whether field can be cast to a number"""

    try:
        float(field)
        return True
    except ValueError:
        return False


def inferred_nulls(metadata):
    """Infer the null value of each column given aggregates.

        :param metadata: (dict) metadata dictionary containing aggregates
        :returns: (list(num)) a list containing the null value for each column"""

    return ni_model.predict(ni_data(metadata))


def ni_data(metadata):
    """Format metadata into a 2D array so that it can be input to the null inference model.
    Columns are:
    [
        "min_1", "min_diff_1", "min_2", "min_diff_1", "min_3",
        "max_1", "max_diff_1", "max_2", "max_diff_1", "max_3",
        "avg"
    ]

        :param metadata: (dict) metadata dictionary containing aggregates
        :returns: (list(list(num))) a 2D array of data"""

    data = [
        [
            col_agg["min"][0] if "min" in col_agg.keys() and len(col_agg["min"]) > 0 else 0,
            col_agg["min"][1] - col_agg["min"][0] if "min" in col_agg.keys() and len(col_agg["min"]) > 1 else 0,
            col_agg["min"][1] if "min" in col_agg.keys() and len(col_agg["min"]) > 1 else 0,
            col_agg["min"][2] - col_agg["min"][1] if "min" in col_agg.keys() and len(col_agg["min"]) > 2 else 0,
            col_agg["min"][2] if "min" in col_agg.keys() and len(col_agg["min"]) > 2 else 0,

            col_agg["max"][0] if "max" in col_agg.keys() and len(col_agg["max"]) > 0 else 0,
            col_agg["max"][0] - col_agg["max"][1] if "max" in col_agg.keys() and len(col_agg["max"]) > 1 else 0,
            col_agg["max"][1] if "max" in col_agg.keys() and len(col_agg["max"]) > 1 else 0,
            col_agg["max"][1] - col_agg["max"][2] if "max" in col_agg.keys() and len(col_agg["max"]) > 2 else 0,
            col_agg["max"][2] if "max" in col_agg.keys() and len(col_agg["max"]) > 2 else 0,

            col_agg["avg"] if "avg" in col_agg.keys() else 0,
        ]
        for col_alias, col_agg in metadata["columns"].iteritems()]

    return data


def extract_topic(file_handle, pass_fail=False):
    """Create free-text metadata JSON from file indicating topic
    and some human-readable indication of its content.

        :param file_handle: (str) file
        :param pass_fail: (bool) whether to exit after ascertaining file class
        :returns: (dict) metadata dictionary"""

    tokenizer = RegexpTokenizer(r'[a-zA-Z]{3,}')
    tag_remover = re.compile('<.+>')

    doc = re.sub(tag_remover, '', file_handle.read())
    doc = tokenizer.tokenize(doc)

    # if the doc is an empty list, it clearly can't be topic modeled
    if not doc:
        raise ExtractionFailed
    elif pass_fail:
        raise ExtractionPassed

    doc_bow = dictionary.doc2bow(doc)

    topics = lda_model[doc_bow]
    # if no words are common to the training corpus, topics will be an empty list
    if not topics:
        raise ExtractionFailed
    max_topic = max(topics, key=lambda (i, p): p)[0]
    topic_words = [str(w[0]) for w in LdaModel.show_topic(lda_model, max_topic)]

    metadata = {
        "topics": topics,
        "tags": topic_words
    }

    return metadata


def frac_numeric(file_handle, sample_length=1000):
    """Determine the fraction of characters that are numeric in a sample of the file.

        :param file_handle: (file) open file object
        :param sample_length: (int) length in bytes of sample to be read from start of file
        :returns: (float) portion numeric characters"""

    # TODO: test heuristic sample_length

    # read in a sample of the file
    file_handle.seek(0)
    sample = file_handle.read(sample_length)

    return float(len(re.sub("[^0-9]", "", sample))) / len(sample)
