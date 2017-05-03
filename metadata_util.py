import json
import numpy
import os
import re
import magic
from netCDF4 import Dataset
from decimal import Decimal
from hashlib import sha256
from heapq import nsmallest, nlargest
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk.tokenize import RegexpTokenizer


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
                "checksum": sha256(file_handle.read()).hexdigest()
            }
        }

        # checksum puts cursor at end of file - reset to beginning for metadata extraction
        file_handle.seek(0)

        text_frac_num = 0.1

        if extension == "nc":
            try:
                metadata.update(extract_netcdf_metadata(file_handle, pass_fail=pass_fail))
            except ExtractionPassed:
                pass
            except ExtractionFailed:
                # not a netCDF file
                pass
        elif any([i in mime_type for i in ["text", "csv", "xml"]]):
            try:
                metadata.update(extract_columnar_metadata(file_handle, pass_fail=pass_fail))
            except ExtractionPassed:
                pass
            except ExtractionFailed:
                # not a columnar file
                # check if this file is a usable abstract-like file
                if frac_numeric(file_handle) < text_frac_num:
                    # extract topic
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


def extract_columnar_metadata(file_handle, pass_fail=False):
    """Get metadata from column-formatted file.

            :param file_handle: (file) open file
            :param pass_fail: (bool) whether to exit after ascertaining file class
            :returns: (dict) ascertained metadata
            :raises: (ExtractionFailed) if the file cannot be read as a columnar file"""

    try:
        return _extract_columnar_metadata(file_handle, ",", pass_fail=pass_fail)
    except ExtractionFailed:
        try:
            return _extract_columnar_metadata(file_handle, "\t", pass_fail=pass_fail)
        except ExtractionFailed:
            return _extract_columnar_metadata(file_handle, " ", pass_fail=pass_fail)


def _extract_columnar_metadata(file_handle, delimiter, pass_fail=False):

    # choose csv.reader parameters based on file type - if not csv, use whitespace-delimited
    reverse_reader = ReverseReader(file_handle, delimiter=delimiter)

    # base dictionary in which to store all the metadata
    metadata = {"columns": {}}

    # minimum number of rows to be considered an extractable table
    min_rows = 5
    # number of rows to skip at the end of the file before reading
    end_rows = 5
    # size of extracted free-text preamble in characters
    preamble_size = 0

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
            # TODO: consider more comprehensive type checking (textual nulls in first row)
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
                add_row_to_aggregates(metadata, row, col_aliases, col_types)

        if pass_fail and num_rows >= min_rows:
            raise ExtractionPassed

    # extraction passed but there are too few rows
    if num_rows < min_rows:
        raise ExtractionFailed

    # add the originally skipped rows into the aggregates
    for row in last_rows:
        if len(row) == row_length:
            add_row_to_aggregates(metadata, row, col_aliases, col_types)

    # extract free-text preamble, which may contain headers
    if not fully_parsed:
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

    # add header list to metadata
    if len(headers) > 0:
        metadata["headers"] = list(set(headers))

    add_final_aggregates(metadata, col_aliases, col_types, num_rows)

    return metadata


def add_row_to_aggregates(metadata, row, col_aliases, col_types):
    """Adds row data to aggregates.

        :param metadata: (dict) metadata dictionary to add to
        :param row: (list(str)) row of strings to add
        :param col_aliases: (list(str)) list of headers
        :param col_types: (list("num" | "str")) list of header types"""

    for i in range(0, len(row)):
        value = row[i]
        col_alias = col_aliases[i]
        col_type = col_types[i]
        is_first_row = col_alias not in metadata["columns"].keys()

        if is_first_row:
            metadata["columns"][col_alias] = {}

        if col_type == "num":
            # cast the field to a number to do numerical aggregates
            # the try except is used to pass over textual and blank space nulls on which type coercion will fail
            try:
                value = float(value)
            except ValueError:
                # skips adding to aggregates
                continue

            # start off the metadata if this is the first row of values
            if is_first_row:
                metadata["columns"][col_alias]["min"] = [float("inf"), float("inf"), float("inf")]
                metadata["columns"][col_alias]["max"] = [None, None, None]
                metadata["columns"][col_alias]["total"] = value

            # add row data to existing aggregates
            else:
                mins = list(set(metadata["columns"][col_alias]["min"] + [value]))
                maxes = list(set(metadata["columns"][col_alias]["max"] + [value]))
                metadata["columns"][col_alias]["min"] = nsmallest(3, mins)
                metadata["columns"][col_alias]["max"] = nlargest(3, maxes)
                metadata["columns"][col_alias]["total"] += value

        elif col_type == "str":
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

        if metadata["columns"][col_alias] == {}:
            metadata["columns"].pop(col_alias)

        if col_types[i] == "num":
            metadata["columns"][col_alias]["max"] = [val for val in metadata["columns"][col_alias]["max"]
                                                     if val is not None]
            metadata["columns"][col_alias]["min"] = [val for val in metadata["columns"][col_alias]["min"]
                                                     if val != float("inf")]

            metadata["columns"][col_alias]["avg"] = round(
                metadata["columns"][col_alias]["total"] / num_rows,
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


def extract_topic(file_handle, pass_fail=False):

    tokenizer = RegexpTokenizer(r'[a-zA-Z]{3,}')

    dictionary = corpora.Dictionary.load('climate_abstracts.dict')
    model = LdaModel.load("climate_abstracts.lda")

    doc = tokenizer.tokenize(file_handle.read())
    doc_bow = dictionary.doc2bow(doc)

    return model[doc_bow]
