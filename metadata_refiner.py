import os
import json
import networkx as nx
import pickle as pkl
from gensim.models.ldamodel import LdaModel
from metadata_util import inferred_nulls, extract_columnar_metadata

# load necessary LDA resources
lda_model = LdaModel.load("../ML_models/lda_model/climate_abstracts.lda")


def refine_metadata(metadata_file_name, new_metadata_file_name, graph_file_name,
                    lda_preamble=True, null_inference=True):
    with open(graph_file_name, "rb") as graph_file:
        G = pkl.load(graph_file)

    with open(metadata_file_name, "r") as metadata_file, open(new_metadata_file_name, "w") as new_metadata_file:
        metadata = json.load(metadata_file)["files"]
        for i in range(0, len(metadata)):
            item = metadata[i]
            full_path = os.path.join(item["system"]["path"], item["system"]["file"])
            print "refining metadata for {}".format(full_path)

            if "lda" not in item["system"]["extractors"]:
                topics = topic_mixture(item, metadata, G)
                item["topics"] = topics

                max_topic = max(topics, key=lambda (i, p): p)[0]
                topic_words = [str(w[0]) for w in LdaModel.show_topic(lda_model, max_topic)]
                item["tags"] = topic_words

            if "columnar" in item["system"]["extractors"] and null_inference:
                nulls = inferred_nulls(item)
                if not all([null == 0 for null in nulls]):
                    with open(full_path, "r") as file_handle:
                        new_columns = extract_columnar_metadata(file_handle, pass_fail=False, lda_preamble=lda_preamble,
                                                                null_inference=null_inference)["columns"]
                    metadata[i]["columns"] = new_columns

        json.dump(metadata, new_metadata_file)


def make_filesystem_graph(dir, graph_file=None):
    G = nx.Graph()
    for root, dirs, files in os.walk(dir):
        for item in files + dirs:
            if item[0] != ".":
                path = os.path.join(root, item)
                G.add_node(path)
                G.add_edge(root, path)

    if graph_file is not None:
        pkl.dump(G, graph_file)

    return G


def topic_mixture(item, metadata, G):
    neighbors = [item["system"]["path"] + item["system"]["file"]]
    in_mixture = []
    while not in_mixture:
        for neighbor in neighbors:
            path_parts = neighbor.strip().rsplit("/", 1)
            path, file_name = path_parts if len(path_parts) > 1 else path_parts + [""]
            path += "/"
            neighbor_metadata = next((m for m in metadata if m["system"]["path"] == path and
                                      m["system"]["file"] == file_name), None)
            if neighbor_metadata is not None and "lda" in neighbor_metadata["system"]["extractors"]:
                in_mixture.append(neighbor_metadata["topics"])

        neighbors = [G.neighbors(n) for n in neighbors]
        # flatten neighbors list
        neighbors = [n for sublist in neighbors for n in sublist]

    mixture = mix_topics(in_mixture)

    return mixture


def mix_topics(all_topics):
    mixture = []
    for topics in all_topics:
        for topic in topics:
            # index of topic within mixture
            i = next((i for i, mixture_topic in enumerate(mixture) if mixture_topic[0] == topic[0]), None)
            if i is not None:
                mixture[i][1] += topic[1]
            else:
                mixture.append(topic)

    # normalize topic mixture to sum to 1
    sum_topics = sum([topic[1] for topic in mixture])
    mixture = [[topic_num, prob / sum_topics] for [topic_num, prob] in mixture]
    # sort topics
    mixture = sorted(mixture, key=lambda t: t[0])

    return mixture
