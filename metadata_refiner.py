import os
import networkx as nx
import pickle as pkl


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
    neighbors = [item]
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
                print neighbor
                print neighbor_metadata["topics"]
                print "\n\n"

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


