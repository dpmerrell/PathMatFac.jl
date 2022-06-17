
import sys


if __name__=="__main__":

    input_txt = sys.argv[1]
    edge_tsv = sys.argv[2]
    node_tsv = sys.argv[3]

    f_in = open(input_txt, "r")
    f_edge = open(edge_tsv, "w")
    f_node = open(node_tsv, "w")

    is_edge = True
    for line in f_in.readlines():
        if line.isspace():
            is_edge = False
            continue
        else:
            if is_edge:
                f_edge.write(line)
            else:
                f_node.write(line)


