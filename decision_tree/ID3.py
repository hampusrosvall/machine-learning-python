from collections import Counter
from graphviz import Digraph
from ToyData import ToyData
import numpy as np

class ID3DecisionTreeClassifier :
    def __init__(self, minSamplesLeaf = 1, minSamplesSplit = 2) :

        self.__nodeCounter = 0

        # the graph to visualise the tree
        self.__dot = Digraph(comment='The Decision Tree')

        # suggested attributes of the classifier to handle training parameters
        self.__minSamplesLeaf = minSamplesLeaf
        self.__minSamplesSplit = minSamplesSplit

        self.__attributes = None


    # Create a new node in the tree with the suggested attributes for the visualisation.
    # It can later be added to the graph with the respective function
    def new_ID3_node(self):
        node = {'id': self.__nodeCounter, 'label': None, 'attribute': None, 'entropy': None, 'samples': None,
                         'classCounts': None, 'nodes': None, 'value' : None}

        self.__nodeCounter += 1
        return node

    # adds the node into the graph for visualisation (creates a dot-node)
    def add_node_to_graph(self, node, parentid=-1):
        nodeString = ''
        for k in node:
            if ((node[k] != None) and (k != 'nodes')):
                nodeString += "\n" + str(k) + ": " + str(node[k])

        self.__dot.node(str(node['id']), label=nodeString)
        if (parentid != -1):
            self.__dot.edge(str(parentid), str(node['id']))
            nodeString += "\n" + str(parentid) + " -> " + str(node['id'])

        print(nodeString)
        
        return


    # make the visualisation available
    def make_dot_data(self) :
        return self.__dot

    def calculate_entropy(self, data, target):
        entropy = 0
        class_count = Counter(target)
        for val in class_count.values():
            p_cl = val / len(target)
            entropy += p_cl * np.log2(p_cl)

        return -1 * entropy

    # For you to fill in; Suggested function to find the best attribute to split with, given the set of
    # remaining attributes, the currently evaluated data and target.
    def find_split_attr(self, data, target, attributes):
        data = np.array(data)
        target = np.array(target)
        G_sa = {}
        entropy = self.calculate_entropy(data, target)
        for attribute in attributes:
            subset_entropy = 0
            for val in attributes[attribute]:
                subset_idx = list()
                for idx, x in enumerate(data):
                    x_zip = dict(zip(self.__attributes, x))
                    if x_zip[attribute] == val:
                        subset_idx.append(idx)

                data_subset = data[subset_idx]
                target_subset = target[subset_idx]
                subset_entropy += len(data_subset)/len(data) * self.calculate_entropy(data_subset, target_subset)

            G_sa[attribute] = entropy - subset_entropy

        return max(G_sa, key = lambda key : G_sa[key])

    def fit(self, data, target, attributes, classes):
        self.__attributes = attributes.keys()
        root = self._fit_rek(data, target, attributes)
        self.add_node_to_graph(root)
        return root



    # the entry point for the recursive ID3-algorithm, you need to fill in the calls to your recursive implementation
    def _fit_rek(self, data, target, attributes):
        data = np.array(data)
        target = np.array(target)

        # Create a root node for the tree
        root = self.new_ID3_node()

        # If all samples belong to one class <class_name>
        if len(np.unique(target)) == 1:
            root['label'] = target[0]
            root['entropy'] = 0
            root['samples'] = len(data)
            root['classCounts'] = Counter(target)
            return root
        elif not attributes.keys():
            cnt = Counter(target)
            root['label'] = max(cnt, key = lambda key : cnt[key])
            root['entropy'] = self.calculate_entropy(data, target)
            root['samples'] = len(data)
            root['classCounts'] = Counter(target)
            return root
        else:
            root['attribute'] = self.find_split_attr(data, target, attributes)
            root['entropy'] = self.calculate_entropy(data, target)
            root['samples'] = len(data)
            root['classCounts'] = Counter(target)

            for val in attributes[root['attribute']]:
                subset_idx = list()
                for idx, ex in enumerate(data):
                    x = dict(zip(self.__attributes, ex))
                    if x[root['attribute']] == val:
                        subset_idx.append(idx)

                subset_data = data[subset_idx]
                subset_target = target[subset_idx]

                if len(subset_idx) < self.__minSamplesSplit:
                    leaf = self.new_ID3_node()
                    leaf['label'] = max(Counter(target), key=lambda key: Counter(target)[key])
                    leaf['samples'] = len(subset_idx)
                    leaf['classCount'] = Counter(subset_target)
                    leaf['value'] = val
                    try:
                        root['nodes'].append(leaf)
                    except:
                        root['nodes'] = list()
                        root['nodes'].append(leaf)
                else:
                    attributes_rem = attributes.copy()
                    del attributes_rem[root['attribute']]
                    leaf = self._fit_rek(subset_data, subset_target, attributes_rem)
                    leaf['value'] = val
                    try:
                        root['nodes'].append(leaf)
                    except:
                        root['nodes'] = list()
                        root['nodes'].append(leaf)

                self.add_node_to_graph(leaf, root['id'])

        return root

    def predict(self, data, tree) :
        predicted = list()

        for x in data:
            predicted.append(self._predict_rek(tree, x))
        return predicted

    def _predict_rek(self, node, x):
        x_zipped = dict(zip(self.__attributes, x))
        if not node['nodes']:
            return node['label']
        else:
            for child in node['nodes']:
                split_atr = node['attribute']
                if x_zipped[split_atr] == child['value']:
                    return self._predict_rek(child, x)

if __name__ == "__main__":
    td = ToyData()
    attributes, classes, X_train, y_train, X_test, y_test = td.get_data()

    tree = ID3DecisionTreeClassifier()
    #print(tree.calculate_entropy(X_train, y_train))
    #print(tree.calculate_entropy([1, 1, 1], ['+', '+', '+']))
    #print(tree.find_split_attr(X_train, y_train, attributes))
    clf = tree.fit(X_train, y_train, attributes, classes)

    plot = tree.make_dot_data()
    plot.render("myTree")

    pred_train = tree.predict(X_train, clf)
    print('Accuracy train: ', np.mean(np.array(pred_train) == np.array(y_train)))

    pred_test = tree.predict(X_test, clf)
    print('Accuracy test: ', np.mean(np.array(pred_test) == np.array(y_test)))

