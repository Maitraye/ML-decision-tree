from node import Node
import math
import random
import copy

def entropy(examples, target):
    # Calculate the frequency of each of the labels of the target attribute - Class
    labelFreq = getValues(examples, target)
    dataEntropy = 0.0

    # Calculate the entropy of the data for the target attribute - Class
    for freq in labelFreq.values():
        dataEntropy += (-freq/len(examples)) * math.log(freq/len(examples), 2) 
      
    return dataEntropy

def gain(examples, attr, target):
    subsetEntropy = 0.0

    #not considering missing attributes for calculating information gain
    examplesWithoutMissingData = [entry for entry in examples if entry[attr] != '?'] 

    # Calculate the frequency of each of the values of the current attribute for which gain is being calculated
    valueFreq = getValues(examples, attr)

    # Calculate the sum of the entropy for each subset of entries weighted by their probability of occuring in the training set.
    for value in valueFreq.keys():
        valueProb = valueFreq[value] / sum(valueFreq.values())
        exampleSubset = [entry for entry in examplesWithoutMissingData if entry[attr] == value]
        subsetEntropy += valueProb * entropy(exampleSubset, target)

    #entropy of the whole examples set
    totalEntropy = entropy(examplesWithoutMissingData, target)
  
    return (totalEntropy - subsetEntropy)

def pick_best_attribute(examples):
    attributes = list(examples[0].keys())
    attributes.remove('Class') #removing the 'Class' attribute 
    target = 'Class'

    best = None
    maxGain = 0

    for attr in attributes:
        newGain = gain(examples, attr, target)
        if newGain > maxGain:
            maxGain = newGain
            best = attr

    if maxGain == 0:
        for attr in attributes:
            if len(getValues(examples, attr)) > 1:
                best = attr
                break           
    return best    

def getValues(examples, attr):
    valueFreq = {}
    for entry in examples:
        if entry[attr] in valueFreq:
            valueFreq[entry[attr]] += 1.0
        else:
            if entry[attr] != '?':
                valueFreq[entry[attr]]  = 1.0
    return valueFreq

def getExamples(examples, best, value):
    exampleSubset = []
    for entry in examples:
        #find entries with the given value
        if entry[best] == value:
            newEntry = {}
            #add attibute: value to the new entry if the attribute is not the best one
            for key in entry.keys():
                if key != best:
                    newEntry[key] = entry[key]
            exampleSubset.append(newEntry)
    return exampleSubset

#find most common value for an attribute
def mode(examples, target):
    #calculate frequency of values in target attr
    labelFreq = getValues(examples, target)

    max = 0
    major = ""
    for key in labelFreq.keys():
        if labelFreq[key]>max:
            max = labelFreq[key]
            major = key
    return major

def ID3(examples, default):
    root = Node()
    if examples:
        attributes = list(examples[0].keys())
        attributes.remove('Class') #removing the 'Class' attribute 
        target = 'Class'

        # list of labels of target attribute in the examples
        targetLabels = [entry[target] for entry in examples]
    
    # If the example set is empty, set the default value as class label 
    if not examples:
        root.label = default
        # print("ID3 returned for example set empty")

    # If all the entries in the examples have the same classification, return that classification.
    elif targetLabels.count(targetLabels[0]) == len(targetLabels):
        root.label = targetLabels[0]
        # print("ID3 returned for homogeneous")

    else:
        best = pick_best_attribute(examples)

        if best is not None: #if a non-trivial split is possible
            root.name = best
            root.major = mode(examples, target) #for pruning later

            valueFreq = getValues(examples, best)

            for value in list(valueFreq.keys()):
                #find out the probability of occuring each value to handle missing data
                prob = valueFreq[value] / sum(valueFreq.values())
                root.missing_attr_children[value] = prob

            # making a deep copy of the examples so that original examples are not altered
            examplesCopy = copy.deepcopy(examples)

            for row in examplesCopy: #replace missing values with probable attribute values
                if row[best] == '?':
                    randomValue = random.random()
                    sortedProb = sorted(list(root.missing_attr_children.values()))
                    probRange = 0
                    for i in range(len(sortedProb)):
                        probRange += sortedProb[i] #creating the range of probability bin
                        if randomValue <= probRange:
                            attr_value = [key for key, value in root.missing_attr_children.items() if value == sortedProb[i]][0]
                            row[best] = attr_value

            for value in list(valueFreq.keys()):
                # Create a subtree for the current value under the "best" field
                exampleSubset = getExamples(examplesCopy, best, value)
                newChild = ID3(exampleSubset, mode(examples, target))
                root.children[value] = newChild

        else: #if no non-trivial split is possible
            root.label = mode(examples, target)
            # print("ID3 returned for no non-trivial split")
    return root

def findPruningGains(root, examples, currentNode, original_acc, max_pruning_gain = 0, best_node_to_prune = None):
    if currentNode.label is None:
        currentNode.is_pruning = True
        currentNode.pruning_gain = test(root, examples) - original_acc
        currentNode.is_pruning = False

        if currentNode.pruning_gain > max_pruning_gain:
            max_pruning_gain = currentNode.pruning_gain
            best_node_to_prune = currentNode

        for childNode in currentNode.children.values():
            max_pruning_gain, best_node_to_prune = findPruningGains(root, examples, childNode, original_acc, max_pruning_gain, best_node_to_prune)

    return max_pruning_gain, best_node_to_prune

def pruneNode(root, currentNode, node_to_prune):
    if root is node_to_prune:   #if root is the node to prune
        newRoot = Node()
        newRoot.label = node_to_prune.major
        root = newRoot
        return
    else:
        if currentNode.label is None:
            for k,v in currentNode.children.items():
                if v is node_to_prune:
                    newNode = Node()
                    newNode.label = node_to_prune.major
                    currentNode.children[k] = newNode
                    return
                else:
                    pruneNode(root, v, node_to_prune)

def treeprint(n):
    # if n.label is not None:
      # print("output: " + n.label)
    if n.name is not None:
      print(n.name)
      for k,v in n.children.items():
        # print("attribute value: " + k)
        treeprint(v) 
        
def prune(node, examples):
    original_acc = test(node, examples)
    max_pruning_gain, best_node_to_prune = findPruningGains(node, examples, node, original_acc)

    while max_pruning_gain > 0:
        pruneNode(node, node, best_node_to_prune)
        # treeprint(node)
        max_pruning_gain, best_node_to_prune = findPruningGains(node, examples, node, original_acc)   

def test(node, examples):
    correct = 0
    for entry in examples:
        ans = evaluate(node, entry)
        # print("answer: " + str(ans) + " original: " + str(entry['Class']))
        if ans == entry['Class']:
            correct += 1

    accuracy = correct/len(examples)
    return accuracy


def evaluate(node, example):
    if node.label is not None:
        return node.label
    else:
        if node.is_pruning:
            return node.major

        elif node.name in example:
            for k,v in node.children.items():
                if k == example[node.name]:
                    return evaluate(v, example)
                    break

            else: # no attribute value matches with the attribute value of the example, i.e., either '?' or an unseen value in the example
                randomValue = random.random()
                sortedProb = sorted(list(node.missing_attr_children.values()))
                probRange = 0
                for i in range(len(sortedProb)):
                    probRange += sortedProb[i] #creating the range of probability bin

                    if randomValue <= probRange:
                        attr_value = [key for key, value in node.missing_attr_children.items() if value == sortedProb[i]][0]
                        return evaluate(node.children[attr_value], example)
                        break


