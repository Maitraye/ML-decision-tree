from node import Node
import math
import random
import copy

def entropy(examples, target):
    labelFreq = {}
    dataEntropy = 0.0
  
  # Calculate the frequency of each of the labels of the target attribute - Class
    for entry in examples:
        if entry[target] in labelFreq:
            labelFreq[entry[target]] += 1.0
        else:
            labelFreq[entry[target]]  = 1.0

  # Calculate the entropy of the data for the target attribute - Class
    for freq in labelFreq.values():
        dataEntropy += (-freq/len(examples)) * math.log(freq/len(examples), 2) 
      
    return dataEntropy

def gain(examples, attr, target):
    valueFreq = {}
    subsetEntropy = 0.0

    #not considering missing attributes for calculating information gain, while building the tree
    examplesWithoutMissingData = [entry for entry in examples if entry[attr] != '?'] 

  # Calculate the frequency of each of the values of the current attribute for which gain is being calculated
    for entry in examplesWithoutMissingData:
        if entry[attr] in valueFreq:
            valueFreq[entry[attr]] += 1.0
        else:
            valueFreq[entry[attr]]  = 1.0

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

  # best = attributes[0] 
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

#get values in the column of the best attribute 
def getValues(examples, attr):
    # values = []
    # for entry in examples:
    #     if entry[best] not in values and entry[best] is not '?': #do not return ? sign as a value
    #         values.append(entry[best])
    # return values
    valueFreq = {}
    for entry in examples:
        if entry[attr] in valueFreq:
            valueFreq[entry[attr]] += 1.0
        else:
            valueFreq[entry[attr]]  = 1.0
    return valueFreq

def getExamples(examples, best, value):
    exampleSubset = []
    for entry in examples:
    #find entries with the give value
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
    labelFreq = {}

  #calculate frequency of values in target attr
    for entry in examples:
        if entry[target] in labelFreq:
            labelFreq[entry[target]] += 1 
        else:
            labelFreq[entry[target]] = 1

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
            #for pruning later
            root.major = mode(examples, target)
            examplesWithoutMissingData = [entry for entry in examples if entry[best] != '?'] 

            valueFreq = getValues(examplesWithoutMissingData, best)
            for value in list(valueFreq.keys()):
        # Create a subtree for the current value under the "best" field
                exampleSubset = getExamples(examplesWithoutMissingData, best, value)

                newDefault = mode(exampleSubset, target)
                newChild = ID3(exampleSubset, newDefault)
                root.children[value] = newChild

                #find out the probability of occuring each value to handle missing data
                prob = valueFreq[value] / sum(valueFreq.values())
                root.missing_attr_children[prob] = root.children[value]

        else: #if no non-trivial split is possible
            root.label = mode(examples, target)
            # print("ID3 returned for no non-trivial split")
    return root
 
max_pruning_gain = 0
best_node_to_prune = None

def findPruningGains(root, examples, currentNode, original_acc):
    global max_pruning_gain
    global best_node_to_prune

    if currentNode.label is None:
        currentNode.is_pruning = True
        currentNode.pruning_gain = test(root, examples) - original_acc
        currentNode.is_pruning = False

        if currentNode.pruning_gain > max_pruning_gain:
            max_pruning_gain = currentNode.pruning_gain
            best_node_to_prune = currentNode

        for childNode in currentNode.children.values():
            findPruningGains(root, examples, childNode, original_acc)

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
    print("original accuracy on validation: " + str(original_acc))

    global max_pruning_gain
    global best_node_to_prune

    findPruningGains(node, examples, node, original_acc)

    while max_pruning_gain > 0:
        print(str(max_pruning_gain))
        print(best_node_to_prune.name) 

        pruneNode(node, node, best_node_to_prune)

        treeprint(node)

        max_pruning_gain = 0
        findPruningGains(node, examples, node, original_acc)   
        

def test(node, examples):
    correct = 0
    for entry in examples:
        ans = evaluate(node, entry)
        # print("answer: " + ans + " original: " + entry['Class'])
        if ans == entry['Class']:
            correct += 1

    # print("correct outputs:" + str(correct))
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
                sortedProb = sorted(list(node.missing_attr_children.keys()))
                probRange = 0
                for i in range(len(sortedProb)):
                    probRange += sortedProb[i] #creating the range of probability bin
                    if randomValue <= probRange:
                        return evaluate(node.missing_attr_children[sortedProb[i]], example)
                        break


