import ID3, parse, random

def testPruningOnHouseData(inFile):
    data = parse.parse(inFile)
    avgWithPruning = []
    avgWithoutPruning = []

    for run in range(300, 0 , -10):
        withPruning = []
        withoutPruning = []

        for i in range(100):
            random.shuffle(data)

            train = data[:int(run*0.7)]
            valid = data[int(run*0.7):run]
            test = data[run:]

            tree = ID3.ID3(train, 'democrat')
            acc = ID3.test(tree, train)
            print("training accuracy: ",acc)
            acc = ID3.test(tree, valid)
            print("validation accuracy: ",acc)
            acc = ID3.test(tree, test)
            print("test accuracy: ",acc)

            ID3.prune(tree, valid)
            acc = ID3.test(tree, train)
            print("pruned tree train accuracy: ",acc)
            acc = ID3.test(tree, valid)
            print("pruned tree validation accuracy: ",acc)
            acc = ID3.test(tree, test)
            print("pruned tree test accuracy: ",acc)
            withPruning.append(acc)
            tree = ID3.ID3(train+valid, 'democrat')
            acc = ID3.test(tree, test)
            print("no pruning test accuracy: ",acc)
            withoutPruning.append(acc)

        print("average with pruning",sum(withPruning)/len(withPruning)," without: ",sum(withoutPruning)/len(withoutPruning))
        avgWithPruning.append(sum(withPruning)/len(withPruning))
        avgWithoutPruning.append(sum(withoutPruning)/len(withoutPruning))

    print(avgWithPruning)
    print(avgWithoutPruning)

if __name__ == "__main__":
    testPruningOnHouseData('house_votes_84.data')