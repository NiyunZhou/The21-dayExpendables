import random
import  numpy
import cPickle
from pprint import pprint


# num of classes = 4716

divide = 3
start_seed = 1337
num_models = 2

divide_num = 4716/divide
class_index = range(4716)

model = []
for n in xrange(num_models):
    print "model"+str(n)
    random.seed(start_seed+n)
    random.shuffle(class_index)
    index = class_index
    for x in xrange(0,4716, divide_num):
        class_list = sorted(index[x:x+divide_num])
        print class_list
        model.append(class_list)

print numpy.shape(model)

with open("./6_1572_model.cPickle","wb") as f:
    cPickle.dump(model, f)
