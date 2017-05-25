import cPickle
import pandas
import matplotlib.pyplot as plt
import sys
from tensorflow import gfile


def norm(lists):
    minv = float(min(lists))
    maxv = float(max(lists))
    div = maxv - minv
    
    for x in xrange(len(lists)):
        lists[x] = (lists[x] - minv) / div
    return lists


data_dir = sys.argv[1]
print "Reading data from",data_dir

with gfile.Open(data_dir+"AP_Confi_i.cPickle", "rb") as f:
    data = cPickle.load(f)
#with open("./num_positive.cPickle","rb") as f:
 #   num_pos = cPickle.load(f)
#vercab = pandas.read_csv("./vocabulary.csv")
with gfile.Open(data_dir+"label_class.cPickle", "rb") as f:
    label_class = cPickle.load(f)

with gfile.Open(data_dir+"confi_class.cPickle","rb") as f:
    confi_class = cPickle.load(f)



#calculate
derive_ap = [data["ap"][0]]
for x in xrange(1,len(data["ap"])):
    derive_ap.append(data["ap"][x]-data["ap"][x-1])

"""
total_val_pos = 1401828 * 3.4
total_train_num = sum(vercab["TrainVideoCount"])
print total_train_num
for x in xrange(len(num_pos)):
    print x, vercab["TrainVideoCount"][x]
    num_pos[x] = num_pos[x] / (total_val_pos * float(vercab["TrainVideoCount"][x]) / total_train_num)

"""
acc_class = []
for x in xrange(len(label_class)):
    pos_num = sum(label_class[x])
    acc = pos_num /  float(len(label_class[x]))
    acc_class.append(acc)


acc_confi = []
confi_cutoff = 0.4
for x in xrange(len(confi_class)):
    confi_num = 0
    for n in xrange(len(confi_class[x])):
        if confi_class[x][n] <= confi_cutoff:
            confi_num += 1
    acc = float(confi_num) / len(confi_class[x])
    acc_confi.append(acc)




#plot
plt.figure(1)

# conficence vs ap
plt.subplot(221)
plt.xlabel("Confidence")
plt.ylabel("AP")
plt.title("Confidence vs AP")
plt.plot(data["confidence"],derive_ap)


# num of pos
plt.subplot(222)
plt.xlabel("Class indeces")
plt.ylabel("Recall TP/(TP+FN)")
plt.title("Recall of each class in validation sets")
plt.plot(acc_class)

# num of sample
plt.subplot(223)
plt.xlabel("Class indeces")
plt.ylabel("Ratio of confidence < "+str(confi_cutoff))
plt.title("Confidence distributions")
plt.plot(acc_confi)

with gfile.Open(data_dir+"GAP_confi_class.png","wb") as f:
    plt.savefig(f)
