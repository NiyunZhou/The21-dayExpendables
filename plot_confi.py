import cPickle
import pandas
import matplotlib.pyplot as plt


def norm(lists):
    minv = float(min(lists))
    maxv = float(max(lists))
    div = maxv - minv
    
    for x in xrange(len(lists)):
        lists[x] = (lists[x] - minv) / div
    return lists

with open("./confi_class.cPickle","rb") as f:
    confi_class = cPickle.load(f)




acc_confi = [[],[],[],[]]
for x in xrange(len(confi_class)):
    confi_num = [0,0,0,0]
    for n in xrange(len(confi_class[x])):
        if confi_class[x][n] > 0:
            if confi_class[x][n] <= 0.1:
                confi_num[0] += 1
            if confi_class[x][n] <= 0.2:
                confi_num[1] += 1
            if confi_class[x][n] <= 0.3:
                confi_num[2] += 1
            if confi_class[x][n] <= 0.4:
                confi_num[3]+= 1

    for k in xrange(4):
        acc_confi[k].append(confi_num[k] / float(len(confi_class[x])))




#plot
plt.figure(1)

# conficence vs ap
plt.subplot(411)
plt.xlabel("Class indeces")
plt.ylabel("Ratio of confidence < "+str(0.1))
plt.title("Confidence distributions")
plt.plot(acc_confi[0])




# num of pos
plt.subplot(412)
plt.xlabel("Class indeces")
plt.ylabel("Ratio of confidence < "+str(0.2))
plt.title("Confidence distributions")
plt.plot(acc_confi[1])



# num of sample
plt.subplot(413)
plt.xlabel("Class indeces")
plt.ylabel("Ratio of confidence < "+str(0.3))
plt.title("Confidence distributions")
plt.plot(acc_confi[2])


plt.subplot(414)
plt.xlabel("Class indeces")
plt.ylabel("Ratio of confidence < "+str(0.4))
plt.title("Confidence distributions")
plt.plot(acc_confi[3])

plt.show()
