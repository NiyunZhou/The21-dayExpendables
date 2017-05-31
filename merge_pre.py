import pandas
import cPickle
import numpy

avg = "AM"  #or "GM" or "HM"
total_class = 4716

final_prediction = {}
final_prediction["VideoId"] = []
final_prediction["LabelConfidencePairs"] = []

predictions = []

prediction_dir = "./"
prediction_list = ["predictions1.csv", "predictions2.csv", "predictions3.csv"]



def accumulate(single_pre, acc_pre, model_idx):
    for x in xrange(0,len(single_pre),2):
        idx = int(single_pre[x])
        confidence = float(single_pre[x+1])
        acc_pre[model_idx[idx]] += confidence
    return acc_pre

def format_lines(video_ids,predictions):
    top_k = 20
    top_indices = numpy.argpartition(predictions, -top_k)[-top_k:]
    line = [(class_index, predictions[class_index]) for class_index in top_indices]
    line = sorted(line, key=lambda p: -p[1])
    return video_ids.decode('utf-8') + "," + " ".join("%i %f" % pair for pair in line) + "\n"



for filename in prediction_list:
    with open(prediction_dir + filename,"rb") as f:
        predictions.append(pandas.read_csv(f))

with open("./6_1572_model.cPickle","rb") as f:
    model_idx = cPickle.load(f)

sample_num = len(predictions[0])
print sample_num

for x in xrange(sample_num):
    acc_pre = [0 for _ in xrange(total_class)]
    for y in xrange(len(predictions)):
        single_pre = predictions[y]["LabelConfidencePairs"][x].split(" ")
        acc_pre = accumulate(single_pre, acc_pre, model_idx[y])
    # Full model prediction acc?
    acc_pre = [n / len(prediction_list) for n in acc_pre]
    print format_lines(predictions[0]["VideoId"][x], acc_pre)



    #print single_pre
#print prediction


