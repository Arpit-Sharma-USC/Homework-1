import weka.core.jvm as jvm
from weka.classifiers import Classifier
from weka.flow.control import Flow, Branch, Sequence
from weka.classifiers import FilteredClassifier
from weka.core.converters import Loader
from weka.classifiers import Evaluation
from weka.core.classes import Random
from weka.filters import Filter
import seaborn as sns
import matplotlib as plt

jvm.start()

loader = Loader(classname="weka.core.converters.CSVLoader")
data = loader.load_file("C:/Arpit/aps.failure_training_set.csv")

# print(str(data))data = loader.load_file( + "aps.failure_training_set.csv")
data.class_is_last()

# remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "1-3"])

cls = Classifier(classname="weka.classifiers.trees.LMT")

fc = FilteredClassifier()
# fc.filter = remove
fc.classifier = cls

evl = Evaluation(data)
evl.crossvalidate_model(fc, data, 10, Random(1))
conf=evl.confusion_matrix
print(evl.percent_incorrect)
# print(evl.summary())
# print(evl.class_details())
sns.heatmap(conf,cmap="YlGnBu",annot=True,linewidths=.5,fmt='d')

print("AUC",evl.area_under_prc)
import weka.plot.classifiers as plcls  # NB: matplotlib is required
plcls.plot_roc(evl, class_index=[0, 1], wait=True)
# test_model(classifier, data, output=None)
plt.show()











