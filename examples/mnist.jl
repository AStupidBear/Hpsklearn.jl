using Utils, Plots; reload("Hpsklearn")

using MLDatasets
xtrn, ytrn = MNIST.traindata()
xtst, ytst = MNIST.testdata()
xtrn = reshape(xtrn, 28*28, 60000)'
xtst = reshape(xtst, 28*28, 10000)'

clf = Hpsklearn.Classifier(; maxevals = 1)
Hpsklearn.fit!(clf, xtrn, ytrn)
# Hpsklearn.fit_demo!(clf, xtrn, ytrn)
Hpsklearn.test(clf, xtst, ytst)
