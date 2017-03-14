using Utils, Plots, RDatasets, Hpsklearn

clf = Classifier(; maxevals = 20, verbose = true)
iris = dataset("datasets", "iris")
X, y = Matrix(iris[1:4]), Vector(iris[5])
encoder = LabelEncoder()
y = fit_transform(encoder, y)
Hpsklearn.fit!(clf, X, y)
Hpsklearn.fit_demo!(clf, X, y)
ypred = Hpsklearn.predict(clf, X)
sum(ypred .== y) / length(y)
