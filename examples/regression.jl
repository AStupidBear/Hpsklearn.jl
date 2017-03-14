using Utils, PlotRecipes, Hpsklearn

reg = Regressor(; maxevals = 20, verbose = true)
X = linspace(0, 6, 100)''
y = sin(X) + sin(6X) + 0.1randn(size(X, 1))
Hpsklearn.fit!(reg, X, y)
Hpsklearn.fit_demo!(reg, X, y)
ypred = predict(reg, X)
plot(plot(y, ypred), plot(X, [y ypred]))
