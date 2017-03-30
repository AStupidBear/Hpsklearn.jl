using Utils, Plots; reload("Hpsklearn")

import Knet; include(Knet.dir("examples", "housing.jl"))
xtrn, ytrn, xtst, ytst = Housing.loaddata()
xtrn, ytrn = xtrn', ytrn'
xtst, ytst = xtst', ytst'

reg = Hpsklearn.Regressor(; maxevals = 20)
Hpsklearn.fit!(reg, xtrn, ytrn)
# Hpsklearn.fit_demo!(reg, xtrn, ytrn)
Hpsklearn.test(reg, xtst, ytst)

ypred = Hpsklearn.predict(reg, xtst)
scatter(ytst, ypred)
savefig(joinpath(tempdir(), "house.html"))
