if !isdir(Pkg.dir("Utils"))
  Pkg.clone("https://github.com/AStupidBear/Utils.jl.git")
  Pkg.build("Utils")
end

if !isdir(Pkg.dir("HyperOpt"))
  Pkg.rm("HyperOpt")
  Pkg.clone("https://github.com/AStupidBear/HyperOpt.jl.git")
  Pkg.build("HyperOpt")
end

run(`pip install -I numpy==1.11.0`)
!isdir("hyperopt-sklearn") && run(`git clone https://github.com/hyperopt/hyperopt-sklearn.git`)
cd("hyperopt-sklearn")
run(`pip install -e .`)
cd("..")
