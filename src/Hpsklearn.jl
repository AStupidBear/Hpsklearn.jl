module Hpsklearn

using Utils, PyCall, HyperOpt.TPE, ScikitLearn

HyperoptEstimator = pyimport("hpsklearn")[:HyperoptEstimator]
any_preprocessing = pyimport("hpsklearn.components")[:any_preprocessing]
any_regressor = pyimport("hpsklearn.components")[:any_regressor]
any_classifier = pyimport("hpsklearn.components")[:any_classifier]

export Regressor
Regressor(;regressor = any_regressor("reg"),
          trial_timeout = 60.0 * 5,
          maxevals = 1,
          verbose = false, o...) =
          HyperoptEstimator(
              preprocessing = any_preprocessing("pp"),
              regressor = regressor,
              algo = TPESUGGEST,
              trial_timeout = trial_timeout, # seconds
              maxevals = maxevals,
              verbose = verbose, o...)

export Classifier
Classifier(;classifier = any_classifier("clf"),
          trial_timeout = 60.0 * 5,
          maxevals = 1,
          verbose = false, o...) =
          HyperoptEstimator(
            preprocessing = any_preprocessing("pp"),
            classifier = classifier,
            algo = TPESUGGEST,
            trial_timeout = trial_timeout, # seconds
            maxevals = maxevals,
            verbose = verbose, o...)

export fit!
function fit!(estimator::PyObject, X, y)
  ScikitLearn.fit!(estimator, X, y)
  report(estimator)
end

export fit_demo!
function fit_demo!(estimator::PyObject, X, y)
  fit_iterator = estimator[:fit_iter](X, y)
  fit_iterator[:next]()
  while length(estimator[:trials][:trials]) < estimator[:maxevals]
    fit_iterator[:send](1) # -- try one more model
    @> estimator demo_plot Main.savefig(tempfile("Hpsklearn.html"))
  end
  estimator[:retrain_best_model_on_full_data](X, y)
  report(estimator)
end

function report(estimator)
  println()
  println("Best preprocessing pipeline:")
  for pp in estimator[:_best_preprocs]
    println(pp)
  end
  println()
  println("Best learner:\n", estimator[:_best_learner])
  println()
  losses = estimator[:trials][:losses]()
  loss = is(losses, nothing) ? nothing : minimum(losses)
  if loss == nothing
    println("Prediction loss in validation is $loss")
  elseif loss < 1
    @printf("Prediction loss in validation is %.1f%%\n", 100*loss)
  elseif loss > 1
    @printf("Prediction loss in validation is %.2f\n", loss)
  end
end

function demo_plot(estimator)
  p1 = scatter_error_vs_time(estimator)
  p2 = plot_minvalid_vs_time(estimator)
  p = Main.plot(p1, p2)
end

function scatter_error_vs_time(estimator)
  losses = estimator[:trials][:losses]()
  Main.scatter(losses, legend=:none, xlabel="Iteration",
              ylabel="validation error rate")
end

function plot_minvalid_vs_time(estimator)
  losses = estimator[:trials][:losses]()
  mins = minimums(losses)
  Main.plot(mins, legend=:none, xlabel="Iteration",
          ylabel="minimum validation error rate to-date")
end

end
