defmodule AutonomousCar.NeuralNetwork.And do
  import Nx.Defn

  defn init_params do
    w1 = Nx.random_normal({2, 3}, 0.0, 0.1, names: [:input, :hidden])
    b1 = Nx.random_normal({3}, 0.0, 0.1, names: [:hidden])
    w2 = Nx.random_normal({3, 2}, 0.0, 0.1, names: [:hidden, :output])
    b2 = Nx.random_normal({2}, 0.0, 0.1, names: [:output])

    {w1, b1, w2, b2}
  end

  defn predict({m, b}, inp) do
    inp
    |> Nx.dot(m)
    |> Nx.add(b)
  end

  # MSE Loss
  defn loss({m, b}, inp, tar) do
    preds = predict({m, b}, inp)
    Nx.mean(Nx.power(tar - preds, 2))
  end

  defn update({m, b}, inp, tar, step) do
    {grad_m, grad_b} = grad({m, b}, loss({m, b}, inp, tar))
    {m - grad_m * step, b - grad_b * step}
  end

  def train(params, epochs) do
    require IEx
    IEx.pry()
    update(params, Nx.tensor([[0,0],[0,1],[1,0],[1,1]]), Nx.tensor([[0],[0],[0],[1]]), 0.001)
  end
end

# Nx.default_backend(Torchx.Backend)

# params = AutonomousCar.NeuralNetwork.Regression.init_random_params()
# m = :rand.normal(0.0, 10.0)
# b = :rand.normal(0.0, 5.0)
# IO.puts("Target m: #{m} Target b: #{b}\n")
#
# lin_fn = fn x -> m * x + b end
# epochs = 100
#
# # These will be very close to the above coefficients
# {time, {trained_m, trained_b}} = :timer.tc(LinReg, :train, [params, epochs, lin_fn])
#
# trained_m =
#   trained_m
#   |> Nx.squeeze()
#   |> Nx.backend_transfer()
#   |> Nx.to_scalar()
#
# trained_b =
#   trained_b
#   |> Nx.squeeze()
#   |> Nx.backend_transfer()
#   |> Nx.to_scalar()
#
# IO.puts("Trained in #{time / 1_000_000} sec.")
# IO.puts("Trained m: #{trained_m} Trained b: #{trained_b}\n")
# IO.puts("Accuracy m: #{m - trained_m} Accuracy b: #{b - trained_b}")
