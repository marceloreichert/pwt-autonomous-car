defmodule AutonomousCar.MixProject do
  use Mix.Project

  def project do
    [
      app: :autonomous_car,
      version: "0.1.0",
      elixir: "~> 1.10.0",
      build_embedded: true,
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      mod: {AutonomousCar, []},
      extra_applications: []
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:scenic, "~> 0.10"},
      {:scenic_driver_glfw, "~> 0.10"},
      {:math, "~> 0.3.0"},
      {:neural_network, path: "~/works/neural_network_elixir"}
    ]
  end
end
