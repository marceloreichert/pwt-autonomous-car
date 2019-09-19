defmodule NeuralNetwork.Network do
  alias NeuralNetwork.{Layer, Network, Neuron}
  alias AutonomousCar.Math.Activation

  defstruct pid: nil, input_layer: nil, hidden_layers: [], output_layer: nil, error: 0

  def start_link(layer_sizes \\ []) do
    {:ok, pid} = Agent.start_link(fn -> %Network{} end)

    layers =
      map_layers(
        input_neurons(layer_sizes),
        hidden_neurons(layer_sizes),
        output_neurons(layer_sizes)
      )

    pid |> update(layers)
    pid |> connect_layers
    {:ok, pid}
  end

  def get(pid), do: Agent.get(pid, & &1)

  def update(pid, fields) do
    fields = Map.merge(fields, %{pid: pid})
    Agent.update(pid, &Map.merge(&1, fields))
  end

  defp input_neurons(layer_sizes) do
    size = layer_sizes |> List.first()
    {:ok, pid} = Layer.start_link(%{neuron_size: size})
    pid
  end

  defp hidden_neurons(layer_sizes) do
    layer_sizes
    |> hidden_layer_slice
    |> Enum.map(fn size ->
      {:ok, pid} = Layer.start_link(%{neuron_size: size})
      pid
    end)
  end

  defp output_neurons(layer_sizes) do
    size = layer_sizes |> List.last()
    {:ok, pid} = Layer.start_link(%{neuron_size: size})
    pid
  end

  defp hidden_layer_slice(layer_sizes) do
    layer_sizes
    |> Enum.slice(1..(length(layer_sizes) - 2))
  end

  defp connect_layers(pid) do
    layers = pid |> Network.get() |> flatten_layers

    layers
    |> Stream.with_index()
    |> Enum.each(fn tuple ->
      {layer, index} = tuple
      next_index = index + 1

      if Enum.at(layers, next_index) do
        Layer.connect(layer, Enum.at(layers, next_index))
      end
    end)
  end

  defp flatten_layers(network) do
    [network.input_layer] ++ network.hidden_layers ++ [network.output_layer]
  end

  def predict(network, input_values) do
    network.input_layer
    |> Layer.activate(input_values)

    Enum.each(network.hidden_layers, fn hidden_layer ->
      hidden_layer
      |> Layer.activate()
    end)

    network.output_layer
    |> Layer.activate()

    prob_actions =
      network.output_layer
      |> Layer.get()
      |> Layer.neurons_output()
      |> Activation.softmax()
      |> IO.inspect

    action =
      prob_actions
      |> Enum.find_index(fn value -> Enum.max(prob_actions) == value end)
      |> IO.inspect
  end

  def train(network, target_outputs) do
    network.output_layer |> Layer.get() |> Layer.train(target_outputs)
    network.pid |> update(%{error: error_function(network, target_outputs)})

    network.hidden_layers
    |> Enum.reverse()
    |> Enum.each(fn layer_pid ->
      layer_pid |> Layer.get |> Layer.train(target_outputs)
    end)

    network.input_layer |> Layer.get() |> Layer.train(target_outputs)
  end

  defp error_function(network, target_outputs) do
    (Layer.get(network.output_layer).neurons
     |> Stream.with_index()
     |> Enum.reduce(0, fn {neuron, index}, sum ->
       target_output = Enum.at(target_outputs, index)
       actual_output = Neuron.get(neuron).output
       squared_error(sum, target_output, actual_output)
     end)) / length(Layer.get(network.output_layer).neurons)
  end

  defp squared_error(sum, target_output, actual_output) do
    sum + 0.5 * :math.pow(target_output - actual_output, 2)
  end

  defp map_layers(input_layer, hidden_layers, output_layer) do
    %{
      input_layer: input_layer,
      output_layer: output_layer,
      hidden_layers: hidden_layers
    }
  end
end
