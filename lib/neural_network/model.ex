defmodule AutonomousCar.NeuralNetwork.Model do
  use GenServer

  def init(pid) do

    {:ok, pid}
  end

  def handle_cast({:push, params}, state) do
    {:noreply, params}
  end

  def handle_call(:pull, _from, state) do
    {:reply, state, state}
  end

  def handle_cast(:reset, state) do
    {:noreply, state = []}
  end

  # Public API
  def start_link() do
    GenServer.start_link(AutonomousCar.NeuralNetwork.Model, [])
  end

  def push(params, pid) do
    GenServer.cast(pid, {:push, params})
  end

  def pull(pid) do
    GenServer.call(pid, :pull)
  end

  def count(pid) do
    GenServer.call(pid, :count)
  end

  def reset(pid) do
    GenServer.cast(pid, :reset)
  end
end
