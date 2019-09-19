defmodule AutonomousCar.Math.Activation do
  def softmax([action]) do
    softmax(action)
  end

  def softmax(action) do
    c = Enum.max(action)
    x1 = Enum.map(action, fn(y) -> y-c end)
    sum = listsum(x1)
    Enum.map(x1, fn(y) -> :math.exp(y)/sum end)
  end

  def listsum([]) do 0 end
  def listsum([x|xs]) do
    :math.exp(x) + listsum(xs)
  end
end
