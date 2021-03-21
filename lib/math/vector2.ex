defmodule AutonomousCar.Math.Vector2 do
  import Math

  def add({ax, ay}, {bx, by}), do: {ax + bx, ay + by}
  def sub({ax, ay}, {bx, by}), do: {ax - bx, ay - by}

  def rotate({x, y} = vector, angle) do
    angle = degrees_to_radians(angle)
    {x * cos(angle) - y * sin(angle), x * sin(angle) + y * cos(angle)}
  end

  def degrees_to_radians(angle) do
    angle * (Math.pi / 180)
  end

  def distance({ax, ay}, {bx, by}) do
    :math.sqrt(:math.pow(ax - bx, 2) + :math.pow(ay - by, 2))
  end
end
