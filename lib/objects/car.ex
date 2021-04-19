defmodule AutonomousCar.Objects.Car do

  def move(%{objects: %{car: car}} = state) do
    car_velocity_rotate = AutonomousCar.Math.Vector2.rotate(car.velocity, car.angle)
    new_coords = Scenic.Math.Vector2.add(car.coords, car_velocity_rotate)

    # Keep car inside
    new_car_coords =
      with {car_coords_x, car_coords_y} <- new_coords,
           viewport_width <- state.viewport_width,
           viewport_height <- state.viewport_height do
        cond do
          car_coords_x + 10 >= viewport_width -> {viewport_width - 20, car_coords_y}
          car_coords_x <= 10 -> {10, car_coords_y}
          car_coords_y + 10 >= viewport_height -> {car_coords_x, viewport_height - 20}
          car_coords_y <= 10 -> {car_coords_x, 10}
          true -> new_coords
        end
      end

    state
    |> put_in([:objects, :car, :last_coords], car.coords)
    |> put_in([:objects, :car, :coords], new_car_coords)
  end

  def update_angle(state, action) do
    rotation = action?(action)

    state
    |> put_in([:objects, :car, :angle], state.objects.car.angle + rotation)
  end

  defp action?(0), do: -20
  defp action?(2), do: 20
  defp action?(_), do: 0
end
