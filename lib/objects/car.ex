defmodule AutonomousCar.Objects.Car do

  alias AutonomousCar.Math.Vector2

  def move(%{objects: %{car: car}} = state) do
    # atualiza a posição do carro de acordo com sua última posição e velocidade
    new_pos = Vector2.add(car.coords, car.velocity)
    rotated = Vector2.rotate({10,0}, car.angle)
    new_coords = Vector2.add(rotated, new_pos)

    # Keep car inside
    new_car_coords =
      with {sensor_center_x, sensor_center_y} <- state.objects.car.sensor.center,
           {car_coords_x, car_coords_y} <- state.objects.car.coords,
           viewport_width <- state.viewport_width do
        case sensor_center_x do
          sensor_center_x when sensor_center_x >= viewport_width -> {viewport_width, car_coords_y}
          _ -> new_coords
        end
      end
    # IO.inspect car_coords, label: 'car_coords -->'

    state
    |> put_in([:objects, :car, :last_coords], car.coords)
    |> put_in([:objects, :car, :coords], new_car_coords)
  end

  def update_rotation(state, action) do
    rotation = action?(action)

    state
    |> put_in([:objects, :car, :angle], state.objects.car.angle + rotation)
  end

  defp action?(1), do: -20
  defp action?(2), do: 20
  defp action?(_), do: 0
end
