defmodule AutonomousCar.Scene.Environment do
  use Scenic.Scene

  alias Scenic.Graph
  alias Scenic.ViewPort

  alias AutonomousCar.Math.Vector2
  alias AutonomousCar.Objects.Car

  alias NeuralNetwork.Network

  import Scenic.Primitives

  # Initial parameters for the game scene!
  def init(_arg, opts) do
    viewport = opts[:viewport]

    # Initializes the graph
    graph = Graph.build(theme: :dark)

    # Calculate the transform that centers the car in the viewport
    {:ok, %ViewPort.Status{size: {viewport_width, viewport_height}}} = ViewPort.info(viewport)

    # Initial pos
    {pos_x, pos_y} = { trunc(viewport_width / 2), trunc(viewport_height / 2)}

    # start a very simple animation timer
    {:ok, timer} = :timer.send_interval(60, :frame)

    # start neural network
    {:ok, neural_network_pid} = Network.start_link([5, 30, 3], %{activation: :relu})

    state = %{
      viewport: viewport,
      viewport_width: viewport_width,
      viewport_height: viewport_height,
      graph: graph,
      frame_count: 0,
      neural_network_pid: neural_network_pid,
      objects: %{
        car: %{
          dimension: %{width: 20, height: 10},
          coords: {pos_x, pos_y},
          last_coords: {pos_x, pos_y},
          velocity: {1, 0},
          angle: 0,
          signal: %{
            left: 0,
            center: 0,
            right: 0
          },
          sensor: %{
            left: {0,0},
            center: {0,0},
            right: {0,0}
          },
          last_reward: 0,
        },
        goal: %{coords: {20,20}}
      }
    }

    graph = draw_objects(graph, state.objects)

    {:ok, state, push: graph}
  end

  def handle_info(:frame, %{frame_count: frame_count} = state) do

    IO.inspect ' --------> PROCESS <----------'

    # Posição atual do carro
    {car_x, car_y} = state.objects.car.coords
    # IO.inspect {car_x, car_y}, label: 'Posição do carro'

    # Última posição do carro
    {car_last_x, car_last_y} = state.objects.car.last_coords
    # IO.inspect {car_last_x, car_last_y}, label: 'Última posição do carro'

    # Última recompensa do carro
    car_last_reward = state.objects.car.last_reward
    # IO.inspect {car_last_reward}, label: 'Última recompensa do carro'

    # Posição atual do objetivo
    {goal_x, goal_y} = state.objects.goal.coords
    # IO.inspect {goal_x, goal_y}, label: 'Posição do Objetivo'

    # Vetor1 - Posição atual do carro menos anterior
    vector1 = Scenic.Math.Vector2.sub({car_x, car_y}, {car_last_x, car_last_y})
    # IO.inspect vector1, label: 'Vector1'

    # Vetor2 - Posição do objetivo - posicao anterior do carro
    vector2 = Scenic.Math.Vector2.sub({goal_x, goal_y}, {car_last_x, car_last_y})
    # IO.inspect vector2, label: 'Vector2'

    # Normaliza vetor1
    vector1_normalized = Scenic.Math.Vector2.normalize(vector1)
    # IO.inspect vector1_normalized, label: 'Vector1 Normalizado'

    # Normaliza vetor2
    vector2_normalized = Scenic.Math.Vector2.normalize(vector2)
    # IO.inspect vector2_normalized, label: 'Vector2 Normalizado'

    # Orientação
    orientation = Scenic.Math.Vector2.dot(vector1_normalized, vector2_normalized)
    # IO.inspect orientation, label: 'Dot dos vetores'

    # Orientação em Radiano
    orientation_rad = Math.acos(orientation)
    # IO.inspect orientation_rad, label: 'Orientação rad'

    # Orientação em Graus
    orientation_grad = (180 / :math.pi) * orientation_rad
    # IO.inspect orientation_grad, label: 'Orientação grad'

    # Rotação do carro
    # car_rotation_rad = state.objects.car.rotation
    # car_rotation_grad = (180 / :math.pi) * car_rotation_rad
    # IO.inspect car_rotation_rad, label: 'Rotação do carro Radius'
    # IO.inspect car_rotation_grad, label: 'Rotação do carro Gradus'

    # Rotacionar o Vetor Velocidade
    # vector_velocity_rotated_x = vector_velocity_x * cos(car_rotation_grad) - vector_velocity_y * sin(car_rotation_grad);
    # vector_velocity_rotated_y = vector_velocity_x * sin(car_rotation_grad) + vector_velocity_y * cos(car_rotation_grad);
    # IO.inspect {vector_velocity_rotated_x, vector_velocity_rotated_y}, label: 'Vetor velocidade rotacionado'

    # Identifica se carro esta na areia ou não
    signal_sensor_1 = 0
    signal_sensor_2 = 0
    signal_sensor_3 = 0

    #Deep Learning AQUI retorna a action
    action =
      state.neural_network_pid
      |> Network.predict([0, 0, 0, orientation, -orientation])
      |> IO.inspect

    state =
      state
      |> Car.update_rotation(action)

    state =
      if rem(frame_count, 2) == 0 do
        state |> Car.move
      else
        state
      end

    graph =
      state.graph
      |> draw_objects(state.objects)

    sensor_left = Graph.get(graph, :sensor_left)
    sensor_center = Graph.get(graph, :sensor_center)
    sensor_right = Graph.get(graph, :sensor_right)

    %{transforms: %{translate: sensor_left}} = List.first(sensor_left)
    %{transforms: %{translate: sensor_center}} = List.first(sensor_center)
    %{transforms: %{translate: sensor_right}} = List.first(sensor_right)

    new_state =
      state
      |> update_in([:frame_count], &(&1 + 1))
      |> put_in([:objects, :car, :sensor, :center], sensor_center)
      |> put_in([:objects, :car, :sensor, :right], sensor_right)
      |> put_in([:objects, :car, :sensor, :left], sensor_left)

    # IO.inspect ' --------> STATE <----------'
    # IO.inspect new_state.objects

    {:noreply, new_state, push: graph}
  end

  # # Keyboard controls
  # def handle_input({:key, {"left", :press, _}}, _context, state) do
  #   {:noreply, Car.update_rotation(state, 1)}
  # end
  #
  # def handle_input({:key, {"right", :press, _}}, _context, state) do
  #   {:noreply, Car.update_rotation(state, 2)}
  # end
  #
  # def handle_input({:key, {"up", :press, _}}, _context, state) do
  #   {:noreply, Car.update_rotation(state, 0)}
  # end
  #
  # def handle_input(_input, _context, state), do: {:noreply, state}

  defp draw_objects(graph, object_map) do
    Enum.reduce(object_map, graph, fn {object_type, object_data}, graph ->
      draw_object(graph, object_type, object_data)
    end)
  end

  defp draw_object(graph, :goal, data) do
    %{coords: coords} = data
    graph
    |> circle(10, fill: :green, translate: coords)
  end

  defp draw_object(graph, :car, data) do
    {sensor_center_x, sensor_center_y} = data.sensor.center
    {sensor_right_x, sensor_right_y} = data.sensor.right
    {sensor_left_x, sensor_left_y} = data.sensor.left

    %{width: width, height: height} = data.dimension

    {x, y} = data.coords

    angle_radians = data.angle |> Vector2.degrees_to_radians

    # {x,y} = Scenic.Math.Vector2.add(data.coords, data.velocity)
    # IO.inspect data, label: 'data-->'
    # IO.inspect data.velocity, label: 'data.velocity-->'
    # IO.inspect {x,y}, label: '{x,y}-->'

    new_graph =
      graph
      |> group(fn(g) ->
        g
        |> rect({width, height}, [fill: :white, translate: {x, y}])
        |> circle(4, fill: :red, translate: {x + 22, y - 5}, id: :sensor_left)
        |> circle(4, fill: :green, translate: {x + 28, y + 5}, id: :sensor_center)
        |> circle(4, fill: :blue, translate: {x + 22, y + 15}, id: :sensor_right)
      end, rotate: angle_radians, pin: {x, y}, id: :car)
  end
end
