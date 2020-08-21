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

    # start memory
    {:ok, memory_pid} = Memory.start_link()

    state = %{
      viewport: viewport,
      viewport_width: viewport_width,
      viewport_height: viewport_height,
      graph: graph,
      frame_count: 0,
      neural_network_pid: neural_network_pid,
      memory_pid: memory_pid,
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
          last_action: 0,
          orientation: 0,
          orientation_rad: 0,
          orientation_grad: 0
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
    last_reward = state.objects.car.last_reward
    # IO.inspect {car_last_reward}, label: 'Última recompensa do carro'

    # Última orientação do carro
    orientation = state.objects.car.orientation
    # IO.inspect {orientation}, label: 'Última orientação do carro'

    # Posição atual do objetivo
    {goal_x, goal_y} = state.objects.goal.coords
    # IO.inspect {goal_x, goal_y}, label: 'Posição do Objetivo'

    # Vetor1 - Posição atual do carro menos objetivo
    vector1 = Scenic.Math.Vector2.sub(state.objects.car.coords, state.objects.goal.coords)
    # IO.inspect vector1, label: 'Vector1'

    # Vetor2 - Posição do objetivo - posicao anterior do carro
    vector2 = Scenic.Math.Vector2.sub(state.objects.car.sensor.center, state.objects.car.coords)
    # IO.inspect vector2, label: 'Vector2'

    # Normaliza vetor1
    vector1_normalized = Scenic.Math.Vector2.normalize(vector1)
    # IO.inspect vector1_normalized, label: 'Vector1 Normalizado'

    # Normaliza vetor2
    vector2_normalized = Scenic.Math.Vector2.normalize(vector2)
    # IO.inspect vector2_normalized, label: 'Vector2 Normalizado'

    # Orientação
    orientation = Scenic.Math.Vector2.dot(vector1_normalized, vector2_normalized)
    # IO.inspect orientation, label: 'Orientation'

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

    #Deep Learning AQUI retorna a action
    prob_actions =
      state.neural_network_pid
      |> Network.predict([0, 0, 0, orientation, -orientation])
    # IO.inspect prob_actions, label: 'prob_actions -->'

    action =
      prob_actions |> Enum.find_index(fn value -> Enum.max(prob_actions) == value end)
    # IO.inspect action, label: 'action -->'

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

    # Identifica se carro esta na areia ou não
    signal_center = 0
    signal_right = 0
    signal_left = 0

    #reward
    reward =
      case orientation do
        _orientation when _orientation > orientation -> 1
        _orientation when _orientation < orientation -> -1
        _ -> 0
      end

    Memory.push(state.memory_pid, %{
      last_signal_center: signal_center,
      last_signal_right: signal_right,
      last_signal_left: signal_left,
      orientation_pos: orientation,
      orientation_neg: -orientation,
      last_reward: reward,
      last_action: state.objects.car.last_action,
      new_action: action
    })

    new_state =
      state
      |> update_in([:frame_count], &(&1 + 1))
      |> put_in([:objects, :car, :sensor, :center], sensor_center)
      |> put_in([:objects, :car, :sensor, :right], sensor_right)
      |> put_in([:objects, :car, :sensor, :left], sensor_left)
      |> put_in([:objects, :car, :signal, :center], signal_center)
      |> put_in([:objects, :car, :signal, :right], signal_right)
      |> put_in([:objects, :car, :signal, :left], signal_left)
      |> put_in([:objects, :car, :last_reward], reward)
      |> put_in([:objects, :car, :orientation], orientation)
      |> put_in([:objects, :car, :orientation_rad], orientation_rad)
      |> put_in([:objects, :car, :orientation_grad], orientation_grad)
      |> put_in([:objects, :car, :last_action], action)

    # IO.inspect ' --------> STATE <----------'
    # IO.inspect new_state.objects
    memory_count = Memory.count(state.memory_pid)

    if memory_count == 20 do
      state.memory_pid
      |> Memory.list()
      |> Enum.shuffle
      |> Enum.each(fn params -> learn_with_memories(state.neural_network_pid, params) end)

      state.memory_pid |> Memory.reset()
    end

    IO.inspect new_state

    # require IEx
    # IEx.pry()

    {:noreply, new_state, push: graph}
  end

  defp learn_with_memories(neural_network_pid,
                           %{ last_signal_center: m_last_signal_center,
                              last_signal_right: m_last_signal_right,
                              last_signal_left: m_last_signal_left,
                              orientation_pos: m_orientation_pos,
                              orientation_neg: m_orientation_neg,
                              last_reward: m_last_reward,
                              last_action: m_last_action,
                              new_action: m_new_action}) do
    last_action_predict = m_last_action
    new_action_predict = neural_network_pid |> Network.forward([0, 0, 0, m_orientation_pos, m_orientation_neg])
    x = Network.get_output_data(neural_network_pid)
    gamma = 0.9
    target = Enum.map(x, fn x1 -> gamma * x1 + m_last_reward  end)
    neural_network_pid |> Network.backward(target)
  end


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
