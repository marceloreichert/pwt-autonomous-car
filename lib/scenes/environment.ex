defmodule AutonomousCar.Scene.Environment do
  use Scenic.Scene

  alias Scenic.Graph
  alias Scenic.ViewPort

  alias AutonomousCar.Math.Vector2
  alias AutonomousCar.Objects.Car
  alias AutonomousCar.NeuralNetwork.{Brain,LossFunctions,Model}

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

    # Start neural network
    {:ok, model_pid} = Model.start_link()

    # Init model params
    Model.push(Brain.init_params(), model_pid)

    # start memory
    {:ok, memory_pid} = Memory.start_link()

    state = %{
      viewport: viewport,
      viewport_width: viewport_width,
      viewport_height: viewport_height,
      graph: graph,
      frame_count: 0,
      model_pid: model_pid,
      memory_pid: memory_pid,
      last_distance: Vector2.distance({pos_x, pos_y}, {20, 20}),
      last_reward: 0,
      objects: %{
        car: %{
          dimension: %{width: 20, height: 10},
          coords: {pos_x, pos_y},
          last_coords: {pos_x, pos_y},
          velocity: {6, 0},
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
          last_action: 0,
          last_prob_actions: [0, 0, 0],
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

    IO.inspect ' --------> START PROCESS <----------'

    # Posição objetivo
    {goal_x, goal_y} = state.objects.goal.coords
    # IO.inspect {goal_x, goal_y}, label: 'Posição do Objetivo'

    # Posição do carro
    {car_x, car_y} = state.objects.car.coords
    # IO.inspect {car_x, car_y}, label: 'Posição do carro'

    # Última orientação do carro
    last_orientation = state.objects.car.orientation
    last_reward = state.last_reward
    last_distance = state.last_distance

    vector_car_velocity = state.objects.car.velocity
    vector_between_car_and_goal = Scenic.Math.Vector2.sub(state.objects.goal.coords, state.objects.car.coords)

    vector_car_velocity_normalized = Scenic.Math.Vector2.normalize(vector_car_velocity)
    vector_between_car_and_goal_normalized = Scenic.Math.Vector2.normalize(vector_between_car_and_goal)

    orientation = Scenic.Math.Vector2.dot(vector_car_velocity_normalized, vector_between_car_and_goal_normalized)
    orientation_rad = Math.acos(orientation)
    orientation_grad = (180 / :math.pi) * orientation_rad

    prob_actions =
      Model.pull(state.model_pid)
      |> Brain.predict(Nx.tensor([0, 0, 0, orientation, -orientation], names: [:data]))
      |> Nx.to_flat_list()

    action =
      prob_actions
      |> Enum.find_index(fn value -> Enum.max(prob_actions) == value end)

    state =
      state
      |> Car.update_angle(action)

    state =
      if rem(frame_count, 2) == 0 do
        state |> Car.move
      else
        state
      end

    graph =
      state.graph
      |> draw_objects(state.objects)

    # -------------------------------------------
    sensor_left = Graph.get(graph, :sensor_left)
    sensor_center = Graph.get(graph, :sensor_center)
    sensor_right = Graph.get(graph, :sensor_right)

    %{transforms: %{translate: sensor_left}} = List.first(sensor_left)
    %{transforms: %{translate: sensor_center}} = List.first(sensor_center)
    %{transforms: %{translate: sensor_right}} = List.first(sensor_right)

    distance = Vector2.distance(state.objects.car.coords, state.objects.goal.coords)
    reward = -0.2

    if distance < last_distance do
      reward = 0.1 # ganha uma pequena recompensa positiva
    end

    if car_x < 10 do
      reward = -1
    end

    if car_x > state.viewport_width - 10 do
      reward = -1
    end

    if car_y < 10 do
      reward = -1 # ganha recompensa negativa
    end

    if car_y > state.viewport_height - 10 do
      reward = -1
    end

    Memory.push(state.memory_pid, %{
      reward: reward,
      last_action: state.objects.car.last_action,
      last_prob_actions: state.objects.car.last_prob_actions,
      action: action,
      prob_actions: prob_actions
    })

    new_state =
      state
      |> update_in([:frame_count], &(&1 + 1))
      |> put_in([:reward], reward)
      |> put_in([:objects, :car, :sensor, :center], sensor_center)
      |> put_in([:objects, :car, :sensor, :right], sensor_right)
      |> put_in([:objects, :car, :sensor, :left], sensor_left)
      |> put_in([:objects, :car, :orientation], orientation)
      |> put_in([:objects, :car, :orientation_rad], orientation_rad)
      |> put_in([:objects, :car, :orientation_grad], orientation_grad)
      |> put_in([:objects, :car, :last_action], action)
      |> put_in([:objects, :car, :last_prob_actions], prob_actions)
      |> put_in([:objects, :car, :last_distance], distance)

    # IO.inspect ' --------> STATE <----------'
    # IO.inspect new_state.objects
    memory_count = Memory.count(state.memory_pid)

    if memory_count == 10 do
      memories =
        state.memory_pid
        |> Memory.list()
        |> Enum.shuffle

      target_tensor =
        memories
        |> target_in_list()
        |> Nx.tensor()

      predict_tensor =
        memories
        |> predict_in_list()
        |> Nx.tensor()

      # loss = AutonomousCar.NeuralNetwork.LossFunctions.loss(predict_tensor, target_tensor)

      old_params = Model.pull(state.model_pid)
      new_params =
        Enum.reduce(memories, Model.pull(state.model_pid), fn mem, acc ->
          require IEx
          IEx.pry()
          AutonomousCar.NeuralNetwork.Brain.update_weights(acc, predict_tensor, target_tensor)
        end)

      Model.push(state.model_pid, new_params)

      state.memory_pid |> Memory.reset()
    end

    IO.inspect(new_state, label: "NEW_STATE ---> ")
    IO.inspect ' --------> END PROCESS <----------'

    {:noreply, new_state, push: graph}
  end

  defp target_in_list(memories) do
    Enum.reduce(memories, [], fn mem, v ->
      [0.9 * Enum.at(mem[:prob_actions], mem[:action]) + mem[:reward] | v]
    end)
  end

  defp predict_in_list(memories) do
    Enum.reduce(memories, [], fn mem, acc ->
      [Enum.at(mem[:prob_actions], mem[:action]) | acc]
    end)
  end

  defp learn_with_memories(model_pid,
                           %{ reward: m_reward,
                              last_action: m_last_action,
                              action: m_action,
                              last_prob_actions: m_last_prob_actions,
                              prob_actions: m_prob_actions} = params) do

    action = Enum.at(m_prob_actions, m_action)
    target = 0.9 * action + m_reward
require IEx
IEx.pry()
    if is_list(m_last_prob_actions) do
      y = Enum.at(m_last_prob_actions, m_last_action)

      # td_loss = LossFunctions.smooth_l1_loss(y, target)
      #
      # Model.pull(model_pid)
      # |> Brain.update_weights(Nx.tensor(td_loss))
      # |> Model.push(model_pid)

    end
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
