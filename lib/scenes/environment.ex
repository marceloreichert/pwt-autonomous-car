defmodule AutonomousCar.Scene.Environment do
  use Scenic.Scene

  alias Scenic.Graph
  alias Scenic.ViewPort

  alias AutonomousCar.Math.Vector2

  alias AutonomousCar.Objects.Car
  alias AutonomousCar.NeuralNetwork.Model

  import Scenic.Primitives
  require Axon
  import Nx.Defn

  @batch_size 100

  # Initial parameters for the game scene!
  def init(_arg, opts) do
    viewport = opts[:viewport]

    # Initializes the graph
    graph = Graph.build(theme: :dark)

    # Calculate the transform that centers the car in the viewport
    {:ok, %ViewPort.Status{size: {viewport_width, viewport_height}}} = ViewPort.info(viewport)

    # start a very simple animation timer
    {:ok, timer} = :timer.send_interval(60, :frame)

    # Start neural network
    {:ok, model_pid} = Model.start_link()

    # start memory
    {:ok, memory_pid} = Memory.start_link()

    # Init model params
    Model.model
    |> Axon.init()
    |> Model.push(model_pid)

    car_coords = {trunc(viewport_width / 2), trunc(viewport_height / 2)}
    goal_coords = {20,20}
    car_velocity = {6,0}
    vector_car_velocity_normalized = Scenic.Math.Vector2.normalize(car_velocity)

    vector_between_car_and_goal = Scenic.Math.Vector2.sub(goal_coords, car_coords)
    vector_between_car_and_goal_normalized = Scenic.Math.Vector2.normalize(vector_between_car_and_goal)

    orientation = Scenic.Math.Vector2.dot(vector_car_velocity_normalized, vector_between_car_and_goal_normalized)
    orientation_rad = Math.acos(orientation)
    orientation_grad = (180 / :math.pi) * orientation_rad

    state = %{
      viewport: viewport,
      viewport_width: viewport_width,
      viewport_height: viewport_height,
      graph: graph,
      frame_count: 0,
      model_pid: model_pid,
      model_fit: false,
      memory_pid: memory_pid,
      distance: Scenic.Math.Vector2.distance(car_coords, goal_coords),
      action: 0,
      reward: 0,
      objects: %{
        goal: %{coords: goal_coords},
        car: %{
          dimension: %{width: 20, height: 10},
          coords: car_coords,
          velocity: car_velocity,
          angle: 0,
          orientation: orientation,
          orientation_rad: orientation_rad,
          orientation_grad: orientation_grad,
          signal: %{
            left: 0,
            center: 0,
            right: 0
          }
        }
      }
    }

    graph =
      graph
      |> draw_objects(state.objects)
      |> draw_vector(car_coords, goal_coords, :blue)

    state = state |> put_in([:graph], graph)

    {:ok, state, push: graph}
  end

  def handle_info(:frame, %{frame_count: frame_count} = state) do
    sensor_center = Graph.get(state.graph, :sensor_center)
    %{transforms: %{translate: sensor_center}} = List.first(sensor_center)

    car_object = Graph.get(state.graph, :car)
    %{transforms: %{rotate: car_rotate}} = List.first(car_object)

    car_look_goal = Graph.get(state.graph, :base)
    %{data: {car_look_goal_from, car_look_goal_to}} = List.first(car_look_goal)

    car_look_forward = Graph.get(state.graph, :velocity)
    %{data: {car_look_forward_from, car_look_forward_to}} = List.first(car_look_forward)

    v_car_look_goal = Scenic.Math.Vector2.sub(car_look_goal_from, car_look_goal_to)
    v_car_look_goal_normalized = Scenic.Math.Vector2.normalize(v_car_look_goal)

    v_car_look_forward = Scenic.Math.Vector2.sub(car_look_forward_from, car_look_forward_to)
    v_car_look_forward_rotate = AutonomousCar.Math.Vector2.rotate(v_car_look_forward, state.objects.car.angle)
    v_car_look_forward_normalized = Scenic.Math.Vector2.normalize(v_car_look_forward_rotate)

    orientation = Scenic.Math.Vector2.dot(v_car_look_goal_normalized, v_car_look_forward_normalized)
    orientation_rad = Math.acos(orientation)
    orientation_grad = (180 / :math.pi) * orientation_rad

    state_actual = [0,0,0, state.objects.car.orientation, -state.objects.car.orientation]
    distance_current = state.distance

    Memory.push(state.memory_pid, %{
      state_actual: state_actual,
      action: state.action,
      reward: state.reward,
      done: false,
      state_prime: [0,0,0, orientation, -orientation]
    })

    prob_actions =
      case state.model_fit do
        true ->
          inputs = Nx.tensor(state_actual) |> Nx.new_axis(0)
          params = state.model_pid |> Model.pull
          Model.model |> Axon.predict(params, Nx.tensor(state_actual))
        _ ->
          Nx.random_uniform({3})
    end

    action = Nx.argmax(prob_actions) |> Nx.to_scalar

    state = state |> Car.update_angle(action)

    state =
      if rem(frame_count, 1) == 0 do
        state |> Car.move
      else
        state
      end


    {car_x, car_y} = state.objects.car.coords
    distance = Scenic.Math.Vector2.distance(state.objects.car.coords, state.objects.goal.coords)

    reward =
      cond do
        distance < distance_current ->
          1
        car_x < 10 ->
          -0.9
        car_x > state.viewport_width - 10 ->
          -0.9
        car_y < 10 ->
          -0.9
        car_y > state.viewport_height - 10 ->
          -0.9
        true ->
          -0.4
      end

    memory_count = Memory.count(state.memory_pid)

    model_fit =
      cond do
        state.model_fit -> true
        memory_count == @batch_size -> true
        true -> false
      end

    # ----------------------------------------------
    graph =
      Graph.build(theme: :dark)
      |> draw_objects(state.objects)
      |> draw_vector(sensor_center, state.objects.goal.coords, :blue)
      |> draw_model_fit(state.model_fit)
    # ----------------------------------------------

    if memory_count == @batch_size do
      memories =
        state.memory_pid
        |> Memory.list()
        |> Enum.shuffle

      train_samples = memories
        |> Enum.map(fn x -> x.state_actual end)
        |> Nx.tensor()
        |> Nx.to_batched_list(@batch_size)

      train_labels = gen_data_labels(memories, state)
        |> Nx.tensor()
        |> Nx.to_batched_list(@batch_size)

      params = Model.pull(state.model_pid)
      {new_params, _} =
        Model.model
        |> Axon.Training.step(:binary_cross_entropy, Axon.Optimizers.sgd(0.01))
        |> Axon.Training.train(train_samples, train_labels, epochs: 1)

      Model.push(new_params, state.model_pid)

      state.memory_pid |> Memory.reset()
    end

    new_state =
      state
      |> update_in([:frame_count], &(&1 + 1))
      |> put_in([:objects, :car, :orientation], orientation)
      |> put_in([:objects, :car, :orientation_rad], orientation_rad)
      |> put_in([:objects, :car, :orientation_grad], orientation_grad)
      |> put_in([:action], action)
      |> put_in([:distance], distance)
      |> put_in([:reward], reward)
      |> put_in([:model_fit], model_fit)
      |> put_in([:graph], graph)

    {:noreply, new_state, push: graph}
  end

  defp gen_data_labels([], _), do: []

  defp gen_data_labels([xp | samples], state) do
    v = get_values(xp.state_actual, state) |> Nx.to_flat_list()
    vr = if xp.done, do: xp.reward, else: calc_r(xp.reward, 0.99, get_values(xp.state_prime, state))

    labels = List.replace_at(v, xp.action, Nx.to_scalar(vr))
    [labels | gen_data_labels(samples, state)]
  end

  # defp get_values(_s, state) do
  #   Nx.random_uniform({3})
  # end

  defp get_values(s, state) do
    inputs = Nx.tensor(s) |> Nx.new_axis(0)

    params = state.model_pid |> Model.pull
    Model.model |> Axon.predict(params, inputs)
  end

  defn calc_r(r, gamma, values) do
    r + (gamma * Nx.reduce_max(values))
  end

  defp draw_vector(graph, from, to, color) do
    graph |> line( {from, to}, stroke: {4, color}, cap: :round, id: :base )
  end

  defp draw_model_fit(graph, model_fit) do
    if model_fit do
      graph |> circle(5, fill: :green, translate: {10,10})
    else
      graph |> circle(5, fill: :red, translate: {10,10})
    end
  end

  defp draw_objects(graph, object_map) do
    Enum.reduce(object_map, graph, fn {object_type, object_data}, graph ->
      draw_object(graph, object_type, object_data)
    end)
  end

  defp draw_object(graph, :goal, data) do
    %{coords: coords} = data
    graph |> circle(10, fill: :yellow, translate: coords)
  end

  defp draw_object(graph, :car, data) do
    %{width: width, height: height} = data.dimension

    {x, y} = data.coords

    angle_radians = data.angle |> degrees_to_radians

    new_graph =
      graph
      |> group(fn(g) ->
        g
        |> rect({width, height}, [fill: :white, translate: {x, y}])
        |> circle(4, fill: :red, translate: {x + 22, y - 5}, id: :sensor_left)
        |> circle(4, fill: :green, translate: {x + 28, y + 5}, id: :sensor_center)
        |> circle(4, fill: :blue, translate: {x + 22, y + 15}, id: :sensor_right)
        |> line({ {x + 28, y + 5}, {x + 28 + 10, y + 5} }, stroke: {3, :blue}, cap: :round, id: :velocity )
      end, rotate: angle_radians, pin: {x, y}, id: :car)
  end

  defp degrees_to_radians(angle) do
    angle * (Math.pi / 180)
  end
end
