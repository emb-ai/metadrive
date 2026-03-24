"""
PlanT/HFLM policy for MetaDrive: wraps HFLM (plant2/PlanT/model.py), builds input from env state, forward, waypoints -> steer/throttle.
Requires engine.global_config["plant_checkpoint_dir"] (dir with .ckpt) and ["plant_carla_garage_path"] (path to plant2/carla_garage).
Loads first .ckpt in checkpoint_dir.
"""
import os
import sys
import gymnasium as gym
import numpy as np

from metadrive.policy.base_policy import BasePolicy

_plant_module_loaded = False


def _ensure_plant_import(carla_garage_path):
    global _plant_module_loaded
    import unittest.mock as _mock
    if _plant_module_loaded:
        return
    carla_garage_path = os.path.abspath(carla_garage_path)
    if carla_garage_path not in sys.path:
        sys.path.insert(0, carla_garage_path)
    orig_cwd = os.getcwd()
    try:
        os.chdir(carla_garage_path)
        if "carla" not in sys.modules:
            class _CarlaColor:
                def __init__(self, r=0, g=0, b=0, a=1.0):
                    pass
            _carla = type(sys)("carla")
            _carla.Color = _CarlaColor
            sys.modules["carla"] = _carla
        if "agents.navigation.global_route_planner" not in sys.modules:
            sys.modules["agents"] = _mock.MagicMock()
            sys.modules["agents.navigation"] = _mock.MagicMock()
            sys.modules["agents.navigation.global_route_planner"] = _mock.MagicMock()
    finally:
        os.chdir(orig_cwd)
    _plant_module_loaded = True

####VIKA'S CHANGE
def _route_lane_index_for_segment(lane_list, ego_vehicle, ego_pos_xy):
    """
    Which lane in this road segment matches the agent — avoid always using lane_list[0].
    """
    ego_lane = getattr(ego_vehicle, "lane", None)
    if ego_lane is not None and ego_lane in lane_list:
        return lane_list.index(ego_lane)
    best_i = 0
    best_lat = float("inf")
    for i, lane in enumerate(lane_list):
        try:
            _, lat = lane.local_coordinates(ego_pos_xy)
        except Exception:
            lat = float("inf")
        alat = abs(float(lat))
        if alat < best_lat:
            best_lat = alat
            best_i = i
    return best_i


def _lanes_ahead_from_checkpoints(nav, lane_idx):
    """One centerline polyline lane per route segment; preserve lateral lane index across segments."""
    ckpt_idx = getattr(nav, "_target_checkpoints_index", None)
    ckpts = getattr(nav, "checkpoints", None) or []
    map_net = getattr(nav, "map", None)
    if ckpt_idx is None or map_net is None or not hasattr(map_net, "road_network"):
        return []
    graph = map_net.road_network.graph
    try:
        start_i = max(0, int(ckpt_idx[0]) if hasattr(ckpt_idx, "__getitem__") else 0)
    except Exception:
        start_i = 0
    out = []
    for i in range(start_i, len(ckpts) - 1):
        sn, en = ckpts[i], ckpts[i + 1]
        if sn in graph and en in graph[sn]:
            lane_list = graph[sn][en]
            if lane_list:
                li = min(int(lane_idx), len(lane_list) - 1)
                out.append(lane_list[li])
    return out


def get_route_points_ego_frame(ego_vehicle, num_points=20, step_m=1.0):
    """
    Sample route densely (every step_m meters) starting from ego's current position
    along the navigation lanes. Returns (route_np (N,2), target_point (2,));
    x=forward, y=right (CARLA convention).
    """
    nav = getattr(ego_vehicle, "navigation", None)
    if nav is None or not hasattr(nav, "current_ref_lanes") or not nav.current_ref_lanes:
        route = np.zeros((num_points, 2), dtype=np.float32)
        for i in range(num_points):
            route[i, 0] = (i + 1) * step_m
        target_point = np.array([step_m, 0.0], dtype=np.float32)
        return route, target_point

    ego_pos = np.array(ego_vehicle.position[:2], dtype=np.float64)

    ref_lanes = nav.current_ref_lanes
    lane_idx = _route_lane_index_for_segment(list(ref_lanes), ego_vehicle, ego_pos)

    lanes_ahead = _lanes_ahead_from_checkpoints(nav, lane_idx)

    # Do NOT use list(current_ref_lanes): those are parallel lanes on one block, not a polyline.
    if not lanes_ahead:
        ego_lane = getattr(ego_vehicle, "lane", None)
        if ego_lane is not None and ego_lane in ref_lanes:
            lanes_ahead = [ego_lane]
        else:
            lanes_ahead = [ref_lanes[min(lane_idx, len(ref_lanes) - 1)]]


    #  ego's longitudinal position 
    lane_start = 0
    ego_s = 0.0
    for li, lane in enumerate(lanes_ahead):
        s_on_lane = lane.local_coordinates(ego_pos)[0]
        if s_on_lane < lane.length - 0.5:
            ego_s = float(np.clip(s_on_lane, 0.0, lane.length))
            lane_start = li
            break
    else:
        # last lane's end
        lane_start = max(0, len(lanes_ahead) - 1)
        ego_s = lanes_ahead[lane_start].length - step_m

    #  step_m intervals from ego_s
    points_world = []
    remaining = num_points
    current_s = ego_s + step_m  # start 1m ahead of ego

    for lane in lanes_ahead[lane_start:]:
        if remaining <= 0:
            break
        s = current_s
        while s <= lane.length and remaining > 0:
            pt = lane.position(s, 0)  # center of lane
            points_world.append(np.array(pt[:2], dtype=np.float64))
            remaining -= 1
            s += step_m
        current_s = s - lane.length  # carry over into next lane
        if current_s < step_m * 0.5:
            current_s = step_m * 0.5  # avoid starting at 0 of next lane

    # Pad with last point if not enough
    if len(points_world) < num_points:
        if points_world:
            last = points_world[-1]
        else:
            heading = float(ego_vehicle.heading_theta)
            last = ego_pos + np.array([np.cos(heading), np.sin(heading)]) * step_m
            points_world.append(last)
        while len(points_world) < num_points:
            points_world.append(last.copy())

    route_ego = np.zeros((num_points, 2), dtype=np.float32)
    for i, pt in enumerate(points_world[:num_points]):
        diff = np.array([pt[0] - ego_pos[0], pt[1] - ego_pos[1]])
        local = ego_vehicle.convert_to_local_coordinates(diff, 0.0)
        route_ego[i, 0] = float(local[0])      
        route_ego[i, 1] = -float(local[1])        
    target_point = route_ego[0] if len(route_ego) else np.array([step_m, 0.0], dtype=np.float32)
    return route_ego, target_point


class PlanTPolicy(BasePolicy):
    DEBUG_MARK_COLOR = (0, 200, 0, 255)

    def __init__(self, control_object, random_seed=None, config=None):
        super(PlanTPolicy, self).__init__(control_object=control_object, random_seed=random_seed, config=config or {})
        self._net = None
        self._config_all = None
        self._device = None
        self._num_route_points = 20
        self._max_objects = 30
        self._speed_cats = {50: 0, 80: 1, 100: 2, 120: 3}
        self._shape_logged = False

    def _load_plant(self):
        if self._net is not None:
            return
        engine = self.engine
        global_config = engine.global_config if engine else {}
        checkpoint_dir = global_config.get("plant_checkpoint_dir")
        carla_garage_path = global_config.get("plant_carla_garage_path")
        if not checkpoint_dir or not carla_garage_path:
            raise ValueError(
                "PlanTPolicy requires global_config['plant_checkpoint_dir'] and global_config['plant_carla_garage_path']"
            )
        _ensure_plant_import(carla_garage_path)
        import torch
        from run_plant_inference import load_hflm_model
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._net, self._config_all = load_hflm_model(checkpoint_dir, self._device)

    def act(self, agent_id):
        self._load_plant()
        import torch
        carla_garage_path = self.engine.global_config.get("plant_carla_garage_path", "")
        if carla_garage_path and carla_garage_path not in sys.path:
            sys.path.insert(0, carla_garage_path)
        from metadrive.policy.metadrive_obs_to_plant2 import metadrive_obs_to_plant2_batch
        from plant2_control import plant2_predictions_to_action, get_target_speed_from_limit

        ego = self.control_object
        ego_speed = float(ego.speed)
        training = getattr(self._config_all.model, "training", {}) or {}
        input_bev = training.get("input_bev", False)
        input_ego_speed = training.get("input_ego_speed", False)

        batch = metadrive_obs_to_plant2_batch(
            self.engine,
            ego,
            route_ego_20x2=None,
            speed_limit_kmh=None,
            max_objects=self._max_objects,
            max_distance=50.0,
            range_factor_front=2.0,
            input_bev=input_bev,
            input_ego_speed=input_ego_speed,
            device=self._device,
        )
        with torch.no_grad():
            logits, targets, pred_plan, _ = self._net(batch)
        speed_limit_idx = int(batch["speed_limit"][0].item())
        target_speed_mps = get_target_speed_from_limit(speed_limit_idx, (50, 80, 100, 120))
        action = plant2_predictions_to_action(
            pred_plan,
            current_speed=ego_speed,
            target_speed_mps=target_speed_mps,
            speed_limit_idx=speed_limit_idx,
            speed_limits_kmh=(50, 80, 100, 120),
            device=self._device,
        )
        if not self._shape_logged:
            import logging
            log = logging.getLogger(__name__)
            pp, pw, _ = pred_plan
            log.info("HFLM pred_path %s pred_wps %s", pp.shape if pp is not None else None, pw.shape if pw is not None else None)
            self._shape_logged = True
        self.action_info["action"] = action
        return action

    @classmethod
    def get_input_space(cls):
        return gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

# """
# PlanT/HFLM policy for MetaDrive: wraps HFLM (plant2/PlanT/model.py), builds input from env state, forward, waypoints -> steer/throttle.
# Requires engine.global_config["plant_checkpoint_dir"] (dir with .ckpt) and ["plant_carla_garage_path"] (path to plant2/carla_garage).
# Loads first .ckpt in checkpoint_dir.
# """
# import os
# import sys
# import gymnasium as gym
# import numpy as np

# from metadrive.policy.base_policy import BasePolicy

# _plant_module_loaded = False


# def _ensure_plant_import(carla_garage_path):
#     global _plant_module_loaded
#     import unittest.mock as _mock
#     if _plant_module_loaded:
#         return
#     carla_garage_path = os.path.abspath(carla_garage_path)
#     if carla_garage_path not in sys.path:
#         sys.path.insert(0, carla_garage_path)
#     orig_cwd = os.getcwd()
#     try:
#         os.chdir(carla_garage_path)
#         if "carla" not in sys.modules:
#             class _CarlaColor:
#                 def __init__(self, r=0, g=0, b=0, a=1.0):
#                     pass
#             _carla = type(sys)("carla")
#             _carla.Color = _CarlaColor
#             sys.modules["carla"] = _carla
#         if "agents.navigation.global_route_planner" not in sys.modules:
#             sys.modules["agents"] = _mock.MagicMock()
#             sys.modules["agents.navigation"] = _mock.MagicMock()
#             sys.modules["agents.navigation.global_route_planner"] = _mock.MagicMock()
#     finally:
#         os.chdir(orig_cwd)
#     _plant_module_loaded = True


# def get_route_points_ego_frame(ego_vehicle, num_points=20, step_m=1.0):
#     """
#     Sample route densely (every step_m meters) starting from ego's current position
#     along the navigation lanes. Returns (route_np (N,2), target_point (2,));
#     x=forward, y=right (CARLA convention).
#     """
#     nav = getattr(ego_vehicle, "navigation", None)
#     if nav is None or not hasattr(nav, "current_ref_lanes") or not nav.current_ref_lanes:
#         route = np.zeros((num_points, 2), dtype=np.float32)
#         for i in range(num_points):
#             route[i, 0] = (i + 1) * step_m
#         target_point = np.array([step_m, 0.0], dtype=np.float32)
#         return route, target_point

#     ego_pos = np.array(ego_vehicle.position[:2], dtype=np.float64)

#     # --- Collect ordered lanes starting from current checkpoint ---
#     lanes_ahead = []
#     try:
#         ckpt_idx = getattr(nav, "_target_checkpoints_index", None)
#         ckpts = getattr(nav, "checkpoints", [])
#         map_net = getattr(nav, "map", None)
#         if ckpt_idx is not None and map_net is not None and hasattr(map_net, "road_network"):
#             graph = map_net.road_network.graph
#             start_i = max(0, int(ckpt_idx[0]) if hasattr(ckpt_idx, '__getitem__') else 0)
#             for i in range(start_i, len(ckpts) - 1):
#                 sn, en = ckpts[i], ckpts[i + 1]
#                 if sn in graph and en in graph[sn]:
#                     lane_list = graph[sn][en]
#                     if lane_list:
#                         lanes_ahead.append(lane_list[0])
#     except Exception:
#         pass

#     # Fallback: use current_ref_lanes
#     if not lanes_ahead:
#         ref_lanes = nav.current_ref_lanes
#         if ref_lanes:
#             lanes_ahead = list(ref_lanes)

#     if not lanes_ahead:
#         # Last resort: straight ahead
#         route = np.zeros((num_points, 2), dtype=np.float32)
#         for i in range(num_points):
#             route[i, 0] = (i + 1) * step_m
#         return route, route[0]

#     # --- Find ego's longitudinal position on the first lane ---
#     first_lane = lanes_ahead[0]
#     ego_s = first_lane.local_coordinates(ego_pos)[0]
#     ego_s = float(np.clip(ego_s, 0.0, first_lane.length))

#     # --- Sample points at step_m intervals starting from ego_s ---
#     points_world = []
#     remaining = num_points
#     current_s = ego_s + step_m  # start 1m ahead of ego

#     for lane in lanes_ahead:
#         if remaining <= 0:
#             break
#         # For the first lane, start from current_s; for subsequent lanes, start from 0
#         s = current_s
#         while s <= lane.length and remaining > 0:
#             pt = lane.position(s, 0)  # center of lane
#             points_world.append(np.array(pt[:2], dtype=np.float64))
#             remaining -= 1
#             s += step_m
#         current_s = s - lane.length  # carry over into next lane
#         if current_s < step_m * 0.5:
#             current_s = step_m * 0.5  # avoid starting at 0 of next lane

#     # Pad with last point if not enough
#     if len(points_world) < num_points:
#         if points_world:
#             last = points_world[-1]
#         else:
#             # No points found, generate straight ahead
#             heading = float(ego_vehicle.heading_theta)
#             last = ego_pos + np.array([np.cos(heading), np.sin(heading)]) * step_m
#             points_world.append(last)
#         while len(points_world) < num_points:
#             points_world.append(last.copy())

#     # --- Convert to ego frame (CARLA convention: x=forward, y=right) ---
#     route_ego = np.zeros((num_points, 2), dtype=np.float32)
#     for i, pt in enumerate(points_world[:num_points]):
#         diff = np.array([pt[0] - ego_pos[0], pt[1] - ego_pos[1]])
#         local = ego_vehicle.convert_to_local_coordinates(diff, 0.0)
#         route_ego[i, 0] = float(local[0])        # forward
#         route_ego[i, 1] = -float(local[1])        # MetaDrive (fwd, LEFT) → CARLA (fwd, RIGHT)
#     target_point = route_ego[0] if len(route_ego) else np.array([step_m, 0.0], dtype=np.float32)
#     return route_ego, target_point


# class PlanTPolicy(BasePolicy):
#     DEBUG_MARK_COLOR = (0, 200, 0, 255)

#     def __init__(self, control_object, random_seed=None, config=None):
#         super(PlanTPolicy, self).__init__(control_object=control_object, random_seed=random_seed, config=config or {})
#         self._net = None
#         self._config_all = None
#         self._device = None
#         self._num_route_points = 20
#         self._max_objects = 30
#         self._speed_cats = {50: 0, 80: 1, 100: 2, 120: 3}
#         self._shape_logged = False

#     def _load_plant(self):
#         if self._net is not None:
#             return
#         engine = self.engine
#         global_config = engine.global_config if engine else {}
#         checkpoint_dir = global_config.get("plant_checkpoint_dir")
#         carla_garage_path = global_config.get("plant_carla_garage_path")
#         if not checkpoint_dir or not carla_garage_path:
#             raise ValueError(
#                 "PlanTPolicy requires global_config['plant_checkpoint_dir'] and global_config['plant_carla_garage_path']"
#             )
#         _ensure_plant_import(carla_garage_path)
#         import torch
#         from run_plant_inference import load_hflm_model
#         self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self._net, self._config_all = load_hflm_model(checkpoint_dir, self._device)

#     def act(self, agent_id):
#         self._load_plant()
#         import torch
#         carla_garage_path = self.engine.global_config.get("plant_carla_garage_path", "")
#         if carla_garage_path and carla_garage_path not in sys.path:
#             sys.path.insert(0, carla_garage_path)
#         from metadrive.policy.metadrive_obs_to_plant2 import metadrive_obs_to_plant2_batch
#         from plant2_control import plant2_predictions_to_action, get_target_speed_from_limit

#         ego = self.control_object
#         ego_speed = float(ego.speed)
#         training = getattr(self._config_all.model, "training", {}) or {}
#         input_bev = training.get("input_bev", False)
#         input_ego_speed = training.get("input_ego_speed", False)

#         batch = metadrive_obs_to_plant2_batch(
#             self.engine,
#             ego,
#             route_ego_20x2=None,
#             speed_limit_kmh=None,
#             max_objects=self._max_objects,
#             max_distance=50.0,
#             range_factor_front=2.0,
#             input_bev=input_bev,
#             input_ego_speed=input_ego_speed,
#             device=self._device,
#         )
#         with torch.no_grad():
#             logits, targets, pred_plan, _ = self._net(batch)
#         speed_limit_idx = int(batch["speed_limit"][0].item())
#         target_speed_mps = get_target_speed_from_limit(speed_limit_idx, (50, 80, 100, 120))
#         action = plant2_predictions_to_action(
#             pred_plan,
#             current_speed=ego_speed,
#             target_speed_mps=target_speed_mps,
#             speed_limit_idx=speed_limit_idx,
#             speed_limits_kmh=(50, 80, 100, 120),
#             device=self._device,
#         )
#         if not self._shape_logged:
#             import logging
#             log = logging.getLogger(__name__)
#             pp, pw, _ = pred_plan
#             log.info("HFLM pred_path %s pred_wps %s", pp.shape if pp is not None else None, pw.shape if pw is not None else None)
#             self._shape_logged = True
#         self.action_info["action"] = action
#         return action

#     @classmethod
#     def get_input_space(cls):
#         return gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
