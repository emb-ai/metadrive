"""
Convert MetaDrive obs/state into PlanT 2.0 input format (HFLM-compatible).

Mirrors PlanT/dataset.py::generate_batch and PlanTVariables.
"""
import os
import sys
import numpy as np

# PlanT object types (PlanTVariables.class_nums): 0=padding, 1=vehicle, 2=pedestrian, 3=static,
# 4=stop_sign, 5=traffic_light, 6=emergency
OBJ_TYPE_PADDING = 0
OBJ_TYPE_VEHICLE = 1
OBJ_TYPE_PEDESTRIAN = 2
OBJ_TYPE_STATIC = 3
OBJ_TYPE_STOP_SIGN = 4
OBJ_TYPE_TRAFFIC_LIGHT = 5
OBJ_TYPE_EMERGENCY = 6

# PlanT2 BEV semantic indices (PlanTVariables.bev_colors)
BEV_IDX_BACKGROUND = 0
BEV_IDX_STREET = 1
BEV_IDX_SIDEWALK = 2
BEV_IDX_ALL_LINES = 3   # solid
BEV_IDX_BROKEN_LINES = 4

# PlanTVariables.speed_cats
SPEED_CATS = {50: 0, 80: 1, 100: 2, 120: 3}


def _get_obj_type(obj):
    """Determine the PlanT object type from a MetaDrive instance."""
    from metadrive.component.vehicle.base_vehicle import BaseVehicle
    from metadrive.component.traffic_participants.base_traffic_participant import BaseTrafficParticipant
    from metadrive.component.traffic_participants.pedestrian import Pedestrian
    from metadrive.component.static_object.traffic_object import TrafficCone, TrafficBarrier, TrafficWarning
    from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
    from metadrive.constants import MetaDriveType

    if isinstance(obj, BaseVehicle):
        if getattr(obj, "break_down", False):
            return OBJ_TYPE_EMERGENCY
        return OBJ_TYPE_VEHICLE
    if isinstance(obj, Pedestrian):
        return OBJ_TYPE_PEDESTRIAN
    if isinstance(obj, BaseTrafficParticipant):
        return OBJ_TYPE_PEDESTRIAN
    if isinstance(obj, BaseTrafficLight):
        return OBJ_TYPE_TRAFFIC_LIGHT
    if isinstance(obj, (TrafficCone, TrafficBarrier, TrafficWarning)):
        return OBJ_TYPE_STATIC
    return OBJ_TYPE_STATIC  # safer fallback than vehicle


def _get_obj_extent(obj):
    """Get the object's extent (half-extents) in metres: (ext_x, ext_y)."""
    w = getattr(obj, "top_down_width", 2.0) or 2.0
    l = getattr(obj, "top_down_length", 4.0) or 4.0
    ext_x = float(l) / 2.0
    ext_y = float(w) / 2.0
    return max(0.1, ext_x), max(0.1, ext_y)


def _get_obj_speed_kmh(obj):
    """Object speed in km/h."""
    if hasattr(obj, "velocity"):
        v = obj.velocity
        if hasattr(v, "__len__"):
            return float(np.sqrt(v[0] ** 2 + v[1] ** 2)) * 3.6
        return float(v) * 3.6
    if hasattr(obj, "speed"):
        return float(obj.speed) * 3.6
    return 0.0


def _passes_distance_filter(x, y, obj_type, max_distance=50.0, range_factor_front=2.0, tl_stop_radius=30.0):
    """
    PlanT2: elliptical filter for regular objects, 30 m circle for TL/stop.
    x, y are in ego frame (x forward).
    """
    if obj_type in (OBJ_TYPE_TRAFFIC_LIGHT, OBJ_TYPE_STOP_SIGN):
        return x * x + y * y <= tl_stop_radius * tl_stop_radius
    x_div = range_factor_front * range_factor_front if x > 0 else 1.0
    return x * x / x_div + y * y <= max_distance * max_distance


def collect_objects_ego_frame(
    engine,
    ego_vehicle,
    max_objects=30,
    max_distance=50.0,
    range_factor_front=2.0,
    tl_stop_radius=30.0,
    include_stop_signs=True,
):
    """
    Collect objects in ego frame. PlanT2: include TL only when Red/Yellow (affects_ego); stop only when affects_ego.
    Elliptical filter: longer reach forward (range_factor_front).
    """
    from metadrive.utils.math import wrap_to_pi
    from metadrive.constants import MetaDriveType

    ego_pos = np.array(ego_vehicle.position[:2])
    ego_heading = float(ego_vehicle.heading_theta)

    objects = []

    def add_obj(obj, obj_type=None):
        if obj is ego_vehicle:
            return
        if not hasattr(obj, "position"):
            return
        rel = np.array([obj.position[0] - ego_pos[0], obj.position[1] - ego_pos[1]])
        local = ego_vehicle.convert_to_local_coordinates(rel, 0.0)
        x, y = float(local[0]), -float(local[1])  # negate y: MetaDrive (fwd, LEFT) → CARLA (fwd, RIGHT)
        t = obj_type if obj_type is not None else _get_obj_type(obj)

        # TL: only Red/Yellow (MetaDrive has no affects_ego — include if Red/Yellow)
        if t == OBJ_TYPE_TRAFFIC_LIGHT:
            status = getattr(obj, "status", MetaDriveType.LIGHT_UNKNOWN)
            if status not in (MetaDriveType.LIGHT_RED, MetaDriveType.LIGHT_YELLOW):
                return
        # Optional stop-sign exclusion for ablation/debug.
        if t == OBJ_TYPE_STOP_SIGN and not include_stop_signs:
            return

        if not _passes_distance_filter(x, y, t, max_distance, range_factor_front, tl_stop_radius):
            return

        yaw_deg = np.degrees(wrap_to_pi(float(obj.heading_theta) - ego_heading))
        speed_kmh = _get_obj_speed_kmh(obj)
        ext_x, ext_y = _get_obj_extent(obj)
        objects.append((t, x, y, yaw_deg, speed_kmh, ext_x, ext_y))

    vehicles = list(getattr(engine.traffic_manager, "vehicles", []))
    for v in vehicles:
        add_obj(v)

    if hasattr(engine, "get_objects"):
        for obj in engine.get_objects(lambda o: hasattr(o, "position")).values():
            if obj in vehicles or obj is ego_vehicle:
                continue
            add_obj(obj)

    if hasattr(engine, "object_manager") and engine.object_manager is not None:
        for obj in getattr(engine.object_manager, "spawned_objects", {}).values():
            add_obj(obj)

    objects.sort(key=lambda o: o[1] ** 2 + o[2] ** 2)
    return objects[:max_objects]


def objects_to_x_batch(objects_list, max_objects=30):
    """
    PlanT2 generate_batch: x_objs is a pool of shape (1+num_objs, 7); idxs is (B, maxseq) with zeros as padding.
    Row format: [type, x, y, yaw_deg, speed_kmh, extent_y*2, extent_x*2] (doorflag=0).
    """
    import torch

    x_list = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]  # padding at idx 0
    for (t, x, y, yaw_deg, speed_kmh, ext_x, ext_y) in objects_list:
        x_list.append([
            float(t), float(x), float(y), float(yaw_deg), float(speed_kmh),
            float(ext_y) * 2, float(ext_x) * 2
        ])
    num_objs = len(x_list) - 1
    return x_list, num_objs


def get_speed_limit_idx(speed_limit_kmh=None):
    """PlanTVariables.speed_cats. Defaults to 1 (80 km/h) when no limit is set."""
    if speed_limit_kmh is None:
        return 1
    return SPEED_CATS.get(speed_limit_kmh, 1)


# PlanTVariables.bev_colors — palette for the semantic BEV (imagenet-friendly)
BEV_COLORS = np.array([
    [0.485, 0.456, 0.406],  # 0 Background (Imagenet mean)
    [0.25, 0.25, 0.75],    # 1 Street
    [0.485, 0.456, 0.406],  # 2 Sidewalk (Imagenet mean)
    [0.75, 0.25, 0.25],    # 3 All lines (solid)
    [0.25, 0.75, 0.25],    # 4 Broken lines
], dtype=np.float32)


def render_bev_plant2(engine, ego_vehicle, resolution=128, size_meters=64.0,
                      device="cpu", return_semantic_map=False):
    """
    PlanT2 BEV: semantic index map (0-4) -> RGB via PlanTVariables.bev_colors.
    Supports both NodeRoadNetwork and EdgeRoadNetwork.

    Returns:
      - default: torch.Tensor (1, 3, H, W) float32 RGB
      - ``return_semantic_map=True``: uint8 ndarray (H, W) with class indices 0-4
        (what PlanTDataset expects under ``bev_no_car_semantics/*.png``)
    """
    import torch
    from metadrive.constants import PGLineType, MetaDriveType
    from metadrive.component.road_network.node_road_network import NodeRoadNetwork
    from metadrive.component.road_network.edge_road_network import EdgeRoadNetwork

    road_network = getattr(engine.current_map, "road_network", None)
    if road_network is None:
        return None

    scale = resolution / size_meters
    ego_pos = np.array(ego_vehicle.position[:2])
    ego_heading = float(ego_vehicle.heading_theta)

    def world_to_ego_xy(wx, wy):
        dx = wx - ego_pos[0]
        dy = wy - ego_pos[1]
        c, s = np.cos(-ego_heading), np.sin(-ego_heading)
        ex = dx * c - dy * s
        ey = dx * s + dy * c
        return ex, ey

    def ego_to_pix(ex, ey):
        px = resolution // 2 - int(ey * scale)   # negate ey: MetaDrive y=left, CARLA right=right in image
        py = resolution // 2 - int(ex * scale)
        return px, py

    sem_map = np.zeros((resolution, resolution), dtype=np.uint8)
    
    # Handle different road network types
    if isinstance(road_network, NodeRoadNetwork):
        # Original logic for NodeRoadNetwork (PG-based)
        for _from in road_network.graph.keys():
            for _to in road_network.graph[_from].keys():
                for lane in road_network.graph[_from][_to]:
                    try:
                        n_pts = max(50, int(lane.length))
                        for i in range(n_pts):
                            s = lane.length * i / max(n_pts - 1, 1)
                            pt = lane.position(s, 0)
                            ex, ey = world_to_ego_xy(pt[0], pt[1])
                            if abs(ex) > size_meters / 2 or abs(ey) > size_meters / 2:
                                continue
                            px, py = ego_to_pix(ex, ey)
                            w = max(1, int(lane.width_at(s) / 2 * scale))
                            for dw in range(-w, w + 1):
                                for dh in range(-w, w + 1):
                                    nx, ny = px + dw, py + dh
                                    if 0 <= nx < resolution and 0 <= ny < resolution:
                                        sem_map[ny, nx] = BEV_IDX_STREET

                        for side in range(2):
                            lt = lane.line_types[side]
                            idx = BEV_IDX_ALL_LINES if lt == PGLineType.CONTINUOUS else BEV_IDX_BROKEN_LINES
                            for i in range(n_pts):
                                s = lane.length * i / max(n_pts - 1, 1)
                                lat = (side - 0.5) * lane.width_at(s)
                                pt = lane.position(s, lat)
                                ex, ey = world_to_ego_xy(pt[0], pt[1])
                                if abs(ex) > size_meters / 2 or abs(ey) > size_meters / 2:
                                    continue
                                px, py = ego_to_pix(ex, ey)
                                if 0 <= px < resolution and 0 <= py < resolution:
                                    sem_map[py, px] = idx
                    except Exception:
                        pass
    
    elif isinstance(road_network, EdgeRoadNetwork):
        # Logic for EdgeRoadNetwork (SUMO-based)
        try:
            # Get map_data from the current map
            map_data = None
            if hasattr(engine, 'map_manager') and hasattr(engine.map_manager, 'current_map'):
                current_map = engine.map_manager.current_map
                if hasattr(current_map, 'blocks') and len(current_map.blocks) > 0:
                    if hasattr(current_map.blocks[-1], 'map_data'):
                        map_data = current_map.blocks[-1].map_data
            
            if map_data is None and hasattr(engine, 'current_map') and hasattr(engine.current_map, 'map_data'):
                map_data = engine.current_map.map_data
            
            if map_data is None:
                # If map_data is missing, fall back to road_network
                if hasattr(road_network, 'graph'):
                    map_data = road_network.graph
            
            if map_data:
                for lane_id, lane_info in map_data.items():
                    # Check that this entry is a lane
                    lane_type = lane_info.get("type", "")
                    if not MetaDriveType.is_lane(lane_type):
                        continue
                    
                    # Get the polyline
                    polyline = None
                    if "polyline" in lane_info:
                        polyline = np.array(lane_info["polyline"])
                    elif "points" in lane_info:
                        polyline = np.array(lane_info["points"])
                    elif "centerline" in lane_info:
                        polyline = np.array(lane_info["centerline"])
                    
                    if polyline is None or len(polyline) < 2:
                        continue
                    
                    # Make sure the polyline is 2D
                    if polyline.shape[1] > 2:
                        polyline = polyline[:, :2]
                    
                    # Get the lane width
                    width = lane_info.get("width", 3.5)
                    if isinstance(width, (list, tuple)):
                        width = width[0]  # Use the first width
                    
                    # Iterate over polyline points
                    for i in range(len(polyline) - 1):
                        pt1 = polyline[i]
                        pt2 = polyline[i + 1]
                        
                        # Convert points to ego frame
                        ex1, ey1 = world_to_ego_xy(pt1[0], pt1[1])
                        ex2, ey2 = world_to_ego_xy(pt2[0], pt2[1])
                        
                        # Check that points are within the visible range
                        if abs(ex1) > size_meters / 2 and abs(ex2) > size_meters / 2:
                            continue
                        
                        # Draw the lane (filled)
                        # Interpolate between points
                        num_steps = max(2, int(np.hypot(ex2 - ex1, ey2 - ey1) * scale))
                        for step in range(num_steps + 1):
                            t = step / num_steps
                            ex = ex1 + t * (ex2 - ex1)
                            ey = ey1 + t * (ey2 - ey1)
                            
                            if abs(ex) > size_meters / 2 or abs(ey) > size_meters / 2:
                                continue
                            
                            px, py = ego_to_pix(ex, ey)
                            w = max(1, int(width / 2 * scale))
                            for dw in range(-w, w + 1):
                                for dh in range(-w, w + 1):
                                    nx, ny = px + dw, py + dh
                                    if 0 <= nx < resolution and 0 <= ny < resolution:
                                        sem_map[ny, nx] = BEV_IDX_STREET
                        
                        # Draw lines (borders)
                        # Left and right borders
                        direction = np.array([pt2[0] - pt1[0], pt2[1] - pt1[1]])
                        direction_norm = np.linalg.norm(direction)
                        if direction_norm > 1e-6:
                            direction = direction / direction_norm
                            perpendicular = np.array([-direction[1], direction[0]])
                            
                            left_offset = perpendicular * (width / 2)
                            right_offset = -perpendicular * (width / 2)
                            
                            for offset, line_type in [(left_offset, 'left'), (right_offset, 'right')]:
                                # Use divider metadata when available.
                                # Fallback to solid lines for lane polygon borders.
                                idx = BEV_IDX_ALL_LINES
                                if line_type == 'left':
                                    left_type = lane_info.get("left_line_type")
                                    if left_type in ("broken", "dashed", "line_broken"):
                                        idx = BEV_IDX_BROKEN_LINES
                                else:
                                    right_type = lane_info.get("right_line_type")
                                    if right_type in ("broken", "dashed", "line_broken"):
                                        idx = BEV_IDX_BROKEN_LINES
                                
                                # Draw the line
                                for t in np.linspace(0, 1, num_steps + 1):
                                    pt = polyline[i] + t * (polyline[i + 1] - polyline[i])
                                    line_pt = pt + offset
                                    
                                    ex, ey = world_to_ego_xy(line_pt[0], line_pt[1])
                                    if abs(ex) > size_meters / 2 or abs(ey) > size_meters / 2:
                                        continue
                                    
                                    px, py = ego_to_pix(ex, ey)
                                    if 0 <= px < resolution and 0 <= py < resolution:
                                        sem_map[py, px] = idx

                # Draw explicit dividers from map_data with accurate broken/solid type.
                for feat_id, feat in map_data.items():
                    feat_type = feat.get("type", "")
                    if feat_type == MetaDriveType.LINE_BROKEN_SINGLE_WHITE:
                        idx = BEV_IDX_BROKEN_LINES
                    elif feat_type in (
                        MetaDriveType.LINE_SOLID_SINGLE_WHITE,
                        MetaDriveType.LINE_SOLID_DOUBLE_WHITE,
                        MetaDriveType.LINE_SOLID_SINGLE_YELLOW,
                        MetaDriveType.LINE_SOLID_DOUBLE_YELLOW,
                    ):
                        idx = BEV_IDX_ALL_LINES
                    else:
                        continue

                    polyline = feat.get("polyline")
                    if polyline is None:
                        continue
                    polyline = np.asarray(polyline, dtype=np.float64)
                    if polyline.ndim != 2 or len(polyline) < 2:
                        continue
                    if polyline.shape[1] > 2:
                        polyline = polyline[:, :2]

                    for i in range(len(polyline) - 1):
                        pt1 = polyline[i]
                        pt2 = polyline[i + 1]
                        ex1, ey1 = world_to_ego_xy(pt1[0], pt1[1])
                        ex2, ey2 = world_to_ego_xy(pt2[0], pt2[1])
                        num_steps = max(2, int(np.hypot(ex2 - ex1, ey2 - ey1) * scale))
                        for step in range(num_steps + 1):
                            t = step / num_steps
                            ex = ex1 + t * (ex2 - ex1)
                            ey = ey1 + t * (ey2 - ey1)
                            if abs(ex) > size_meters / 2 or abs(ey) > size_meters / 2:
                                continue
                            px, py = ego_to_pix(ex, ey)
                            if 0 <= px < resolution and 0 <= py < resolution:
                                sem_map[py, px] = idx
                                        
        except Exception as e:
            print(f"Warning: Error rendering BEV for EdgeRoadNetwork: {e}")
            pass

    if return_semantic_map:
        return sem_map

    # Convert to RGB
    rgb = BEV_COLORS[sem_map]
    bev_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return bev_t


def metadrive_obs_to_plant2_batch(
    engine,
    ego_vehicle,
    route_ego_20x2=None,
    speed_limit_kmh=None,
    max_objects=30,
    max_distance=50.0,
    range_factor_front=2.0,
    input_bev=False,
    input_ego_speed=False,
    bev_resolution=128,
    bev_size_meters=64.0,
    device="cpu",
    include_stop_signs=True,
):
    """
    Convert MetaDrive state into a batch for PlanT/HFLM.
    Mirrors PlanT generate_batch: idxs (B, maxseq) with zeros as padding, x_objs is a pool.
    """
    import torch
    from metadrive.policy.plant_policy import get_route_points_ego_frame

    num_route_points = 20

    if route_ego_20x2 is None:
        route_ego, _ = get_route_points_ego_frame(ego_vehicle, num_route_points)
        route_ego_20x2 = route_ego

    route_ego_20x2 = np.asarray(route_ego_20x2, dtype=np.float32)
    if route_ego_20x2.shape[0] < num_route_points:
        pad = np.tile(route_ego_20x2[-1], (num_route_points - route_ego_20x2.shape[0], 1))
        route_ego_20x2 = np.vstack([route_ego_20x2, pad])
    route_ego_20x2 = route_ego_20x2[:num_route_points]

    objects = collect_objects_ego_frame(
        engine, ego_vehicle,
        max_objects=max_objects,
        max_distance=max_distance,
        range_factor_front=range_factor_front,
        include_stop_signs=include_stop_signs,
    )
    x_list, num_objs = objects_to_x_batch(objects, max_objects)

    # IMPORTANT for batching: pad/truncate pool to fixed (max_objects+1, 7)
    # Index 0 is padding token; valid object indices are 1..max_objects.
    pool_size = max_objects + 1
    if len(x_list) > pool_size:
        x_list = x_list[:pool_size]
        num_objs = min(num_objs, max_objects)
    elif len(x_list) < pool_size:
        x_list = x_list + [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * (pool_size - len(x_list))

    x_batch_objs = torch.tensor(x_list, dtype=torch.float32, device=device)
    maxseq = max_objects
    batch_idxs = torch.zeros((1, maxseq), dtype=torch.int32, device=device)
    if num_objs > 0:
        batch_idxs[0, :num_objs] = torch.arange(1, 1 + num_objs, dtype=torch.int32, device=device)

    route = torch.tensor(route_ego_20x2, dtype=torch.float32, device=device).unsqueeze(0)
    speed_limit_idx = get_speed_limit_idx(speed_limit_kmh)
    speed_limit = torch.tensor([min(3, max(0, int(speed_limit_idx)))], dtype=torch.long, device=device)

    batch = {
        "idxs": batch_idxs,
        "x_objs": x_batch_objs,
        "route_original": route,
        "speed_limit": speed_limit,
        "y_objs": None,
    }

    if input_bev:
        bev_t = render_bev_plant2(
            engine, ego_vehicle,
            resolution=bev_resolution,
            size_meters=bev_size_meters,
            device=device,
        )
        if bev_t is not None:
            # BEV from render_bev_plant2 already has forward=top,
            # matching CARLA convention after rot90 CCW — no extra rotation needed.
            batch["BEV"] = torch.rot90(bev_t, k=-1, dims=[2, 3])
        else:
            batch["BEV"] = torch.zeros(1, 3, bev_resolution, bev_resolution, dtype=torch.float32, device=device)


    if input_ego_speed:
        ego_speed = float(getattr(ego_vehicle, "speed", 0.0))
        batch["input_ego_speed"] = torch.tensor([[ego_speed]], dtype=torch.float32, device=device)


    return batch
