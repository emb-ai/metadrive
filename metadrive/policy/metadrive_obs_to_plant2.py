"""
Convert MetaDrive obs/state into PlanT 2.0 input format (HFLM-compatible).

Mirrors PlanT/dataset.py::generate_batch and PlanTVariables.

Spatial PDD signs in ``x_objs`` come from the same ``plant2_frames.collect_boxes``
used by the train dump (class = PDD code, position, affects_ego=True).
"""
import os
import sys
from pathlib import Path
import numpy as np

# PlanT object types (PlanTVariables.class_nums): 0=padding, 1=vehicle, 2=pedestrian, 3=static,
# 4=stop_sign, 5=traffic_light, 6=emergency, 7..=PDD codes in SIGN_CODES
OBJ_TYPE_PADDING = 0
OBJ_TYPE_VEHICLE = 1
OBJ_TYPE_PEDESTRIAN = 2
OBJ_TYPE_STATIC = 3
OBJ_TYPE_STOP_SIGN = 4
OBJ_TYPE_TRAFFIC_LIGHT = 5
OBJ_TYPE_EMERGENCY = 6

# The dump collector and PlanT sit at different depths in the two checkouts
# this file is vendored into, so walk up for a marker instead of counting
# parents: `finetune/plant2_frames.py` in this repository,
# `pdd-bench/scripts/per_sign_bench/bench/plant2_frames.py` in the legacy one.
_NEW_COLLECTOR = Path("finetune") / "plant2_frames.py"
_LEGACY_COLLECTOR = (Path("pdd-bench") / "scripts" / "per_sign_bench"
                     / "bench" / "plant2_frames.py")


def _resolve_layout():
    """Return (repo_root, sys.path entries) for whichever checkout this is."""
    for root in Path(__file__).resolve().parents:
        if (root / _NEW_COLLECTOR).is_file():
            return root, [root, root / "third_party" / "plant2" / "PlanT"]
        if (root / _LEGACY_COLLECTOR).is_file():
            return root, [
                root / "pdd-bench" / "scripts" / "per_sign_bench",
                root / "pdd-bench",
                root / "plant2" / "PlanT",
            ]
    return None, []


_TRB_ROOT, _IMPORT_PATHS = _resolve_layout()

# Explicit PDD sign token ids (mirrors PlanT/util/sign_id.py SIGN_CODES order).
_SIGN_FALLBACK = {
    "2.1": 1, "2.3.1": 2, "2.3.2": 3, "2.3.3": 4, "2.4": 5, "2.5": 6,
    "3.1": 7, "3.2": 8, "3.24": 9,
    "4.2.1": 10, "4.2.2": 11, "4.2.3": 12, "4.3": 13, "4.6": 14,
    "5.7.1": 15, "5.7.2": 16, "5.15.1": 17, "5.15.2": 18, "5.19": 19,
    "5.21": 20, "5.31": 21,
}


def _ensure_collect_boxes_import_paths() -> None:
    """Make the dump collector, ``plant_variables`` and ``util.sign_id`` importable."""
    for p in _IMPORT_PATHS:
        ps = str(p)
        if p.is_dir() and ps not in sys.path:
            sys.path.insert(0, ps)


def _from_collector(name):
    """Pull one name off the train-dump collector, whichever tree ships it.

    Eval must read its objects through the very module that wrote the training
    boxes -- a second implementation is how the sign silently stops matching
    between train and eval.
    """
    _ensure_collect_boxes_import_paths()
    errors = []
    for mod in ("finetune.plant2_frames", "bench.plant2_frames", "plant2_frames"):
        try:
            module = __import__(mod, fromlist=[name])
        except ImportError as exc:
            errors.append(f"{mod}: {exc}")
            continue
        return getattr(module, name)
    raise ImportError(
        f"cannot import {name} from the plant2 frame collector; tried "
        + "; ".join(errors)
    )


def _import_collect_boxes():
    """Literal train-dump collector: ``plant2_frames.collect_boxes``."""
    return _from_collector("collect_boxes")


def _import_resolve_pdd_code_from_sign():
    return _from_collector("resolve_pdd_code_from_sign")


def _yaw_rad_to_deg(yaw_rad: float) -> float:
    """Match PlanTDataset.rad2deg / normalize_angle_degree."""
    x = float(np.rad2deg(yaw_rad)) % 360.0
    if x > 180.0:
        x -= 360.0
    return x


def _get_type_nums_and_sign_like():
    """PlanTVariables.class_nums + PDD sign class set (train mapping)."""
    _ensure_collect_boxes_import_paths()
    try:
        from plant_variables import PlanTVariables
        type_nums = dict(PlanTVariables.class_nums)
        sign_like = {"stop_sign"} | set(PlanTVariables.pdd_object_classes)
        car_types = set(PlanTVariables.car_types)
        return type_nums, sign_like, car_types
    except Exception:
        # Minimal fallback if PlanT is not on path yet.
        type_nums = {
            "car": 1.0, "walker": 2.0, "static": 3.0, "stop_sign": 4.0,
            "traffic_light": 5.0, "emergency": 6.0,
        }
        for i, code in enumerate(_SIGN_FALLBACK):
            type_nums[code] = float(7 + i)
        sign_like = {"stop_sign"} | set(_SIGN_FALLBACK)
        return type_nums, sign_like, {"car", "walker", "emergency"}


def _maybe_move_yield_sign_onto_npc(engine, ego_vehicle) -> None:
    """Diagnostic: teleport PDD sign mesh onto the nearest non-ego vehicle.

    Enabled by ``PLANT2_REPLACE_NPC_WITH_MOVING_SIGN``:
      ``1`` / ``true`` — prefer YieldSign, else StopSign, else first sign
      ``2.4`` / ``2.5`` — prefer that PDD code / class name
    Combined with skipping car tokens in ``boxes_to_objects_list``, PlanT sees a
    *moving* sign-class token at the NPC pose (dump-like).
    """
    flag = (os.environ.get("PLANT2_REPLACE_NPC_WITH_MOVING_SIGN") or "").strip()
    if not flag:
        return
    sign_mgr = getattr(engine, "traffic_sign_manager", None)
    if sign_mgr is None:
        return
    signs = [s for s in (getattr(sign_mgr, "signs", None) or []) if s is not None]
    if not signs:
        return

    want = flag if flag not in ("1", "true", "True") else None

    def _rank(s) -> int:
        name = type(s).__name__
        pdd = str(getattr(s, "pdd_code", None) or "")
        if want == "2.5":
            if "Stop" in name or pdd == "2.5":
                return 0
            return 2
        if want == "2.4":
            if "Yield" in name or pdd == "2.4":
                return 0
            return 2
        # generic: Yield first, then Stop, then anything
        if "Yield" in name or pdd == "2.4":
            return 0
        if "Stop" in name or pdd == "2.5":
            return 1
        return 2

    signs_sorted = sorted(signs, key=_rank)
    sign = signs_sorted[0]

    ego_id = getattr(ego_vehicle, "id", None)
    candidates = []
    traffic = getattr(engine, "traffic_manager", None)
    vehicles = list(getattr(traffic, "vehicles", None) or [])
    if not vehicles and hasattr(engine, "get_objects"):
        try:
            vehicles = [
                o for o in engine.get_objects().values()
                if o is not None and type(o).__name__.endswith("Vehicle")
            ]
        except Exception:
            vehicles = []
    for v in vehicles:
        if v is None or v is ego_vehicle:
            continue
        if ego_id is not None and getattr(v, "id", None) == ego_id:
            continue
        try:
            pos = np.asarray(v.position[:2], dtype=np.float64)
            ego_pos = np.asarray(ego_vehicle.position[:2], dtype=np.float64)
            dist = float(np.linalg.norm(pos - ego_pos))
        except Exception:
            continue
        candidates.append((dist, v))
    if not candidates:
        return
    candidates.sort(key=lambda t: t[0])
    npc = candidates[0][1]
    try:
        pos = npc.position
        heading = float(getattr(npc, "heading_theta", 0.0))
        if hasattr(sign, "set_position"):
            z = float(pos[2]) if len(pos) > 2 else 0.5
            sign.set_position([float(pos[0]), float(pos[1]), z])
        if hasattr(sign, "set_heading_theta"):
            sign.set_heading_theta(heading)
        elif hasattr(sign, "_heading_theta"):
            sign._heading_theta = heading
        if hasattr(sign, "_position"):
            sign._position = np.array([float(pos[0]), float(pos[1])], dtype=np.float64)
    except Exception as exc:
        if os.environ.get("PLANT2_DEBUG_BOXES"):
            print(f"[PLANT2_REPLACE_NPC_WITH_MOVING_SIGN] failed: {exc}", flush=True)


def boxes_to_objects_list(boxes, max_objects=30, include_stop_signs=True):
    """Convert ``collect_boxes`` dicts → object tuples for ``objects_to_x_batch``.

    Mirrors PlanTDataset input construction (skip ego, cars + staticish/PDD).
    Tuple: (type, x, y, yaw_deg, speed_kmh, ext_x, ext_y) with half-extents
    (``objects_to_x_batch`` multiplies extents by 2, same as train full size).

    Diagnostics (env):
      PLANT2_REMAP_NPC_TO_SIGN=2.4|2.5
          rewrite non-ego car tokens as that PDD class (pose kept).
      PLANT2_REMAP_KEEP_SPEED=1
          keep car speed_kmh after remap (default: force 0 like static signs).
      PLANT2_REPLACE_NPC_WITH_MOVING_SIGN=1|2.4|2.5
          skip car tokens entirely (sign mesh was teleported onto NPC).
      PLANT2_DROP_STATIC_SIGN_WHEN_REMAP=1
          drop boxes whose class equals the remap code (avoid double tokens).
      PLANT2_OBJ_ORDER=dump|dist
          token order. `dist` (default) sorts by distance to ego; `dump`
          reproduces training order — cars in box order, then static-ish and
          signs — because the trunk is a BERT and its position embeddings make
          the order part of the input. Under `dump` the cap still keeps the
          nearest `max_objects`, it just emits them in training order.
    """
    type_nums, sign_like, car_types = _get_type_nums_and_sign_like()
    # PLANT2_SIGN_OBJS=0 demotes PDD signs back to the generic `static` class —
    # the pre-fix behaviour, so an A/B can isolate what the sign class buys.
    sign_objs = (os.environ.get("PLANT2_SIGN_OBJS", "1") or "1").strip() not in ("0", "false", "False")
    remap_sign = (os.environ.get("PLANT2_REMAP_NPC_TO_SIGN") or "").strip()
    remap_type = float(type_nums[remap_sign]) if remap_sign in type_nums else None
    keep_speed = (os.environ.get("PLANT2_REMAP_KEEP_SPEED") or "").strip() in ("1", "true", "True")
    replace_flag = (os.environ.get("PLANT2_REPLACE_NPC_WITH_MOVING_SIGN") or "").strip()
    replace_mode = bool(replace_flag)
    drop_static = (os.environ.get("PLANT2_DROP_STATIC_SIGN_WHEN_REMAP") or "").strip() in (
        "1", "true", "True",
    )
    # Ego is always first in collect_boxes.
    labels = boxes[1:] if boxes and boxes[0].get("class") == "car" and boxes[0].get("id") == 0 else boxes

    # Kept apart so `dump` order can put every car before the static-ish tokens,
    # the way PlanTDataset builds input_objects (cars list, then the rest).
    cars: list = []
    staticish: list = []
    for x in labels:
        cls_raw = x.get("class")
        if not isinstance(cls_raw, str):
            continue
        cls_key = cls_raw if cls_raw in type_nums else cls_raw.lower()
        if cls_key not in type_nums:
            continue
        if not include_stop_signs and cls_key in ("stop_sign", "2.5"):
            continue
        if drop_static and remap_sign and cls_key == remap_sign:
            continue

        pos = x.get("position") or [0.0, 0.0, 0.0]
        extent = x.get("extent") or [1.0, 1.0, 0.75]
        yaw_deg = _yaw_rad_to_deg(float(x.get("yaw", 0.0)))
        # Half-extents as stored by collect_boxes; objects_to_x_batch does *2.
        ext_x = float(extent[0])
        ext_y = float(extent[1])

        if cls_key in car_types:
            if replace_mode and cls_key == "car":
                # Sign mesh carries the moving PDD token; drop the car token.
                continue
            speed_kmh = float(x.get("speed", 0.0)) * 3.6
            if remap_type is not None and cls_key == "car":
                cars.append((
                    remap_type,
                    float(pos[0]), float(pos[1]), yaw_deg,
                    speed_kmh if keep_speed else 0.0,
                    ext_x, ext_y,
                ))
            else:
                cars.append((
                    float(type_nums[cls_key]) if (sign_objs or cls_key not in sign_like)
                    else float(type_nums["static"]),
                    float(pos[0]), float(pos[1]), yaw_deg,
                    speed_kmh,
                    ext_x, ext_y,
                ))
            continue

        if cls_key == "traffic_light":
            if x.get("state") not in ("Red", "Yellow") or not x.get("affects_ego"):
                continue
        elif cls_key in sign_like:
            if not x.get("affects_ego", True):
                continue
            # Counterfactual on the OBJECT channel. The sign reaches the model
            # twice: as the global sign_id token and as an object whose class
            # already separates the codes (4.2.1/4.2.2/4.2.3 are types 16/17/18).
            # Forcing only the global token therefore proves nothing about the
            # side the model chooses. This rewrites the object's class, leaving
            # its pose untouched, so both channels can be swapped together.
            forced_obj = (os.environ.get("PLANT2_FORCE_SIGN_OBJ_CODE") or "").strip()
            if forced_obj and forced_obj in type_nums:
                cls_key = forced_obj

        staticish.append((
            float(type_nums[cls_key]) if (sign_objs or cls_key not in sign_like)
                    else float(type_nums["static"]),
            float(pos[0]), float(pos[1]), yaw_deg,
            # Mirrors PlanTDataset: the slot is 0 for anything static, except a
            # speed-limit plate, whose own number rides here in km/h. Zeroing
            # it at eval while training carried the value would hide the only
            # channel that distinguishes 20 from 60.
            float(x.get("speed", 0.0)) * 3.6,
            ext_x, ext_y,
        ))

    objects = cars + staticish
    dist2 = lambda o: o[1] ** 2 + o[2] ** 2  # noqa: E731
    order = (os.environ.get("PLANT2_OBJ_ORDER") or "dist").strip().lower()
    if len(objects) > max_objects:
        # Cap by proximity in both modes; `dump` only differs in the order the
        # survivors are emitted, so truncation never silently drops the sign.
        nearest = sorted(range(len(objects)), key=lambda i: dist2(objects[i]))[:max_objects]
        objects = [objects[i] for i in sorted(nearest)]
    if order != "dump":
        objects.sort(key=dist2)
    return objects


def collect_objects_ego_frame_from_plant2_boxes(
    engine,
    ego_vehicle,
    max_objects=30,
    max_distance=50.0,
    range_factor_front=2.0,
    include_stop_signs=True,
):
    """Train-dump path: ``collect_boxes`` → ``boxes_to_objects_list`` (ego frame)."""
    collect_boxes = _import_collect_boxes()
    boxes = collect_boxes(
        engine,
        ego_vehicle,
        max_distance=max_distance,
        range_factor_front=range_factor_front,
    )
    return boxes_to_objects_list(
        boxes,
        max_objects=max_objects,
        include_stop_signs=include_stop_signs,
    )


def collect_objects_ego_frame(
    engine,
    ego_vehicle,
    max_objects=30,
    max_distance=50.0,
    range_factor_front=2.0,
    include_stop_signs=True,
):
    """Legacy alias — same as the plant2_boxes collector."""
    return collect_objects_ego_frame_from_plant2_boxes(
        engine,
        ego_vehicle,
        max_objects=max_objects,
        max_distance=max_distance,
        range_factor_front=range_factor_front,
        include_stop_signs=include_stop_signs,
    )


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
    """PlanTVariables.speed_cats. Defaults to 1 (80 km/h) when no limit is set.

    `SPEED_CATS` was never defined in this module, so any caller passing a real
    limit hit a NameError instead of getting a token. Read the table from
    PlanTVariables and, like PlanTDataset, fall back to the nearest known limit
    rather than to a fixed bin -- posted PDD limits (20/40/60) are not in the
    table and training maps them to 50, not to 80.
    """
    if speed_limit_kmh is None:
        return 1
    _ensure_collect_boxes_import_paths()
    try:
        from plant_variables import PlanTVariables
        cats = dict(PlanTVariables.speed_cats)
    except Exception:
        cats = {50: 0, 80: 1, 100: 2, 120: 3}
    key = int(round(float(speed_limit_kmh)))
    if key not in cats:
        key = min(cats, key=lambda k: abs(k - key))
    return cats[key]


def _sign_code_to_id(code, value_kmh=None):
    try:
        _ensure_collect_boxes_import_paths()
        from util.sign_id import sign_code_to_id as _fn
        return _fn(code, value_kmh)
    except Exception:
        if not code:
            return 0
        return _SIGN_FALLBACK.get(str(code).strip(), 0)


_SIGN_DEBUG_SEEN = set()


# Which attribute of a sign object carries the number on its plate. Mirrors the
# dump's _SIGN_VALUE_ATTR: reading whichever speed attribute happens to exist
# would put a road speed on plates that prescribe nothing.
_SIGN_VALUE_ATTR = {
    "3.24": "speed_limit",
    "5.31": "speed_limit",
    "4.6": "min_speed",
}


def resolve_sign_value_from_engine(engine, code):
    """The number on the placed plate in km/h, or None if it carries none.

    Training tokenises a speed sign together with its number -- "3.24@40" is a
    different token from "3.24@20", because the two demand opposite speeds. An
    eval that passes the bare code hands the model a token it never saw trained
    against any target, and the speed head answers at chance while every log
    still shows the sign as present.
    """
    attr = _SIGN_VALUE_ATTR.get(str(code or "").strip())
    if attr is None:
        return None
    try:
        _value_kmh = _from_collector("_sign_value_kmh")
    except Exception:
        _value_kmh = None
    mgr = getattr(engine, "traffic_sign_manager", None)
    if mgr is None:
        return None
    for sign in getattr(mgr, "signs", []) or []:
        if sign is None:
            continue
        if _value_kmh is not None:
            val = _value_kmh(sign, code)
            if val is not None:
                return float(val)
        raw = getattr(sign, attr, None)
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass
    return None


def resolve_sign_code_from_engine(engine, explicit_code=None):
    """Best-effort PDD code from env config or traffic_sign_manager.

    Uses the same ``resolve_pdd_code_from_sign`` as the train dump when possible.
    """
    if explicit_code:
        return str(explicit_code)
    cfg = getattr(engine, "global_config", None) or {}
    if hasattr(cfg, "get"):
        for key in ("sign_type", "sign_code", "pdd_code"):
            val = cfg.get(key)
            if val:
                return str(val)
    mgr = getattr(engine, "traffic_sign_manager", None)
    if mgr is None:
        return None
    try:
        resolve_pdd = _import_resolve_pdd_code_from_sign()
    except Exception:
        resolve_pdd = None
    for sign in getattr(mgr, "signs", []) or []:
        if sign is None:
            continue
        if resolve_pdd is not None:
            code = resolve_pdd(sign)
            if code:
                return str(code)
        for attr in ("pdd_code", "sign_code", "sign_type", "code"):
            val = getattr(sign, attr, None)
            if val:
                return str(val)
    return None


# Semantic class indices for the BEV map. The EdgeRoadNetwork branch references
# these by name; without them it raised NameError on every frame, the warning
# swallowed it, and SUMO scenes fed the model a BEV with no street layer.
BEV_IDX_BACKGROUND = 0
BEV_IDX_STREET = 1
BEV_IDX_SIDEWALK = 2
BEV_IDX_ALL_LINES = 3
BEV_IDX_BROKEN_LINES = 4

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
    sign_code=None,
    include_sign_id=False,
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

    _maybe_move_yield_sign_onto_npc(engine, ego_vehicle)

    # Prefer train-dump collector (includes spatial PDD signs in x_objs).
    try:
        objects = collect_objects_ego_frame_from_plant2_boxes(
            engine, ego_vehicle,
            max_objects=max_objects,
            max_distance=max_distance,
            range_factor_front=range_factor_front,
            include_stop_signs=include_stop_signs,
        )
    except Exception as exc:
        if os.environ.get("PLANT2_DEBUG_BOXES"):
            print(f"[metadrive_obs_to_plant2] collect_boxes failed ({exc}); "
                  f"falling back to legacy collect_objects_ego_frame", flush=True)
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
    # generate_batch sets maxseq to the batch's own object count, so training
    # sequences are as long as the scene needs. A fixed 30 here pads the tail
    # and pushes speed_token — the token the ego-speed head reads — to an
    # absolute position the model may never have seen (BERT position embeddings).
    maxseq = max_objects
    if (os.environ.get("PLANT2_SEQ_FIT") or "").strip() in ("1", "true", "True"):
        maxseq = max(1, num_objs)
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

    if include_sign_id:
        code = resolve_sign_code_from_engine(engine, explicit_code=sign_code)
        value_kmh = resolve_sign_value_from_engine(engine, code)
        _sid = _sign_code_to_id(code, value_kmh)
        if os.environ.get("PLANT2_DEBUG_SIGN", "") in ("1", "true", "True"):
            # The token the model actually receives. A sign that is placed,
            # visible and correctly scored can still reach the model as id 0 or
            # as the number-less variant of its code, and nothing downstream
            # says so -- the run just behaves as if the plate were blank.
            global _SIGN_DEBUG_SEEN
            key = (str(code), value_kmh, int(_sid))
            if key not in _SIGN_DEBUG_SEEN:
                _SIGN_DEBUG_SEEN.add(key)
                print(f"[PlanT2Sign] code={code!r} value_kmh={value_kmh!r} sign_id={_sid}",
                      flush=True)
        batch["sign_id"] = torch.tensor([_sid], dtype=torch.long, device=device)

    if input_bev:
        # Reproduce the training pipeline exactly. The dumper renders the same
        # semantic map at 256 px over 64 m (plant2_frames: _BEV_RESOLUTION /
        # _BEV_SIZE_METERS) and PlanTDataset then rotates it by +1 and keeps the
        # central 128 px -- 32 m at 0.25 m/px. Evaluation used to rotate by -1
        # and skip the crop, so the model was shown the scene turned 180 degrees
        # at half the resolution over twice the area. PLANT2_BEV_LEGACY=1 brings
        # the old behaviour back for an A/B.
        legacy = os.environ.get("PLANT2_BEV_LEGACY", "").strip() in ("1", "true", "True")
        render_res = bev_resolution if legacy else bev_resolution * 2
        bev_t = render_bev_plant2(
            engine, ego_vehicle,
            resolution=render_res,
            size_meters=bev_size_meters,
            device=device,
        )
        if bev_t is None:
            batch["BEV"] = torch.zeros(1, 3, bev_resolution, bev_resolution,
                                       dtype=torch.float32, device=device)
        elif legacy:
            batch["BEV"] = torch.rot90(bev_t, k=-1, dims=[2, 3])
        else:
            rotated = torch.rot90(bev_t, k=1, dims=[2, 3])
            pad = bev_resolution // 2
            batch["BEV"] = rotated[:, :, pad:-pad, pad:-pad]


    if input_ego_speed:
        ego_speed = float(getattr(ego_vehicle, "speed", 0.0))
        batch["input_ego_speed"] = torch.tensor([[ego_speed]], dtype=torch.float32, device=device)


    return batch
