import copy
from metadrive.engine.logger import get_logger

from metadrive.utils.doc_utils import generate_gif
import math
from collections import deque
from typing import Optional, Union, Iterable

import numpy as np

from metadrive.component.map.scenario_map import ScenarioMap
from metadrive.constants import Decoration, TARGET_VEHICLES
from metadrive.constants import TopDownSemanticColor, MetaDriveType, PGDrivableAreaProperty
from metadrive.obs.top_down_obs_impl import WorldSurface, ObjectGraphics, LaneGraphics, history_object
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.utils.utils import import_pygame
from metadrive.utils.utils import is_map_related_instance
from metadrive.component.navigation_module.edge_network_navigation import EdgeNetworkNavigation

pygame = import_pygame()

color_white = (255, 255, 255)
CROSSWALK_BG_COLOR = (150, 150, 150)
CROSSWALK_STRIPE_WHITE = (248, 248, 248)
CROSSWALK_STRIPE_YELLOW = (246, 214, 73)
CROSSWALK_BORDER_COLOR = (70, 70, 70)
CROSSWALK_STRIPE_PITCH_PX = 6.0
CROSSWALK_STRIPE_GAP_RATIO = 0.18
CROSSWALK_MIN_STRIPE_COUNT = 4
CROSSWALK_STOP_LINE_COLOR = (0, 0, 0)
CROSSWALK_STOP_LINE_WIDTH_PX = 3


def _uturn_side_from_dest(src_x, src_y, heading, dest_polyline):
    """Determine which side (left/right) the U-turn destination lane is on.

    Uses the cross product of the source lane's forward direction with
    the vector from source anchor to the destination lane's start point.

    Returns +1 for left, -1 for right (in lane-local coordinates).
    """
    if dest_polyline is None or len(dest_polyline) < 1:
        return +1  # default: left (standard for right-hand traffic)
    dest_x = float(dest_polyline[0][0])
    dest_y = float(dest_polyline[0][1])
    # Forward direction of source lane (world coords)
    fwd_x = math.cos(heading)
    fwd_y = math.sin(heading)
    # Vector from source to destination
    to_x = dest_x - src_x
    to_y = dest_y - src_y
    # 2D cross product: positive means destination is to the LEFT
    cross = fwd_x * to_y - fwd_y * to_x
    return +1 if cross >= 0 else -1


def _draw_lane_thin_arrows(surface, map_data):
    """Draw  vector arrows on each lane, indicating allowed
    directions.Internal/junction lanes (SUMO IDs with ``:``) are skipped.
    """
    for feat_id, data in map_data.items():
        feat_type = data.get("type")
        if feat_type is None or not MetaDriveType.is_lane(feat_type):
            continue
        raw_lane_id = feat_id[5:] if feat_id.startswith("lane_") else feat_id
        if ":" in raw_lane_id:
            continue
        turns = data.get("turns")
        if not turns:
            continue
        polyline = data.get("polyline")
        if polyline is None or len(polyline) < 2:
            continue

        dirs = [t["direction"] for t in turns]

        end_x, end_y = float(polyline[-1][0]), float(polyline[-1][1])
        prev_x, prev_y = float(polyline[-2][0]), float(polyline[-2][1])
        lane_heading = math.atan2(end_y - prev_y, end_x - prev_x)
        lane_width = data.get("width", 3.5)

        pullback = 6.0
        dx, dy = end_x - prev_x, end_y - prev_y
        seg_len = math.hypot(dx, dy)
        if seg_len > 1e-6:
            frac = min(pullback / seg_len, 0.9)
            ax = end_x - dx * frac
            ay = end_y - dy * frac
        else:
            ax, ay = end_x, end_y

        # u-turn side
        uturn_side = +1
        for t in turns:
            if t["direction"] == 't':
                dest_id = t.get("to_lane")
                if dest_id and dest_id in map_data:
                    dest_poly = map_data[dest_id].get("polyline")
                    uturn_side = _uturn_side_from_dest(
                        ax, ay, lane_heading, dest_poly
                    )
                break

        _draw_thin_arrows_at(surface, ax, ay, lane_heading, dirs, lane_width,
                             uturn_side=uturn_side)


def _draw_thin_arrows_at(surface, world_x, world_y, heading, dirs, lane_width,
                         color=(255, 255, 255), shaft_m=5.0, line_w=2,
                         uturn_side=1):
    """Draw a complex  arrow with shared stem and per-category branches """
    if not dirs:
        return

    #  count raw straights
    _CAT = {'l': 'left', 'L': 'left', 's': 'straight',
            'r': 'right', 'R': 'right', 't': 'uturn'}
    seen = set()
    cats = []
    straight_count = 0
    for d in dirs:
        c = _CAT.get(d)
        if c == 'straight':
            straight_count += 1
        if c and c not in seen:
            seen.add(c)
            cats.append(c)
    if not cats:
        return

    fwd_x, fwd_y = math.cos(heading), -math.sin(heading)
    rgt_x, rgt_y = math.sin(heading), math.cos(heading)

    px = shaft_m * surface.scaling
    lw = max(1, line_w)
    h_len = px * 0.10               # arrowhead length
    h_hw = max(1, px * 0.05)        # arrowhead half-width
    stem = px * 0.45                # shared stem length
    branch = px * 0.25              # branch length
    spread = branch * 0.2           # lateral offset for left/right
    branch_gap = px * 0.4           # backward shift of left/right junction from stem tip

    cx, cy = surface.pos2pix(world_x, world_y)
    # Shared stem
    bx = cx + stem * fwd_x
    by = cy + stem * fwd_y
    pygame.draw.line(surface, color, (cx, cy), (bx, by), lw)

    # Left/right branches start here (pulled back from stem tip)
    jx = bx - branch_gap * fwd_x
    jy = by - branch_gap * fwd_y

    def _head(tx, ty, dx, dy):
        """Filled triangle arrowhead at (tx,ty) pointing along (dx,dy)."""
        m = math.hypot(dx, dy)
        if m < 1e-6:
            return
        nx, ny = dx / m, dy / m
        px_, py_ = -ny, nx
        bx_ = tx - h_len * nx
        by_ = ty - h_len * ny
        pygame.draw.polygon(surface, color, [
            (tx, ty),
            (bx_ + h_hw * px_, by_ + h_hw * py_),
            (bx_ - h_hw * px_, by_ - h_hw * py_),
        ])

    for cat in cats:
        if cat == 'straight':
            tx, ty = bx + branch * fwd_x, by + branch * fwd_y
            pygame.draw.line(surface, color, (bx, by), (tx, ty), lw)
            _head(tx, ty, fwd_x, fwd_y)

            if straight_count >= 2:
                # Second straight
                lx, ly = -rgt_x, -rgt_y
                q = branch * 0.65
                p1x = bx + q * 0.3 * fwd_x
                p1y = by + q * 0.3 * fwd_y
                p2x = p1x + q * 1.2 * lx
                p2y = p1y + q * 1.2 * ly
                p3x = p2x + q * 1.4 * fwd_x
                p3y = p2y + q * 1.4 * fwd_y
                pygame.draw.lines(surface, color, False,
                                  [(bx, by), (p1x, p1y), (p2x, p2y), (p3x, p3y)], lw)
                _head(p3x, p3y, fwd_x, fwd_y)

        elif cat in ('left', 'right'):
            s = -1 if cat == 'left' else 1
            ox = jx + spread * s * rgt_x
            oy = jy + spread * s * rgt_y
            tx = ox + branch * s * rgt_x
            ty = oy + branch * s * rgt_y
            pygame.draw.line(surface, color, (jx, jy), (ox, oy), lw)
            pygame.draw.line(surface, color, (ox, oy), (tx, ty), lw)
            _head(tx, ty, s * rgt_x, s * rgt_y)

        elif cat == 'uturn':
            lx = -rgt_x * uturn_side
            ly = -rgt_y * uturn_side
            q = branch * 0.65
            p1x, p1y = bx + q * 0.3 * fwd_x, by + q * 0.3 * fwd_y
            p2x, p2y = p1x + q * 1.2 * lx, p1y + q * 1.2 * ly
            p3x, p3y = p2x - q * 1.4 * fwd_x, p2y - q * 1.4 * fwd_y
            pygame.draw.lines(surface, color, False,
                              [(bx, by), (p1x, p1y), (p2x, p2y), (p3x, p3y)], lw)
            _head(p3x, p3y, -fwd_x, -fwd_y)


def _crosswalk_reference_quad(polygon) -> Optional[np.ndarray]:
    """
    Build a stable 4-point quad for crosswalk rendering.
    SUMO crossing polygons may contain many points; in that case, estimate an oriented
    rectangle via PCA so zebra stripes are drawn over the full crossing area.
    """
    poly = np.asarray(polygon, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[1] < 2 or poly.shape[0] < 4:
        return None
    poly = poly[:, :2]
    if np.linalg.norm(poly[0] - poly[-1]) < 1e-6:
        poly = poly[:-1]
    if poly.shape[0] < 4:
        return None
    if poly.shape[0] == 4:
        return poly

    center = np.mean(poly, axis=0)
    centered = poly - center
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return poly[:4]

    axis_long = vh[0]
    axis_short = vh[1]
    proj_long = centered @ axis_long
    proj_short = centered @ axis_short
    l_min, l_max = float(np.min(proj_long)), float(np.max(proj_long))
    s_min, s_max = float(np.min(proj_short)), float(np.max(proj_short))

    # Order: a,b,c,d where b-a and c-d follow the "forward" axis used by stop-line logic.
    a = center + axis_long * l_min + axis_short * s_min
    b = center + axis_long * l_min + axis_short * s_max
    c = center + axis_long * l_max + axis_short * s_max
    d = center + axis_long * l_max + axis_short * s_min
    return np.asarray([a, b, c, d], dtype=np.float64)


def _draw_unified_drivable_area_boundary(surface, all_lanes) -> bool:
    lane_polygons = []
    for data in all_lanes.values():
        lane_type = data.get("type", None)
        polygon = data.get("polygon", None)
        if MetaDriveType.is_lane(lane_type) and polygon is not None:
            lane_polygons.append(polygon)
    if not lane_polygons:
        return False

    try:
        from shapely.geometry import Polygon, MultiPolygon
        from shapely.ops import unary_union
    except Exception:
        color = tuple(int(v) for v in TopDownSemanticColor.get_color(MetaDriveType.BOUNDARY_LINE))
        width = max(1, surface.pix(PGDrivableAreaProperty.LANE_LINE_WIDTH) * 2)
        for polygon in lane_polygons:
            pts = [surface.pos2pix(p[0], p[1]) for p in polygon]
            if len(pts) >= 2:
                pygame.draw.lines(surface, color, closed=True, points=pts, width=width)
        return True

    valid = []
    for coords in lane_polygons:
        arr = np.asarray(coords, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 3:
            continue
        arr = arr[:, :2]
        if np.linalg.norm(arr[0] - arr[-1]) < 1e-6 and arr.shape[0] >= 4:
            arr = arr[:-1]
        if arr.shape[0] < 3:
            continue
        poly = Polygon(arr)
        if poly.is_empty:
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        valid.append(poly)

    if not valid:
        return False

    merged = unary_union(valid)
    if isinstance(merged, Polygon):
        polygons = [merged]
    elif isinstance(merged, MultiPolygon):
        polygons = list(merged.geoms)
    else:
        polygons = [g for g in getattr(merged, "geoms", []) if isinstance(g, Polygon)]

    color = tuple(int(v) for v in TopDownSemanticColor.get_color(MetaDriveType.BOUNDARY_LINE))
    width = max(1, surface.pix(PGDrivableAreaProperty.LANE_LINE_WIDTH) * 2)
    for poly in polygons:
        ext = np.asarray(poly.exterior.coords, dtype=np.float64)
        if ext.shape[0] >= 2:
            pygame.draw.lines(
                surface, color, closed=True, points=[surface.pos2pix(p[0], p[1]) for p in ext], width=width
            )
        for interior in poly.interiors:
            hole = np.asarray(interior.coords, dtype=np.float64)
            if hole.shape[0] < 2:
                continue
            pygame.draw.lines(
                surface, color, closed=True, points=[surface.pos2pix(p[0], p[1]) for p in hole], width=width
            )
    return True


def draw_top_down_map_native(
    map,
    semantic_map=True,
    draw_center_line=False,
    return_surface=False,
    film_size=(2000, 2000),
    scaling=None,
    semantic_broken_line=True
) -> Optional[Union[np.ndarray, pygame.Surface]]:
    """
    Draw the top_down map on a pygame surface
    Args:
        map: MetaDrive.BaseMap instance
        semantic_map: return semantic map
        draw_center_line: Draw the center line of the lane
        return_surface: Return the pygame.Surface in fime_size instead of cv2.image
        film_size: The size of the film to draw the map
        scaling: the scaling factor, how many pixels per meter
        semantic_broken_line: Draw broken line on semantic map

    Returns: cv2.image or pygame.Surface

    """
    surface = WorldSurface(film_size, 0, pygame.Surface(film_size))
    if map is None:
        surface.move_display_window_to([0, 0])
        surface.fill([230, 230, 230])
        return surface if return_surface else WorldSurface.to_cv2_image(surface)

    b_box = map.road_network.get_bounding_box()
    x_len = b_box[1] - b_box[0]
    y_len = b_box[3] - b_box[2]
    max_len = max(x_len, y_len)
    # scaling and center can be easily found by bounding box
    if scaling is None:
        scaling = (film_size[1] / max_len - 0.1)
    else:
        scaling = min(scaling, (film_size[1] / max_len - 0.1))
    surface.scaling = scaling
    centering_pos = ((b_box[0] + b_box[1]) / 2, (b_box[2] + b_box[3]) / 2)
    surface.move_display_window_to(centering_pos)
    line_sample_interval = 2

    if semantic_map:
        all_lanes = map.get_map_features(line_sample_interval)

        # ScenarioMap (SUMO, Waymo, nuScenes): draw polygons from raw map_data — identical to ScenarioNet
        if isinstance(map, ScenarioMap) and map.blocks:
            block = map.blocks[-1]
            for feat_id, data in block.map_data.items():
                feat_type = data.get("type")
                if not feat_type:
                    continue
                polygon = data.get(ScenarioDescription.POLYGON)
                if polygon is None or len(polygon) < 3:
                    continue
                # Lanes, sidewalks, crosswalks — all with polygon
                if not (MetaDriveType.is_lane(feat_type) or MetaDriveType.is_sidewalk(feat_type)
                        or MetaDriveType.is_crosswalk(feat_type)):
                    continue
                polygon = np.asarray(polygon)[..., :2]
                pts = [surface.pos2pix(float(p[0]), float(p[1])) for p in polygon]
                color = tuple(int(c) for c in TopDownSemanticColor.get_color(feat_type))
                pygame.draw.polygon(surface, color, pts)
        else:
            for obj in all_lanes.values():
                if MetaDriveType.is_lane(obj["type"]) and not draw_center_line:
                    polygon = obj.get("polygon")
                    if polygon is not None and len(polygon) >= 3:
                        polygon = np.asarray(polygon)[..., :2]
                        pts = [surface.pos2pix(float(p[0]), float(p[1])) for p in polygon]
                        color = tuple(int(c) for c in TopDownSemanticColor.get_color(obj["type"]))
                        pygame.draw.polygon(surface, color, pts)

        for obj in all_lanes.values():
            if (MetaDriveType.is_road_line(obj["type"])
                  or (MetaDriveType.is_lane(obj["type"]) and draw_center_line)):
                polyline = obj.get("polyline")
                if polyline is None or len(polyline) < 2:
                    continue
                if semantic_broken_line and MetaDriveType.is_broken_line(obj["type"]):
                    points_to_skip = math.floor(PGDrivableAreaProperty.STRIPE_LENGTH * 2 / line_sample_interval) * 2
                else:
                    points_to_skip = 1
                for index in range(0, len(polyline) - 1, points_to_skip):
                    color = [255, 0, 0] if MetaDriveType.is_lane(obj["type"]) and index==0\
                        else TopDownSemanticColor.get_color(obj["type"])
                    if index + 1 < len(polyline):
                        s_p = polyline[index]
                        e_p = polyline[index + 1]
                        pygame.draw.line(
                            surface,
                            color,
                            surface.vec2pix([s_p[0], s_p[1]]),
                            surface.vec2pix([e_p[0], e_p[1]]),
                            # max(surface.pix(LaneGraphics.STRIPE_WIDTH),
                            surface.pix(PGDrivableAreaProperty.LANE_LINE_WIDTH) * 2
                        )
    else:
        if isinstance(map, ScenarioMap):
            line_sample_interval = 2
            all_lanes = map.get_map_features(line_sample_interval)
            # Draw boundary as contour of unified drivable area (same geometry as gray semantic surface).
            _draw_unified_drivable_area_boundary(surface, all_lanes)
            for id, data in all_lanes.items():
                if ScenarioDescription.POLYLINE not in data:
                    continue
                LaneGraphics.display_scenario_line(
                    data["polyline"], data["type"], surface, line_sample_interval=line_sample_interval
                )
        else:
            for _from in map.road_network.graph.keys():
                decoration = True if _from == Decoration.start else False
                for _to in map.road_network.graph[_from].keys():
                    for l in map.road_network.graph[_from][_to]:
                        two_side = True if l is map.road_network.graph[_from][_to][-1] or decoration else False
                        LaneGraphics.display(l, surface, two_side, use_line_color=True)

    # Draw thin direction arrows on each lane near its end
    if isinstance(map, ScenarioMap) and map.blocks:
        _draw_lane_thin_arrows(surface, map.blocks[-1].map_data)

    # Draw crosswalk polygons for ScenarioMap/PGMap if enabled in env config.
    show_crosswalk = True
    engine = getattr(map, "engine", None)
    no_stop_before_m = 5.0
    if engine is not None:
        show_crosswalk = bool(engine.global_config.get("show_crosswalk", True))
        ped_cfg = engine.global_config.get("pedestrian_manager", {})
        if hasattr(ped_cfg, "get_dict"):
            ped_cfg = ped_cfg.get_dict()
        if isinstance(ped_cfg, dict):
            no_stop_before_m = float(ped_cfg.get("no_stop_before_crosswalk_m", 5.0))

    if show_crosswalk:
        crosswalks = getattr(map, "crosswalks", {})
        for crosswalk in crosswalks.values():
            polygon = crosswalk.get("polygon", None)
            if polygon is None or len(polygon) < 3:
                continue
            quad_world = _crosswalk_reference_quad(polygon)
            pts = [surface.pos2pix(p[0], p[1]) for p in polygon]

            if semantic_map:
                # Semantic mode: keep class color.
                pygame.draw.polygon(surface, TopDownSemanticColor.get_color(MetaDriveType.CROSSWALK), pts)
            else:
                # Non-semantic mode: draw yellow/white zebra crossing.
                pygame.draw.polygon(surface, CROSSWALK_BG_COLOR, pts)
                if quad_world is not None:
                    quad_pts = [surface.pos2pix(p[0], p[1]) for p in quad_world]
                    a = np.asarray(quad_pts[0], dtype=np.float64)
                    b = np.asarray(quad_pts[1], dtype=np.float64)
                    c = np.asarray(quad_pts[2], dtype=np.float64)
                    d = np.asarray(quad_pts[3], dtype=np.float64)
                    sweep_len = max(float(np.linalg.norm(d - a)), float(np.linalg.norm(c - b)))
                    stripe_count = max(CROSSWALK_MIN_STRIPE_COUNT, int(round(sweep_len / CROSSWALK_STRIPE_PITCH_PX)))
                    for i in range(stripe_count):
                        t0 = i / stripe_count
                        t1 = (i + 1) / stripe_count
                        center_t = 0.5 * (t0 + t1)
                        half_w = 0.5 * (t1 - t0) * (1.0 - CROSSWALK_STRIPE_GAP_RATIO)
                        s0 = max(0.0, center_t - half_w)
                        s1 = min(1.0, center_t + half_w)
                        q1 = (1 - s0) * a + s0 * d
                        q2 = (1 - s0) * b + s0 * c
                        q3 = (1 - s1) * b + s1 * c
                        q4 = (1 - s1) * a + s1 * d
                        color = CROSSWALK_STRIPE_WHITE if (i % 2 == 0) else CROSSWALK_STRIPE_YELLOW
                        pygame.draw.polygon(surface, color, [q1, q2, q3, q4])
                else:
                    pygame.draw.polygon(surface, CROSSWALK_STRIPE_WHITE, pts)
                pygame.draw.polygon(surface, CROSSWALK_BORDER_COLOR, pts, 1)

            # Draw no-stop lines before crosswalk boundaries (default: 5m).
            if no_stop_before_m > 0.0 and quad_world is not None:
                poly = np.asarray(quad_world, dtype=np.float64)
                if poly.ndim == 2 and poly.shape[0] == 4 and poly.shape[1] >= 2:
                    a = poly[0, :2]
                    b = poly[1, :2]
                    c = poly[2, :2]
                    d = poly[3, :2]
                    forward = (b - a) + (c - d)
                    norm = float(np.linalg.norm(forward))
                    if norm > 1e-6:
                        forward = forward / norm
                        near_start_1 = a - forward * no_stop_before_m
                        near_end_1 = d - forward * no_stop_before_m
                        near_start_2 = b + forward * no_stop_before_m
                        near_end_2 = c + forward * no_stop_before_m
                        p1 = surface.pos2pix(near_start_1[0], near_start_1[1])
                        p2 = surface.pos2pix(near_end_1[0], near_end_1[1])
                        p3 = surface.pos2pix(near_start_2[0], near_start_2[1])
                        p4 = surface.pos2pix(near_end_2[0], near_end_2[1])
                        width = max(CROSSWALK_STOP_LINE_WIDTH_PX, surface.pix(PGDrivableAreaProperty.LANE_LINE_WIDTH))
                        pygame.draw.line(surface, CROSSWALK_STOP_LINE_COLOR, p1, p2, width)
                        pygame.draw.line(surface, CROSSWALK_STOP_LINE_COLOR, p3, p4, width)

    return surface if return_surface else WorldSurface.to_cv2_image(surface)


def draw_top_down_trajectory(
    surface: WorldSurface, episode_data: dict, entry_differ_color=False, exit_differ_color=False, color_list=None
):
    if entry_differ_color or exit_differ_color:
        assert color_list is not None
    color_map = {}
    if not exit_differ_color and not entry_differ_color:
        color_type = 0
    elif exit_differ_color ^ entry_differ_color:
        color_type = 1
    else:
        color_type = 2

    if entry_differ_color:
        # init only once
        if "spawn_roads" in episode_data:
            spawn_roads = episode_data["spawn_roads"]
        else:
            spawn_roads = set()
            first_frame = episode_data["frame"][0]
            for state in first_frame[TARGET_VEHICLES].values():
                spawn_roads.add((state["spawn_road"][0], state["spawn_road"][1]))
        keys = [road[0] for road in list(spawn_roads)]
        keys.sort()
        color_map = {key: color for key, color in zip(keys, color_list)}

    for frame in episode_data["frame"]:
        for k, state, in frame[TARGET_VEHICLES].items():
            if color_type == 0:
                color = color_white
            elif color_type == 1:
                if exit_differ_color:
                    key = state["destination"][1]
                    if key not in color_map:
                        color_map[key] = color_list.pop()
                    color = color_map[key]
                else:
                    color = color_map[state["spawn_road"][0]]
            else:
                key_1 = state["spawn_road"][0]
                key_2 = state["destination"][1]
                if key_1 not in color_map:
                    color_map[key_1] = dict()
                if key_2 not in color_map[key_1]:
                    color_map[key_1][key_2] = color_list.pop()
                color = color_map[key_1][key_2]
            start = state["position"]
            pygame.draw.circle(surface, color, surface.pos2pix(start[0], start[1]), 1)
    for step, frame in enumerate(episode_data["frame"]):
        for k, state in frame[TARGET_VEHICLES].items():
            if not state["done"]:
                continue
            start = state["position"]
            if state["done"]:
                pygame.draw.circle(surface, (0, 0, 0), surface.pos2pix(start[0], start[1]), 5)
    return surface


class TopDownRenderer:
    def __init__(
        self,
        film_size=(2000, 2000),  # draw map in size = film_size/scaling. By default, it is set to 400m
        scaling=5,  # None for auto-scale
        screen_size=(800, 800),
        num_stack=15,
        history_smooth=0,
        show_agent_name=False,
        camera_position=None,
        target_agent_heading_up=False,
        target_vehicle_heading_up=None,
        draw_target_vehicle_trajectory=False,
        semantic_map=False,
        draw_center_line=False,
        semantic_broken_line=True,
        draw_contour=True,
        window=True,
        screen_record=False,
        center_on_map=False,
    ):
        """
        Launch a top-down renderer for current episode. Usually, it is launched by env.render(mode="topdown") and will
        be closed when next env.reset() is called or next episode starts.
        Args:
            film_size: The size of the film used to draw the map. The unit is pixel. It should cover the whole map to
            ensure it is complete in the rendered result. It works with the argument scaling to select the region
            to draw. For example, (2000, 2000) film size with scaling=5 can draw any maps whose width and height
            less than 2000/5 = 400 meters.

            scaling: The scaling determines how many pixels are used to draw one meter.

            screen_size: The size of the window popped up. It shows a region with width and length = screen_size/scaling

            num_stack: How many history steps to keep. History trajectory will show in faded color. It should be > 1

            history_smooth: Smoothing the trajectory by drawing less history positions. This value determines the sample
            rate. By default, this value is 0, meaning positions in previous num_stack steps will be shown.

            show_agent_name: Draw the name of the agent.

            camera_position: Set the (x,y) position of the top_down camera. If it is not specified, the camera will move
            with the ego car.

            target_agent_heading_up: Whether to rotate the camera according to the ego agent's heading. When enabled,
            The ego car always faces upwards.

            target_vehicle_heading_up: Deprecated, use target_agent_heading_up instead!

            draw_target_vehicle_trajectory: Whether to draw the ego car's whole trajectory without faded color

            semantic_map: Whether to draw semantic color for each object. The color scheme is in TopDownSemanticColor.

            draw_center_line: Whether to draw center line for each lane, this can be used to debug the lane connectivity

            semantic_broken_line: Whether to draw broken line for semantic map

            draw_contour: Whether to draw a counter for objects

            window: Whether to pop up the window. Setting it to 'False' enables off-screen rendering

            screen_record: Whether to record the episode. The recorded result can be accessed by
            env.top_down_renderer.screen_frames or env.top_down_renderer.generate_gif(file_name, fps)

            center_on_map: Whether to center the camera on the map. If set to True, the camera will not move with the
            ego car, and the camera position will be fixed at the center of the map.
        """
        # doc-end
        # LQY: do not delete the above line !!!!!

        # Setup some useful flags

        self.sign_icon_raw = {}
        self.sign_icon_surfaces = {}
        if hasattr(self.engine, 'traffic_sign_manager'):
            sign_mgr = self.engine.traffic_sign_manager
            for sign in sign_mgr.signs:
                sign_type = type(sign).__name__
                icon_path = sign.icon_path
                if icon_path is not None:
                    try:
                        self.sign_icon_raw[sign_type] = pygame.image.load(icon_path)
                    except Exception as e:
                        print(f"Failed to load icon for {sign_type}: {e}")

        self.logger = get_logger()
        if num_stack < 1:
            self.logger.warning("num_stack should be greater than 0. Current value: {}. Set to 1".format(num_stack))
            num_stack = 1

        if target_vehicle_heading_up is not None:
            self.logger.warning("target_vehicle_heading_up is deprecated! Use target_agent_heading_up instead!")
            assert target_agent_heading_up is False
            target_agent_heading_up = target_vehicle_heading_up

        self.position = camera_position
        self.center_on_map = center_on_map
        self.target_agent_heading_up = target_agent_heading_up
        self.show_agent_name = show_agent_name
        self.draw_target_vehicle_trajectory = draw_target_vehicle_trajectory
        self.contour = draw_contour
        self.semantic_broken_line = semantic_broken_line
        self.no_window = not window

        if self.show_agent_name:
            pygame.init()

        self.screen_record = screen_record
        self._screen_frames = []
        self.pygame_font = None
        self.map = self.engine.current_map
        self.stack_frames = deque(maxlen=num_stack)
        self.history_objects = deque(maxlen=num_stack)
        self.history_target_vehicle = []
        self.history_smooth = history_smooth
        # self.current_track_agent = current_track_agent
        if self.target_agent_heading_up:
            assert self.current_track_agent is not None, "Specify which vehicle to track"
        self._text_render_pos = [50, 50]
        self._font_size = 25
        self._text_render_interval = 20
        self.semantic_map = semantic_map
        self.scaling = scaling
        self.film_size = film_size
        self._screen_size = screen_size

        # Setup the canvas
        # (1) background is the underlying layer that draws map.
        # It is fixed and will never change unless the map changes.
        self._background_canvas = draw_top_down_map_native(
            self.map,
            draw_center_line=draw_center_line,
            scaling=self.scaling,
            semantic_map=self.semantic_map,
            return_surface=True,
            film_size=self.film_size,
            semantic_broken_line=self.semantic_broken_line
        )
        self.scaling = self._background_canvas.scaling

        # (2) frame is a copy of the background so you can draw movable things on it.
        # It is super large as the background.
        self._frame_canvas = self._background_canvas.copy()

        # (3) canvas_rotate is only used when target_vehicle_heading_up=True and is use to center the tracked agent.
        if self.target_agent_heading_up:
            max_screen_size = max(self._screen_size[0], self._screen_size[1])
            self.canvas_rotate = pygame.Surface((max_screen_size * 2, max_screen_size * 2))

        # (4) screen_canvas is a regional surface where only part of the super large background will draw.
        # This will be used to as the final image shown in the screen & saved.
        self._screen_canvas = pygame.Surface(self._screen_size
                                             ) if self.no_window else pygame.display.set_mode(self._screen_size)
        self._screen_canvas.set_alpha(None)
        self._screen_canvas.fill(color_white)

        # Draw
        self.blit()

        # key accept
        self.need_reset = False

    @property
    def screen_canvas(self):
        return self._screen_canvas

    def refresh(self):
        self._frame_canvas.blit(self._background_canvas, (0, 0))
        self.screen_canvas.fill(color_white)

    def render(self, text, to_image=True, *args, **kwargs):
        if "semantic_map" in kwargs:
            self.semantic_map = kwargs["semantic_map"]

        self.need_reset = False
        if not self.no_window:
            key_press = pygame.key.get_pressed()
            if key_press[pygame.K_r]:
                self.need_reset = True

        # Record current target vehicle
        objects = self.engine.get_objects(lambda obj: not is_map_related_instance(obj))

        # Exclude signs from objects before adding to history_objects
        if hasattr(self.engine, 'traffic_sign_manager'):
            sign_mgr = self.engine.traffic_sign_manager
            if sign_mgr and sign_mgr.signs:
                sign_names = {sign.name for sign in sign_mgr.signs if sign is not None and hasattr(sign, 'name')}
                objects = {name: obj for name, obj in objects.items() if name not in sign_names}

        this_frame_objects = self._append_frame_objects(objects)
        self.history_objects.append(this_frame_objects)

        if self.draw_target_vehicle_trajectory:
            self.history_target_vehicle.append(
                history_object(
                    type=MetaDriveType.VEHICLE,
                    name=self.current_track_agent.name,
                    heading_theta=self.current_track_agent.heading_theta,
                    WIDTH=self.current_track_agent.top_down_width,
                    LENGTH=self.current_track_agent.top_down_length,
                    position=self.current_track_agent.position,
                    color=self.current_track_agent.top_down_color,
                    done=False
                )
            )

        self._handle_event()
        self.refresh()
        self._draw(*args, **kwargs)
        self._add_text(text)
        self.blit()
        ret = self.screen_canvas
        if not self.no_window:
            ret = ret.convert(24)
        ret = WorldSurface.to_cv2_image(ret) if to_image else ret
        if self.screen_record:
            self._screen_frames.append(ret)
        return ret

    def generate_gif(self, gif_name="demo.gif", duration=30):
        return generate_gif(self._screen_frames, gif_name, is_pygame_surface=False, duration=duration)

    def _add_text(self, text: dict):
        if not text:
            return
        if not pygame.get_init():
            pygame.init()
        font2 = pygame.font.SysFont('didot.ttc', 25)
        # pygame do not support multiline text render
        count = 0
        for key, value in text.items():
            line = str(key) + ":" + str(value)
            img2 = font2.render(line, True, (0, 0, 0))
            this_line_pos = copy.copy(self._text_render_pos)
            this_line_pos[-1] += count * self._text_render_interval
            self._screen_canvas.blit(img2, this_line_pos)
            count += 1

    def blit(self):
        if not self.no_window:
            pygame.display.update()

    def close(self):
        self.clear()
        pygame.quit()

    def clear(self):
        # # Reset the super large background
        self._background_canvas = None

        # Reset several useful variables.
        self._frame_canvas = None
        self.canvas_rotate = None

        self.history_objects.clear()
        self.stack_frames.clear()
        self.history_target_vehicle.clear()
        self.screen_frames.clear()

    @property
    def current_track_agent(self):
        return self.engine.current_track_agent

    def _draw_lane_arrows(self, lane, turns, map_data):
        """Draw direction arrows for a lane using its geometry and turn info.

        Uses the same pullback (6m) as _draw_lane_thin_arrows so all arrows
        appear at the same distance from the lane end.
        """
        if not turns:
            return
        pullback = 5.0
        long_pos = max(0.1, lane.length - pullback)
        pos = lane.position(long_pos, 0)
        heading = lane.heading_theta_at(long_pos)
        width = lane.width_at(long_pos)

        dirs = [t["direction"] for t in turns]

        uturn_side = +1
        for t in turns:
            if t["direction"] == 't':
                to_id = t.get("to_lane")
                if to_id and to_id in map_data:
                    dest_poly = map_data[to_id].get("polyline")
                    uturn_side = _uturn_side_from_dest(
                        float(pos[0]), float(pos[1]), heading, dest_poly
                    )
                break

        _draw_thin_arrows_at(
            self._frame_canvas, float(pos[0]), float(pos[1]),
            heading, dirs, width, uturn_side=uturn_side
        )

    def _is_sign_obj(self, v):
        """Check if a history_object or live object is a traffic sign."""
        sign_mgr = getattr(self.engine, 'traffic_sign_manager', None)
        if sign_mgr:
            if any(s is not None and getattr(s, 'name', None) == v.name for s in sign_mgr.signs):
                return True
        if getattr(v, 'type', None) == MetaDriveType.TRAFFIC_OBJECT:
            if getattr(v, 'WIDTH', 2) < 1.0 and getattr(v, 'LENGTH', 2) < 1.0:
                return True
        return False

    @staticmethod
    def _append_frame_objects(objects):
        frame_objects = []
        for name, obj in objects.items():
            frame_objects.append(
                history_object(
                    name=name,
                    type=obj.metadrive_type if hasattr(obj, "metadrive_type") else MetaDriveType.OTHER,
                    heading_theta=obj.heading_theta,
                    WIDTH=obj.top_down_width,
                    LENGTH=obj.top_down_length,
                    position=obj.position,
                    color=obj.top_down_color,
                    done=False
                )
            )
        return frame_objects

    def _draw(self, *args, **kwargs):
        """
        This is the core function to process the
        """
        if len(self.history_objects) == 0:
            return

        for i, objects in enumerate(self.history_objects):
            if i == len(self.history_objects) - 1:
                continue
            i = len(self.history_objects) - i
            if self.history_smooth != 0 and (i % self.history_smooth != 0):
                continue
            for v in objects:
                if self._is_sign_obj(v):
                    continue
                c = v.color
                h = v.heading_theta
                h = h if abs(h) > 2 * np.pi / 180 else 0
                x = abs(int(i))
                alpha_f = x / len(self.history_objects)
                if self.semantic_map:
                    c = TopDownSemanticColor.get_color(v.type) * (1 - alpha_f) + alpha_f * 255
                else:
                    c = (c[0] + alpha_f * (255 - c[0]), c[1] + alpha_f * (255 - c[1]), c[2] + alpha_f * (255 - c[2]))
                ObjectGraphics.display(object=v, surface=self._frame_canvas, heading=h, color=c, draw_contour=False)

        # Draw the whole trajectory of ego vehicle with no gradient colors:
        if self.draw_target_vehicle_trajectory:
            for i, v in enumerate(self.history_target_vehicle):
                i = len(self.history_target_vehicle) - i
                if self.history_smooth != 0 and (i % self.history_smooth != 0):
                    continue
                c = v.color
                h = v.heading_theta
                h = h if abs(h) > 2 * np.pi / 180 else 0
                x = abs(int(i))
                alpha_f = min(x / len(self.history_target_vehicle), 0.5)
                # alpha_f = 0
                ObjectGraphics.display(
                    object=v,
                    surface=self._frame_canvas,
                    heading=h,
                    color=(c[0] + alpha_f * (255 - c[0]), c[1] + alpha_f * (255 - c[1]), c[2] + alpha_f * (255 - c[2])),
                    draw_contour=False
                )

        # Draw current vehicle with black contour
        # Use this line if you wish to draw "future" trajectory.
        # i is the index of vehicle that we will render a black box for it.
        # i = int(len(self.history_vehicles) / 2)
        i = -1
        for v in self.history_objects[i]:
            if self._is_sign_obj(v):
                continue
            h = v.heading_theta
            c = v.color
            h = h if abs(h) > 2 * np.pi / 180 else 0
            alpha_f = 0
            if self.semantic_map:
                c = TopDownSemanticColor.get_color(v.type) * (1 - alpha_f) + alpha_f * 255
            else:
                c = (c[0] + alpha_f * (255 - c[0]), c[1] + alpha_f * (255 - c[1]), c[2] + alpha_f * (255 - c[2]))
            ObjectGraphics.display(
                object=v, surface=self._frame_canvas, heading=h, color=c, draw_contour=self.contour, contour_width=2
            )

        if not hasattr(self, "_deads"):
            self._deads = []

        for v in self._deads:
            pygame.draw.circle(
                surface=self._frame_canvas,
                color=(255, 0, 0),
                center=self._frame_canvas.pos2pix(v.position[0], v.position[1]),
                radius=5
            )

        for v in self.history_objects[i]:
            if v.done:
                pygame.draw.circle(
                    surface=self._frame_canvas,
                    color=(255, 0, 0),
                    center=self._frame_canvas.pos2pix(v.position[0], v.position[1]),
                    radius=5
                )
                self._deads.append(v)

        # Load sign icons dynamically if not already loaded
        if hasattr(self.engine, 'traffic_sign_manager'):
            sign_mgr = self.engine.traffic_sign_manager
            for sign in sign_mgr.signs:
                sign_type = type(sign).__name__
                # Load icon if not already loaded
                if sign_type not in self.sign_icon_raw and hasattr(sign, 'icon_path') and sign.icon_path:
                    try:
                        loaded_img = pygame.image.load(sign.icon_path)
                        # Check if image is valid (not empty)
                        if loaded_img.get_size()[0] > 0 and loaded_img.get_size()[1] > 0:
                            self.sign_icon_raw[sign_type] = loaded_img
                    except Exception as e:
                        print(f"Failed to load icon for {sign_type} from {sign.icon_path}: {e}")

        # Create surfaces from raw icons if not already created
        # Only create surfaces for icons that are not yet created
        for name, img in self.sign_icon_raw.items():
            if name not in self.sign_icon_surfaces:
                try:
                    scaled = pygame.transform.smoothscale(img, (24, 24))
                    # Check if scaled image is valid
                    if scaled.get_size()[0] > 0 and scaled.get_size()[1] > 0:
                        # For images without alpha (like JPG), convert to surface with alpha
                        # For PNG images, they should already have alpha
                        try:
                            # Try to preserve alpha if it exists
                            if scaled.get_flags() & pygame.SRCALPHA:
                                self.sign_icon_surfaces[name] = scaled
                            else:
                                # Convert to surface with alpha channel for JPG images
                                self.sign_icon_surfaces[name] = scaled.convert_alpha()
                        except Exception:
                            # Fallback: use as-is
                            self.sign_icon_surfaces[name] = scaled
                except Exception as e:
                    print(f"Failed to create surface for {name}: {e}")

        if (self.current_track_agent is not None and
            hasattr(self.current_track_agent, 'navigation') and
            hasattr(self.current_track_agent.navigation, 'final_lane') and
            self.current_track_agent.navigation.final_lane is not None):

            final_lane = self.current_track_agent.navigation.final_lane
            target_pos = final_lane.position(final_lane.length, 0)

            pixel_x, pixel_y = self._frame_canvas.pos2pix(target_pos[0], target_pos[1])

            pygame.draw.circle(
                surface=self._frame_canvas,
                color=(255, 0, 0),
                center=(pixel_x, pixel_y),
                radius=6,
                width=0
            )
            pygame.draw.circle(
                surface=self._frame_canvas,
                color=(255, 255, 255),
                center=(pixel_x, pixel_y),
                radius=6,
                width=2
            )
        if (isinstance(self.current_track_agent.navigation, EdgeNetworkNavigation) and self.current_track_agent is not None and
            hasattr(self.current_track_agent, 'navigation') and
            hasattr(self.current_track_agent.navigation, 'checkpoints')):

            checkpoints = self.current_track_agent.navigation.checkpoints
            route_color = (0, 255, 0)
            route_width = 3
            map_data = self.map.blocks[-1].map_data
            route_points = []

            for lane_id in checkpoints:
                if lane_id in map_data and "polyline" in map_data[lane_id]:
                    route_points.extend(map_data[lane_id]["polyline"])
            if len(route_points) > 1:
                for i in range(len(route_points) - 1):
                    start = self._frame_canvas.pos2pix(route_points[i][0], route_points[i][1])
                    end = self._frame_canvas.pos2pix(route_points[i+1][0], route_points[i+1][1])
                    pygame.draw.line(self._frame_canvas, route_color, start, end, route_width)

        if hasattr(self.engine, 'traffic_sign_manager'):
            sign_mgr = self.engine.traffic_sign_manager
            for sign in sign_mgr.signs:
                if sign is None:
                    continue
                sign_type = type(sign).__name__

                if sign_type == "DirectionSign":
                    continue

                if sign_type == "TrafficLightSign":
                    if hasattr(sign, 'position') and hasattr(sign, 'top_down_color'):
                        pixel_x, pixel_y = self._frame_canvas.pos2pix(sign.position[0], sign.position[1])

                        color = sign.top_down_color

                        radius = 8
                        pygame.draw.circle(
                            surface=self._frame_canvas,
                            color=color,
                            center=(pixel_x, pixel_y),
                            radius=radius
                        )

                icon = self.sign_icon_surfaces.get(sign_type)

                if icon is not None and hasattr(sign, 'position'):
                    try:
                        icon_size = icon.get_size()
                        if icon_size[0] > 0 and icon_size[1] > 0:
                            pixel_x, pixel_y = self._frame_canvas.pos2pix(sign.position[0], sign.position[1])
                            # Compute rotation from sign heading:
                            # heading_theta = lane_heading + pi/2 (perpendicular to lane)
                            # For pygame: rotate so icon faces the driver
                            rotation_deg = getattr(sign, 'top_down_icon_rotation_deg', None)
                            if rotation_deg is None:
                                sign_heading = getattr(sign, 'heading_theta', 0.0)
                                rotation_deg = float(np.rad2deg(sign_heading) - 180.0)
                            draw_icon = pygame.transform.rotate(icon, rotation_deg)
                            rect = draw_icon.get_rect(center=(pixel_x, pixel_y))
                            self._frame_canvas.blit(draw_icon, rect)
                    except Exception:
                        pass

        v = self.current_track_agent
        canvas = self._frame_canvas
        field = self._screen_canvas.get_size()
        if not self.target_agent_heading_up:
            if self.position is not None or v is not None:
                if self.center_on_map:
                    frame_canvas_size = self._frame_canvas.get_size()
                    position = (frame_canvas_size[0] / 2, frame_canvas_size[1] / 2)
                else:
                    cam_pos = (self.position or v.position)
                    position = self._frame_canvas.pos2pix(*cam_pos)
            else:
                position = (field[0] / 2, field[1] / 2)
            off = (position[0] - field[0] / 2, position[1] - field[1] / 2)
            self.screen_canvas.blit(source=canvas, dest=(0, 0), area=(off[0], off[1], field[0], field[1]))
        else:
            position = self._frame_canvas.pos2pix(*v.position)
            area = (
                position[0] - self.canvas_rotate.get_size()[0] / 2, position[1] - self.canvas_rotate.get_size()[1] / 2,
                self.canvas_rotate.get_size()[0], self.canvas_rotate.get_size()[1]
            )
            self.canvas_rotate.fill(color_white)
            self.canvas_rotate.blit(source=canvas, dest=(0, 0), area=area)

            rotation = -np.rad2deg(v.heading_theta) + 90
            new_canvas = pygame.transform.rotozoom(self.canvas_rotate, rotation, 1)

            size = self._screen_canvas.get_size()
            self._screen_canvas.blit(
                new_canvas,
                (0, 0),
                (
                    new_canvas.get_size()[0] / 2 - size[0] / 2,  # Left
                    new_canvas.get_size()[1] / 2 - size[1] / 2,  # Top
                    size[0],  # Width
                    size[1]  # Height
                )
            )

        if self.show_agent_name:
            raise ValueError("This function is broken")
            # FIXME check this later
            if self.pygame_font is None:
                self.pygame_font = pygame.font.SysFont("Arial.ttf", 30)
            agents = [agent.name for agent in list(self.engine.agents.values())]
            for v in self.history_objects[i]:
                if v.name in agents:
                    position = self._frame_canvas.pos2pix(*v.position)
                    new_position = (position[0] - off[0], position[1] - off[1])
                    img = self.pygame_font.render(
                        text=self.engine.object_to_agent(v.name),
                        antialias=True,
                        color=(0, 0, 0, 128),
                    )
                    # img.set_alpha(None)
                    self.screen_canvas.blit(
                        source=img,
                        dest=(new_position[0] - img.get_width() / 2, new_position[1] - img.get_height() / 2),
                        # special_flags=pygame.BLEND_RGBA_MULT
                    )

    def _handle_event(self) -> None:
        """
        Handle pygame events for moving and zooming in the displayed area.
        """
        if self.no_window:
            return
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    import sys
                    sys.exit()

    @property
    def engine(self):
        from metadrive.engine.engine_utils import get_engine
        return get_engine()

    @property
    def screen_frames(self):
        return copy.deepcopy(self._screen_frames)

    def get_map(self):
        """
        Convert the map pygame surface to array

        Returns: map in array

        """
        return pygame.surfarray.array3d(self._background_canvas)
