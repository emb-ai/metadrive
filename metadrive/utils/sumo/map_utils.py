from __future__ import \
    annotations  # https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class

import logging

import numpy as np
from metadrive.scenario import ScenarioDescription as SD
from metadrive.type import MetaDriveType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dataclasses import dataclass
from typing import List, Dict, Optional
try:
    import sumolib
except ImportError:
    raise ImportError("Please install sumolib before running this script via: pip install sumolib")
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.base import CAP_STYLE, JOIN_STYLE
from shapely.ops import unary_union
from collections import defaultdict


def buffered_shape(shape, width: float = 1.0) -> Polygon:
    """Generates a shape with a buffer of `width` around the original shape."""
    ls = LineString(shape).buffer(
        width / 2,
        1,
        cap_style=CAP_STYLE.flat,
        join_style=JOIN_STYLE.round,
        mitre_limit=5.0,
    )
    if isinstance(ls, MultiPolygon):
        # Sometimes it oddly outputs a MultiPolygon and then we need to turn it into a convex hull
        ls = ls.convex_hull
    elif not isinstance(ls, Polygon):
        raise RuntimeError("Shapely `object.buffer` behavior may have changed.")
    return ls


class LaneShape:
    def __init__(
        self,
        shape,
        width: float,
    ):
        """
        Lane shape
        """
        shape = buffered_shape(shape.getShape(), shape.getWidth())
        self.shape = shape


@dataclass
class RoadShape:
    left_border: np.ndarray
    right_border: np.ndarray

    def __post_init__(self):
        """
        post process
        """
        self.polygon = self.left_border + list(reversed(self.right_border))


class JunctionNode:
    def __init__(self, sumolib_obj):
        """Node for junction node."""
        self.sumolib_obj: sumolib.net.node = sumolib_obj
        self.name = sumolib_obj.getID()
        self.type = sumolib_obj.getType()
        self.shape = sumolib_obj.getShape()
        self.area: float = 0.0
        self.route_dist: float = 0.0

        self.incoming: List[RoadNode] = []
        self.outgoing: List[RoadNode] = []
        self.roads: List[RoadNode] = []
        self.lanes: List[LaneNode] = []


class LaneNode:
    # Class-level minimum lane width (metres).  Set before graph construction
    # to widen all driving lanes, e.g.  ``LaneNode.MIN_LANE_WIDTH = 4.5``.
    MIN_LANE_WIDTH = 0.0

    def __init__(self, sumolib_obj):
        """
        Node for a lane
        """
        self.sumolib_obj: sumolib.net.lane = sumolib_obj
        self.name: str = sumolib_obj.getID()
        self.edge_type: str = sumolib_obj.getEdge().getType()
        self.index: int = sumolib_obj.getIndex()
        if len(self.edge_type.strip()) and len(self.edge_type.split('|')) > 1:
            self.type = self.edge_type.split('|')[self.index]
        elif self.edge_type == 'highway.service':
            # Считаем service дороги пригодными для автомобилей (для привязки знаков)
            self.type = 'driving'
        elif (sumolib_obj.allows('pedestrian') and not sumolib_obj.allows('passenger')):
            self.type = 'sidewalk'
        elif sumolib_obj.getEdge().getFunction() == 'walkingarea':
            self.type = 'sidewalk'
        else:
            self.type = 'driving'
        raw_width = sumolib_obj.getWidth()
        if self.type == 'driving' and LaneNode.MIN_LANE_WIDTH > 0:
            self.width: float = max(raw_width, LaneNode.MIN_LANE_WIDTH)
        else:
            self.width: float = raw_width
        self.length: float = sumolib_obj.getLength()
        self.speed: float = sumolib_obj.getSpeed()
        self.shape: LaneShape = LaneShape(sumolib_obj, self.width)

        if sumolib_obj.getEdge().getFunction() == 'walkingarea':
            shape = [[p[0], p[1]] for p in sumolib_obj.getShape()]
            shape.append(sumolib_obj.getShape()[0])
            self.shape.shape = Polygon(shape)

        self.road = None
        self.left_neigh: Optional[LaneNode] = None
        self.right_neigh: Optional[LaneNode] = None
        self.incoming: List[LaneNode] = []
        self.outgoing: List[LaneNode] = []
        self.function: Optional[str] = None


class RoadNode:
    def __init__(
        self,
        sumolib_obj,
        lanes,
        from_junction,
        to_junction,
    ):
        """
        Node for a road
        """
        self.sumolib_obj: sumolib.net.edge = sumolib_obj
        self.name: str = sumolib_obj.getID()
        self.type = sumolib_obj.getType()
        if sumolib_obj.getFunction() == 'crossing':
            for lane in lanes:
                lane.type = 'crossing'
        self.lanes: List[LaneNode] = lanes
        self.from_junction: JunctionNode = from_junction
        self.to_junction: JunctionNode = to_junction
        self.width: float = sum([lane.width for lane in lanes])
        self.length: float = max([lane.length for lane in lanes])
        self.priority: int = sumolib_obj.getPriority()
        self.function: str = sumolib_obj.getFunction()

        # leftmost_lane = list(filter(lambda l: l.left_neigh is None, lanes))[0]
        # rightmost_lane = list(filter(lambda l: l.right_neigh is None, lanes))[0]
        # road_shape = RoadShape(
        #     leftmost_lane.shape.left_border,
        #     rightmost_lane.shape.right_border,
        # )
        # self.shape: RoadShape = road_shape

        for lane in self.lanes:  # Link to parent road
            lane.road = self
            lane.function = self.function

        self.junction: Optional[JunctionNode] = None
        self.incoming: List[RoadNode] = []
        self.outgoing: List[RoadNode] = []


class RoadLaneJunctionGraph:
    def __init__(
        self,
        sumo_net_path,
    ):
        """Init the graph"""

        self.sumo_net = sumolib.net.readNet(
            sumo_net_path, withInternal=True, withPedestrianConnections=True, withPrograms=True
        )

        xmin, ymin, xmax, ymax = self.sumo_net.getBoundary()
        center_x = (xmax + xmin) / 2
        center_y = (ymax + ymin) / 2
        self.sumo_net.move(-center_x, -center_y)

        self.tls = self.sumo_net.getTrafficLights()
        self.traffic_lights = []
        self.lane_to_tl_signals = {}

        for tl in self.tls:
            tl_info = {
                'id': tl.getID(),
                'controlled_lanes': [],        # заполним позже
                'programs': tl.getPrograms()
            }
            # Собираем управляемые полосы
            for link in tl.getConnections():
                from_lane = link[0].getID()
                to_lane = link[1].getID()
                tl_info['controlled_lanes'].append({
                    'from': from_lane,
                    'to': to_lane,
                    'tl_index': link[2]
                })
            # print("tl_info: ", tl_info)
            self.traffic_lights.append(tl_info)


            # Создаем маппинг from_lane -> сигналы светофора
            for conn in tl_info['controlled_lanes']:
                from_lane = conn['from']
                to_lane = conn['to']
                tl_index = conn['tl_index']

                phases = []
                for _, program in tl_info['programs'].items():
                    program_phases = program.getPhases()
                    # print("program_phases:  ", program_phases)
                    for phase in program_phases:
                        if tl_index < len(phase.state):
                            state = phase.state[tl_index]
                            duration = phase.duration
                            phases.append((state, duration))

                if from_lane not in self.lane_to_tl_signals:
                    self.lane_to_tl_signals[from_lane] = []

                self.lane_to_tl_signals[from_lane].append({
                    'to_lane': to_lane,
                    'phases': phases,
                    'tl_id': tl_index
                })

        self.roads: Dict[str, RoadNode] = {}
        self.lanes: Dict[str, LaneNode] = {}
        self.junctions: Dict[str, JunctionNode] = {}

        for edge in self.sumo_net.getEdges(withInternal=True):  # Normal edges first
            lanes = []
            lane_index_to_lane = {}
            for lane in edge.getLanes():  # Create initial LaneNode objects
                lane_node = LaneNode(lane)
                self.lanes[lane_node.name] = lane_node
                lanes.append(lane_node)
                lane_index_to_lane[lane.getIndex()] = lane_node

            for lane in lanes:  # Setting left and right neighbors
                if lane.index - 1 in lane_index_to_lane:
                    lane.right_neigh = lane_index_to_lane[lane.index - 1]
                if lane.index + 1 in lane_index_to_lane:
                    lane.left_neigh = lane_index_to_lane[lane.index + 1]

            junctions = []  # Create initial JunctionNode objects connected to current road
            for i, node in enumerate([edge.getFromNode(), edge.getToNode()]):
                name = node.getID()

                if node.getID() not in self.junctions:

                    junction_node = JunctionNode(node)
                    self.junctions[name] = junction_node
                else:
                    junction_node = self.junctions[name]
                junctions.append(junction_node)

            # Create RoadShape for Road
            name = edge.getID()
            road_node = RoadNode(
                edge,
                lanes,
                junctions[0],  # from_node
                junctions[1],  # to_node
            )
            self.roads[name] = road_node

        for junction_id, junction in self.junctions.items():
            junction.sumolib_obj.setShape(
                [(x - center_x, y - center_y, z) for x, y, z in junction.sumolib_obj.getShape3D()]
            )
            junction.shape = junction.sumolib_obj.getShape()

        for junction_id, junction in self.junctions.items():

            for incoming in junction.sumolib_obj.getIncoming():  # Link junction
                junction.incoming.append(self.roads[incoming.getID()])
            for outgoing in junction.sumolib_obj.getOutgoing():
                junction.outgoing.append(self.roads[outgoing.getID()])

            conns = junction.sumolib_obj.getConnections()
            for conn in conns:
                from_lane_id = conn.getFromLane().getID()
                to_lane_id = conn.getToLane().getID()
                via_lane_id = conn.getViaLaneID()
                from_road_id = conn.getFrom().getID()  # Link roads
                to_road_id = conn.getTo().getID()
                if via_lane_id == '':  # Maybe we could skip this, but not sure
                    self.lanes[from_lane_id].outgoing.append(self.lanes[to_lane_id])
                    self.lanes[to_lane_id].incoming.append(self.lanes[from_lane_id])
                    self.roads[from_road_id].outgoing.append(self.roads[to_road_id])
                    self.roads[to_road_id].incoming.append(self.roads[from_road_id])
                else:
                    via_road_id = self.sumo_net.getLane(conn.getViaLaneID()).getEdge().getID()
                    self.lanes[from_lane_id].outgoing.append(self.lanes[via_lane_id])
                    self.lanes[to_lane_id].incoming.append(self.lanes[via_lane_id])
                    self.lanes[via_lane_id].incoming.append(self.lanes[from_lane_id])
                    self.lanes[via_lane_id].outgoing.append(self.lanes[to_lane_id])
                    self.roads[from_road_id].outgoing.append(self.roads[via_road_id])
                    self.roads[to_road_id].incoming.append(self.roads[via_road_id])
                    self.roads[via_road_id].incoming.append(self.roads[from_road_id])
                    self.roads[via_road_id].outgoing.append(self.roads[to_road_id])

                    junction.roads.append(self.roads[via_road_id])  # Add roads/lanes to junction
                    junction.lanes.append(self.lanes[via_lane_id])
                    self.roads[via_road_id].junction = junction  # Add junction reference

        lane_dividers, edge_dividers, road_boundaries = self._compute_traffic_dividers()

        self.lane_dividers = lane_dividers
        self.edge_dividers = edge_dividers
        self.road_boundaries = road_boundaries

    def _compute_traffic_dividers(self, threshold=1):
        """Find the road dividers"""
        lane_dividers = []  # divider between lanes with same traffic direction
        edge_dividers = []  # divider between lanes with opposite traffic direction
        edge_borders = []
        for edge in self.sumo_net.getEdges():
            if edge.getFunction() in ["internal", "walkingarea", 'crossing']:
                continue

            lanes = edge.getLanes()
            for i in range(len(lanes)):
                lane_obj = self.lanes.get(lanes[i].getID())
                if lane_obj and lane_obj.type != 'driving':
                    continue
                shape = lanes[i].getShape()
                left_side = sumolib.geomhelper.move2side(shape, -lanes[i].getWidth() / 2)
                right_side = sumolib.geomhelper.move2side(shape, lanes[i].getWidth() / 2)

                if i == 0:
                    edge_borders.append(right_side)

                if i == len(lanes) - 1:
                    edge_borders.append(left_side)
                else:
                    lane_dividers.append(left_side)

        # The edge borders that overlapped in positions form an edge divider
        for i in range(len(edge_borders) - 1):
            for j in range(i + 1, len(edge_borders)):
                edge_border_i = np.array([edge_borders[i][0], edge_borders[i][-1]])  # start and end position
                edge_border_j = np.array(
                    [edge_borders[j][-1], edge_borders[j][0]]
                )  # start and end position with reverse traffic direction

                # The edge borders of two lanes do not always overlap perfectly, thus relax the tolerance threshold to 1
                if np.linalg.norm(edge_border_i - edge_border_j) < threshold:
                    edge_dividers.append(edge_borders[i])

        # Use the exterior contour of merged driving area as true road boundaries.
        road_boundaries = self._extract_external_road_boundaries()

        return lane_dividers, edge_dividers, road_boundaries

    def _extract_external_road_boundaries(self, min_length=3.0):
        driving_polygons = []
        for lane in self.lanes.values():
            if lane.type != "driving":
                continue
            if lane.function in ["internal", "walkingarea", "crossing"]:
                continue
            poly = getattr(getattr(lane, "shape", None), "shape", None)
            if not isinstance(poly, Polygon) or poly.is_empty:
                continue
            driving_polygons.append(poly)

        if not driving_polygons:
            return []

        merged = unary_union(driving_polygons)
        if isinstance(merged, Polygon):
            polygons = [merged]
        elif isinstance(merged, MultiPolygon):
            polygons = list(merged.geoms)
        else:
            polygons = []

        road_boundaries = []
        for poly in polygons:
            coords = list(poly.exterior.coords)
            if len(coords) < 2:
                continue
            if LineString(coords).length < min_length:
                continue
            road_boundaries.append([(float(x), float(y)) for x, y in coords])
        return road_boundaries


def extract_map_features(graph):
    """This func extracts the map features like lanes/lanelines from the SUMO map"""
    from shapely.geometry import Polygon

    ret = {}

    # Build junction polygons (intersection areas) and their outer boundary lines
    for junction_id, junction in graph.junctions.items():
        if len(junction.shape) <= 2:
            continue
        boundary_polygon = Polygon(junction.shape)
        if not boundary_polygon.is_valid or boundary_polygon.is_empty:
            continue
        boundary_coords = [(x, y) for x, y in boundary_polygon.exterior.coords]
        id = "junction_{}".format(junction.name)
        ret[id] = {
            SD.TYPE: MetaDriveType.LANE_SURFACE_STREET,
            SD.POLYLINE: junction.shape,
            SD.POLYGON: boundary_coords,
        }

    #     # Collect lane endpoints at the junction side to identify road openings
    #     lane_endpoints = []
    #     for road in junction.incoming:
    #         for lane in road.lanes:
    #             if lane.type != 'driving':
    #                 continue
    #             shape = lane.sumolib_obj.getShape()
    #             lane_endpoints.append(np.array(shape[-1][:2]))  # end of lane → junction
    #     for road in junction.outgoing:
    #         for lane in road.lanes:
    #             if lane.type != 'driving':
    #                 continue
    #             shape = lane.sumolib_obj.getShape()
    #             lane_endpoints.append(np.array(shape[0][:2]))  # start of lane → junction

    #     # Filter: keep only perimeter segments whose midpoint is far from any lane endpoint
    #     # (segments near lane endpoints are road openings, not outer walls)
    #     open_threshold = 3.0  # meters
    #     seg_idx = 0
    #     current_segment = []
    #     for k in range(len(boundary_coords) - 1):
    #         p0 = np.array(boundary_coords[k][:2])
    #         p1 = np.array(boundary_coords[k + 1][:2])
    #         mid = (p0 + p1) / 2
    #         is_opening = any(np.linalg.norm(mid - ep) < open_threshold for ep in lane_endpoints)
    #         if not is_opening:
    #             if not current_segment:
    #                 current_segment.append(tuple(p0))
    #             current_segment.append(tuple(p1))
    #         else:
    #             if len(current_segment) >= 2:
    #                 ret["junction_boundary_{}_{}".format(junction.name, seg_idx)] = {
    #                     SD.TYPE: MetaDriveType.BOUNDARY_LINE,
    #                     SD.POLYLINE: current_segment,
    #                 }
    #                 seg_idx += 1
    #             current_segment = []
    #     if len(current_segment) >= 2:
    #         ret["junction_boundary_{}_{}".format(junction.name, seg_idx)] = {
    #             SD.TYPE: MetaDriveType.BOUNDARY_LINE,
    #             SD.POLYLINE: current_segment,
    #         }

    # Сначала создадим все полосы без связей
    lane_names = set()
    for road_id, road in graph.roads.items():
        for lane in road.lanes:
            id = "lane_{}".format(lane.name)
            lane_names.add(id)
            boundary_polygon = [(x, y) for x, y in lane.shape.shape.exterior.coords]
            if lane.type == 'driving':
                ret[id] = {
                    SD.TYPE: MetaDriveType.LANE_SURFACE_STREET,
                    SD.POLYLINE: lane.sumolib_obj.getShape(),
                    SD.POLYGON: boundary_polygon,
                    SD.WIDTH: lane.width,
                    "speed": lane.speed,
                    # Заглушки — заполним ниже
                    "entry_lanes": [],
                    "turns": [],
                    "exit_lanes": [],
                    "left_lanes": [],
                    "right_lanes": [],
                    "tl_signals": []
                }
            # elif lane.type == 'sidewalk':
            #     ret[id] = {
            #         SD.TYPE: MetaDriveType.BOUNDARY_SIDEWALK,
            #         SD.POLYGON: boundary_polygon,
            #     }
            elif lane.type == 'shoulder':
                ret[id] = {
                    SD.TYPE: MetaDriveType.BOUNDARY_SIDEWALK,
                    SD.POLYGON: boundary_polygon,
                }
            elif lane.type == 'crossing':
                # print('hello')
                ret[id] = {
                    SD.TYPE: MetaDriveType.CROSSWALK,
                    SD.POLYGON: boundary_polygon,
                }


    for road_id, road in graph.roads.items():
        for lane in road.lanes:
            if lane.type != 'driving':
                continue
            lane_id = f"lane_{lane.name}"
            if lane_id not in ret:
                continue

            left_lanes = []
            right_lanes = []
            if lane.left_neigh and lane.left_neigh.type == 'driving':
                left_lanes.append(f"lane_{lane.left_neigh.name}")
            if lane.right_neigh and lane.right_neigh.type == 'driving':
                right_lanes.append(f"lane_{lane.right_neigh.name}")

            ret[lane_id]["left_lanes"] = left_lanes
            ret[lane_id]["right_lanes"] = right_lanes

    for lane_name, lane_node in graph.lanes.items():
        lane_id = "lane_{}".format(lane_name)
        if lane_id not in ret:
            continue

        exit_ids = []
        for out_lane in lane_node.outgoing:
            out_id = "lane_{}".format(out_lane.name)
            if out_id in ret:
                exit_ids.append(out_id)
        ret[lane_id]["exit_lanes"] = exit_ids

        entry_ids = []
        for in_lane in lane_node.incoming:
            in_id = "lane_{}".format(in_lane.name)
            if in_id in ret:
                entry_ids.append(in_id)
        ret[lane_id]["entry_lanes"] = entry_ids

    for lane_name, lane_node in graph.lanes.items():
        base_lane_id = f"lane_{lane_name}"
        if base_lane_id not in ret or lane_node.type != 'driving':
            continue
        # Skip internal/junction lanes
        if ":" in lane_name:
            continue

        # GNELane.cpp: edge->myConnections, connection.toEdge
        # conn.getToLane() gives the destination lane
        turns = []
        seen_turns = set()
        _dir_map = {'s': 's', 'l': 'l', 'r': 'r', 't': 't',
                    'L': 'L', 'R': 'R', 'T': 't'}
        for conn in lane_node.sumolib_obj.getOutgoing():
            raw_dir = conn.getDirection()
            direction = _dir_map.get(raw_dir)
            if direction is None:
                continue
            dest_lane = conn.getToLane()
            to_lane_id = f"lane_{dest_lane.getID()}"
            key = (direction, to_lane_id)
            if to_lane_id in ret and key not in seen_turns:
                seen_turns.add(key)
                via_id = conn.getViaLaneID()
                turn_entry = {
                    "direction": direction,
                    "to_lane": to_lane_id,
                }
                if via_id:
                    turn_entry["via_lane"] = f"lane_{via_id}"
                turns.append(turn_entry)

        if turns:
            ret[base_lane_id]["turns"] = turns

        if lane_name in graph.lane_to_tl_signals:
            tl_signals = []
            for signal_info in graph.lane_to_tl_signals[lane_name]:
                to_lane_id = f"lane_{signal_info['to_lane']}"
                if to_lane_id in ret:  # только если целевая полоса существуе
                    tl_signals.append({
                        "phases": signal_info['phases'],
                        "to_lane": to_lane_id,
                        "tl_id": signal_info['tl_id']  # опционально
                    })
            if tl_signals:
                ret[base_lane_id]["tl_signals"] = tl_signals

    for lane_divider_id, lane_divider in enumerate(graph.lane_dividers):
        id = "lane_divider_{}".format(lane_divider_id)
        ret[id] = {SD.TYPE: MetaDriveType.LINE_BROKEN_SINGLE_WHITE, SD.POLYLINE: lane_divider}

    for edge_divider_id, edge_divider in enumerate(graph.edge_dividers):
        id = "edge_divider_{}".format(edge_divider_id)
        ret[id] = {SD.TYPE: MetaDriveType.LINE_SOLID_SINGLE_YELLOW, SD.POLYLINE: edge_divider}

    # for boundary_id, boundary in enumerate(getattr(graph, "road_boundaries", [])):
    #     id = "road_boundary_{}".format(boundary_id)
    #     ret[id] = {SD.TYPE: MetaDriveType.BOUNDARY_LINE, SD.POLYLINE: boundary}

    return ret
