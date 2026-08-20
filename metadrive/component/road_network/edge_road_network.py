import logging
from collections import namedtuple
from typing import List

from metadrive.component.road_network.base_road_network import BaseRoadNetwork
from metadrive.component.road_network.base_road_network import LaneIndex
from metadrive.scenario.scenario_description import ScenarioDescription as SD
from metadrive.utils.math import get_boxes_bounding_box
from metadrive.utils.pg.utils import get_lanes_bounding_box

from collections import deque 

lane_info = namedtuple("edge_lane", ["lane", "entry_lanes", "exit_lanes", "left_lanes", "right_lanes", "turns", "speed", "width", "tl_signals"])


class EdgeRoadNetwork(BaseRoadNetwork):
    """
    Compared to NodeRoadNetwork representing the relation of lanes in a node-based graph, EdgeRoadNetwork stores the
    relationship in edge-based graph, which is more common in real map representation
    """
    def __init__(self):
        super(EdgeRoadNetwork, self).__init__()
        self.graph = {}

    def add_lane(self, lane) -> None:
        assert lane.index is not None, "Lane index can not be None"
        self.graph[lane.index] = lane_info(
            lane=lane,
            entry_lanes=lane.entry_lanes or [],
            exit_lanes=lane.exit_lanes or [],
            left_lanes=lane.left_lanes or [],
            right_lanes=lane.right_lanes or [],
            turns=lane.turns or [],
            tl_signals=lane.tl_signals or [],
            speed=lane.speed or [],
            width=lane.width or []
        )
        
    def find_rightmost_lane_by_road_id(self, original_road_id):
        target = str(original_road_id)
        candidates = []

        for lane_key in self.graph.keys():
            if not isinstance(lane_key, str) or not lane_key.startswith("lane_"):
                continue
            parts = lane_key.split("_")
            if len(parts) < 3:
                continue

            try:
                edge_id = parts[1]
                lane_index = int(parts[2])
            except (ValueError, IndexError):
                continue

            if edge_id == target:
                candidates.append((lane_key, lane_index))

        if not candidates:
            # A random lane from anywhere on the map used to be returned here.
            # Nothing downstream could tell that apart from a real answer, so a
            # road id that does not exist placed the object -- or spawned the
            # ego, via vehicle_config["spawn_lane_index"] -- kilometres away in
            # silence. It also hid the callers' own fallbacks, which pick the
            # rightmost lane from the scene's own lane list: those never ran
            # because a random lane is neither None nor an exception.
            logging.warning(
                "find_rightmost_lane_by_road_id: no lane belongs to road %r "
                "(%d lanes in the network). Returning None.",
                target, len(self.graph)
            )
            return None

        rightmost = min(candidates, key=lambda x: x[1])
        return rightmost[0]

    def get_lane(self, index: LaneIndex):
        return self.graph[index].lane

    def __isub__(self, other):
        for id, lane_info in other.graph.items():
            self.graph.pop(id)
        return self

    def add(self, other, no_intersect=True):
        for id, lane_info in other.graph.items():
            if no_intersect:
                assert id not in self.graph.keys(), "Intersect: {} exists in two network".format(id)
            self.graph[id] = other.graph[id]
        return self

    def _get_bounding_box(self):
        """
       By using this bounding box, the edge length of x, y direction and the center of this road network can be
       easily calculated.
       :return: minimum x value, maximum x value, minimum y value, maximum y value
       """
        lanes = []
        for id, lane_info, in self.graph.items():
            lanes.append(lane_info.lane)
        res_x_max, res_x_min, res_y_max, res_y_min = get_boxes_bounding_box([get_lanes_bounding_box(lanes)])
        return res_x_min, res_x_max, res_y_min, res_y_max

    def shortest_path(self, start: str, goal: str):
        a = self.find_path(start, goal, max_len=10)
        return a

    def has_connection(self, lane_index_1, lane_index_2):
        if lane_index_1 not in self.graph or lane_index_2 not in self.graph:
            return False
        
        if lane_index_1 in self.graph and lane_index_2 in self.graph:
            return True
        
        lane_data = self.graph[lane_index_1]
        if lane_index_2 in lane_data.exit_lanes:
            return True

        for turn in lane_data.turns or []:
            to_lane = turn.get("to_lane")
            via_lane = turn.get("via_lane")
            if to_lane == lane_index_2 or via_lane == lane_index_2:
                return True
            if via_lane in self.graph and lane_index_2 in self.graph[via_lane].exit_lanes:
                return True
        return False

    def find_path(self, start: str, goal: str, max_len: int = 10) -> List[str]:
        """
        BFS route search with bounded path length.

        If goal is None, returns first valid route with length >= max_len,
        otherwise returns shortest path to goal (bounded by max_len).
        """
        if start not in self.graph:
            return []

        start_entry_lanes = [
            lane_id
            for lane_id in self.graph[start].entry_lanes
            if lane_id in self.graph and ":" not in lane_id
        ]
        fallback_path = [start]

        if len(start_entry_lanes) > 0:
            forced_entry_lane = start_entry_lanes[0]
            queue = deque([(start, [forced_entry_lane, start])])
            fallback_path = [forced_entry_lane, start]
        else:
            lanes = [start]
            lanes.extend(self.graph[start].right_lanes)
            seed_paths = [(lane, [lane]) for lane in lanes if lane in self.graph]
            queue = deque(seed_paths)
            if seed_paths:
                fallback_path = seed_paths[0][1]

        while queue:
            lane, path = queue.popleft()
            if lane not in self.graph:
                continue

            lane_data = self.graph[lane]
            blocked_uturn_targets = set()
            for turn in getattr(lane_data, "turns", []) or []:
                if turn.get("direction") == "t":
                    to_lane = turn.get("to_lane")
                    via_lane = turn.get("via_lane")
                    if to_lane is not None:
                        blocked_uturn_targets.add(to_lane)
                    if via_lane is not None:
                        blocked_uturn_targets.add(via_lane)

            if len(lane_data.exit_lanes) == 0:
                continue

            for _next in sorted(set(lane_data.exit_lanes)):
                is_uturn_target = _next in blocked_uturn_targets
                if is_uturn_target:
                    target_lane_data = self.graph.get(_next)
                    allow_uturn = target_lane_data is not None and len(target_lane_data.right_lanes) > 0
                    if not allow_uturn:
                        continue

                if len(path) >= 2 and _next == path[-2] and not is_uturn_target:
                    continue
                if _next in path:
                    continue
                if _next not in self.graph:
                    continue

                new_path = path + [_next]

                if goal is None:
                    if len(new_path) >= max_len:
                        return new_path
                else:
                    if _next == goal:
                        return new_path

                if len(new_path) < max_len:
                    queue.append((_next, new_path))
                    if len(new_path) > len(fallback_path):
                        fallback_path = new_path

        return fallback_path if goal is None else []

    def bfs_paths(self, start: str, goal: str) -> List[List[str]]:
        """Backward-compatible wrapper: yields at most one path."""
        path = self.find_path(start, goal, max_len=10)
        if path:
            yield path

    def get_peer_lanes_from_index(self, lane_index):
        info: lane_info = self.graph[lane_index]
        ret = [self.graph[lane_index].lane]
        for left_n in info.left_lanes:
            ret.append(self.graph[left_n].lane)
        for right_n in info.right_lanes:
            ret.append(self.graph[right_n].lane)
        return ret

    def destroy(self):
        """
        Destroy all lanes in this road network
        Returns: None

        """
        super(EdgeRoadNetwork, self).destroy()
        if self.graph is not None:
            for k, v in self.graph.items():
                v.lane.destroy()
                self.graph[k]: lane_info = None
            self.graph = None

    def __del__(self):
        logging.debug("{} is released".format(self.__class__.__name__))

    def get_map_features(self, interval=2):

        ret = {}
        for id, lane_info in self.graph.items():
            assert id == lane_info.lane.index
            ret[id] = {
                SD.POLYLINE: lane_info.lane.get_polyline(interval),
                SD.POLYGON: lane_info.lane.polygon,
                SD.TYPE: lane_info.lane.metadrive_type,
                SD.ENTRY: lane_info.entry_lanes,
                SD.EXIT: lane_info.exit_lanes,
                SD.WIDTH: lane_info.width,
                SD.LEFT_NEIGHBORS: lane_info.left_lanes,
                SD.RIGHT_NEIGHBORS: lane_info.right_lanes,
                SD.TURNS: lane_info.turns,
                SD.TL_SIGNALS: lane_info.tl_signals,
                "speed_limit_kmh": lane_info.lane.speed_limit
            }
        return ret

    def get_all_lanes(self):
        """
        This function will return all lanes in the road network
        :return: list of lanes
        """
        ret = []
        for id, lane_info in self.graph.items():
            ret.append(lane_info.lane)
        return ret


class OpenDriveRoadNetwork(EdgeRoadNetwork):
    def add_lane(self, lane) -> None:
        assert lane.index is not None, "Lane index can not be None"
        self.graph[lane.index] = lane_info(
            lane=lane, entry_lanes=None, exit_lanes=None, left_lanes=None, right_lanes=None
        )
