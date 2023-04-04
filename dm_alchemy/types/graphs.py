# Lint as: python3
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Types and functions for graphs in alchemy."""

import collections
import copy
import itertools
from typing import Any, Dict, List, Mapping, MutableSequence, Optional, Sequence, Set, Tuple, Union

import dataclasses
from dm_alchemy.types import helpers
from dm_alchemy.types import stones_and_potions
import numpy as np


NO_EDGE = 0
KNOWN_EDGE = 1
_POSS_EDGE_VALS = [helpers.UNKNOWN, NO_EDGE, KNOWN_EDGE]
_EDGE_VAL_INDICES = {edge_val: ind for ind, edge_val in
                     enumerate(_POSS_EDGE_VALS)}

# TODO(b/173785778): This information is contained in stones_and_potions. Remove
#  this copy.
_TOTAL_VERTICES = 8


def num_edges_in_cube() -> int:
  num_axes = stones_and_potions.get_num_axes()
  assert num_axes >= 1
  # pytype cannot tell that num edges is an int because this depends on num axes
  # being >= 1 so we tell it that it is an int.
  num_edges = num_axes * 2 ** (num_axes - 1)  # type: int
  return num_edges


def edge_val_to_index(edge_val: int) -> int:
  return _EDGE_VAL_INDICES[edge_val]


def index_to_edge_val(ind: int) -> int:
  return _POSS_EDGE_VALS[ind]

# Each constraint is a list with 3 entries, 1 for each axis in latent space.
# For axis i the entry is a list of 3 strings which can be either 1, -1, * or X.
# The value in string j shows the value that a stone must have for its jth
# latent space dimension to move in axis i (a * means any value).
# When i == j the string is always an X (this just helps with readability so you
# can see which axis is being moved along).
# For example the constraint [['X', '*', '1'], ['*', 'X', '*'], ['*', '*', 'X']]
# means that a stone can move on the y axis regardless of the value of x and z,
# and similarly can move on the z axis regardless of the value of x and y, but
# can only move on the x axis if z = 1, but y can be any value.
Constraint = List[List[str]]


# go from topology to nodes
class Node:
  """A node within a graph, for calculating shortest paths."""

  def __init__(self, idx: int, coords: Sequence[int]):
    self.idx = idx
    self.coords = coords

  def __str__(self) -> str:
    return 'Node ' + str(self.idx) + '(' + str(self.coords) + ')'

  def __hash__(self) -> int:
    return hash((self.idx, tuple(self.coords)))

  def __lt__(self, other) -> bool:
    return (self.idx, tuple(self.coords)) < (other.idx, tuple(other.coords))


class NodeList:
  """A list of nodes."""

  def __init__(self, nodes: MutableSequence[Node] = None):
    if nodes is None:
      self.nodes = []
    else:
      self.nodes = nodes

  def add_node(self, node: Node) -> None:
    self.nodes.append(node)

  def remove(self, node: Node) -> None:
    self.nodes.remove(node)

  def _get_node_by(self, feature: str, value: Any) -> Optional[Node]:
    matching_nodes = [n for n in self.nodes if getattr(n, feature) == value]
    if matching_nodes:
      assert len(matching_nodes) == 1, 'There should only be 1 node with '
      return matching_nodes[0]
    return None

  def get_node_by_idx(self, idx: int) -> Optional[Node]:
    return self._get_node_by('idx', idx)

  def get_node_by_coords(self, coords: Sequence[int]) -> Optional[Node]:
    return self._get_node_by('coords', coords)

  def filter_nodes(self, dimension: int, value: int) -> 'NodeList':
    # Return a list of nodes filtered by coordinates on a specific dimension
    return NodeList([n for n in self.nodes if n.coords[dimension] == value])

  def get_highest_value_node(
      self, reward_weights: stones_and_potions.RewardWeights
  ) -> Node:
    reward_vals = self.get_node_values(reward_weights)
    return self.nodes[np.argmax(reward_vals)]

  def get_node_values(
      self, reward_weights: stones_and_potions.RewardWeights
  ) -> List[int]:
    return [reward_weights(n.coords) for n in self.nodes]

  def get_nodes_by_value(
      self, reward_weights: stones_and_potions.RewardWeights, value: int
  ) -> List[Node]:
    reward_vals = self.get_node_values(reward_weights)
    return [n for (v, n) in zip(reward_vals, self.nodes) if v == value]

  def __hash__(self) -> int:
    return hash(tuple(self.nodes))


Edge = Tuple[int, Optional[stones_and_potions.Potion]]
ConnectedComponent = Set[Node]


class EdgeList:
  """An edge list that contains connections between nodes."""

  def __init__(self):
    self.edges: Dict[Node, Dict[Node, Edge]] = dict()
    # only gets weakly connected components
    self._connected_components: List[ConnectedComponent] = []

  def __str__(self) -> str:
    to_print: List[str] = []
    for k, v in self.edges.items():
      to_print.append('From')
      to_print.append(str(k))
      for k2, v2 in v.items():
        to_print.extend(['to', str(k2), 'weight', str(v2[0]), 'using potion',
                         str(v2[1]), '\n'])
    return ' '.join(to_print)

  def add_edge(
      self, startnode: Node, endnode: Node,
      potion: Optional[stones_and_potions.Potion] = None,
      weight: int = 1) -> None:
    """Adds an edge defined by 2 nodes."""
    if startnode not in self.edges:
      self.edges[startnode] = dict()
    if endnode in self.edges[startnode]:
      prev_edge = self.edges[startnode][endnode]
      self.edges[startnode][endnode] = prev_edge[0] + weight, prev_edge[1]
    else:
      self.edges[startnode][endnode] = weight, potion
    # check if either startnode or endnode are in a component, otherwise
    # create a new one
    if not self._connected_components:
      # no components at all, so start the first one
      self._connected_components.append({startnode, endnode})
    else:
      startnode_in = [startnode in c for c in self._connected_components]
      endnode_in = [endnode in c for c in self._connected_components]
      either_in = [s or e for s, e in zip(startnode_in, endnode_in)]
      if not any(either_in):
        # node not in either component, so start a new one
        self._connected_components.append({startnode, endnode})
      elif not any(startnode_in):
        # add startnode to component endnode is in
        target_component = [i for i, x in enumerate(endnode_in) if x]
        assert len(target_component) == 1
        to_component = target_component[0]
        self._connected_components[to_component].add(startnode)
      elif not any(endnode_in):
        # add startnode to component startnode is in
        target_component = [i for i, x in enumerate(startnode_in) if x]
        assert len(target_component) == 1
        to_component = target_component[0]
        self._connected_components[to_component].add(endnode)
      else:
        # if both in different components, combine components
        target_component1 = [i for i, x in enumerate(endnode_in) if x]
        assert len(target_component1) == 1
        target_component2 = [i for i, x in enumerate(startnode_in) if x]
        assert len(target_component2) == 1
        to_component1 = target_component1[0]
        to_component2 = target_component2[0]
        if to_component1 != to_component2:
          self._connected_components[to_component1].update(
              self._connected_components[to_component2])
          del self._connected_components[to_component2]

  def get_target_nodes(self, startnode: Node) -> Dict[Node, Edge]:
    # match on ID rather than instance
    matching_node = [n for n in self.edges if n.idx == startnode.idx][0]
    return self.edges[matching_node]

  def get_edge(self, startnode: Node, endnode: Node) -> Optional[Edge]:
    """If the edge exists returns it otherwise returns None."""
    matching_starts = [n for n in self.edges if n.idx == startnode.idx]
    if not matching_starts:
      return None
    matching_start = matching_starts[0]
    start = self.edges[matching_start]
    matching_ends = [n for n in start if n.idx == endnode.idx]
    if not matching_ends:
      return None
    matching_end = matching_ends[0]
    return self.edges[matching_start][matching_end]

  def has_edge(self, startnode: Node, endnode: Node) -> bool:
    return self.get_edge(startnode, endnode) is not None

  def get_connected_components(self) -> List[ConnectedComponent]:
    return self._connected_components

  def __hash__(self) -> int:
    return hash(tuple(sorted([(k, tuple(sorted([(k2, tuple(v2))
                                                for k2, v2 in v.items()])))
                              for k, v in self.edges.items()])))


@dataclasses.dataclass
class Graph:
  """An alchemy graph."""

  node_list: NodeList
  edge_list: EdgeList

  def __hash__(self) -> int:
    return hash((self.node_list, self.edge_list))

  def __eq__(self, other):
    return constraint_from_graph(self) == constraint_from_graph(other)


def all_nodes_in_graph() -> NodeList:
  """Gets a node list with all the nodes that can appear in an alchemy graph."""
  node_list = NodeList()
  for i in range(_TOTAL_VERTICES):
    coord = list(reversed([2 * int(x) - 1 for x in format(i, '03b')]))
    node_list.add_node(Node(i, coord))
  return node_list


def create_graph_from_constraint(
    constraint: Constraint,
    potions: Optional[Sequence[stones_and_potions.Potion]] = None
) -> Graph:
  """Creates the graph from the constraint string and the existing potions.

  Args:
    constraint: list of list of strings describing how stones can be transformed
      by potions (further explanation above where Constraint type is defined).
    potions: list of Potion objects; if None, assume all possible potions are
      available.

  Returns:
    A Graph object
  """
  node_list = all_nodes_in_graph()
  edge_list = EdgeList()

  def add_edge(potion: stones_and_potions.Potion):
    """Adds an edge defined by a potion."""

    dimension = potion.dimension
    direction = potion.direction
    startnodes = node_list.filter_nodes(dimension, -direction)
    dimension_constraints = [(i, x)
                             for i, x in enumerate(constraint[dimension])
                             if x == '1' or x == '-1']
    for dimension_constraint in dimension_constraints:
      startnodes = startnodes.filter_nodes(
          dimension_constraint[0], int(dimension_constraint[1]))
    for node in startnodes.nodes:
      endnode_coord = copy.copy(node.coords)  # type: MutableSequence[int]
      endnode_coord[dimension] = direction
      endnode = node_list.get_node_by_coords(endnode_coord)
      edge_list.add_edge(node, endnode, potion)

  if potions:
    for potion in potions:
      add_edge(potion)
  else:
    count = 0
    for latent_potion in stones_and_potions.possible_latent_potions():
      add_edge(stones_and_potions.Potion(
          count, latent_potion.latent_dim, latent_potion.latent_dir))
      count += 1

  return Graph(node_list, edge_list)


def constraint_from_graph(graph: Graph) -> Constraint:
  """Creates the constraint string corresponding to the graph passed in."""
  # For each axis consider the possible values on the other axes and check which
  # of the edges exist.
  constraint = []
  for axis in range(3):
    other_can_be = [set(), set()]
    for other_vals in itertools.product([-1, 1], [-1, 1]):
      coords_start = list(other_vals[:axis]) + [-1] + list(other_vals[axis:])
      coords_end = list(other_vals[:axis]) + [1] + list(other_vals[axis:])
      node_start = graph.node_list.get_node_by_coords(coords_start)
      node_end = graph.node_list.get_node_by_coords(coords_end)
      if graph.edge_list.has_edge(node_start, node_end):
        other_can_be[0].add(other_vals[0])
        other_can_be[1].add(other_vals[1])
    axis_constraint = ['*' if len(can_be) > 1 else str(list(can_be)[0])
                       for can_be in other_can_be]
    constraint.append(axis_constraint[:axis] + ['X'] + axis_constraint[axis:])
  return constraint


def convert_graph_to_adj_mat(graph: Graph) -> np.ndarray:
  """Converts from node_list, edge list to adjacency matrix.

  Args:
    graph: a graphs.Graph object.

  Returns:
    An adjacency matrix indicating which nodes are connected. The value at
    row i and column j indicates that node i is connected to node j using a
    potion with index equal to the value - 1.
  """
  node_coords = [s.coords for s in graph.node_list.nodes]
  num_nodes = len(node_coords)
  adj_mat = NO_EDGE * np.ones((num_nodes, num_nodes), int)
  for start_node, edge_list in graph.edge_list.edges.items():
    for end_node, potion_list in edge_list.items():
      from_ind = start_node.idx
      to_ind = end_node.idx
      adj_mat[from_ind, to_ind] = potion_list[1].as_index + 1
  return adj_mat


def convert_adj_mat_to_graph(
    adj_mat: np.ndarray,
    init_potions: Optional[Sequence[stones_and_potions.Potion]] = None
) -> Graph:
  """Converts from adjacency matrix and init_potions to graph.

  Args:
    adj_mat: a numpy matrix indicating which nodes are connected. The value at
      row i and column j indicates that node i is connected to node j using a
      potion with index equal to the value - 1, with NO_EDGE indicating no
      connection.
    init_potions: a list of Potion objects.

  Returns:
    A Graph object
  """
  node_list = all_nodes_in_graph()
  node_coords = [s.coords for s in node_list.nodes]
  edge_list = EdgeList()
  from_inds, to_inds = np.where(adj_mat != NO_EDGE)
  for row, column in zip(from_inds, to_inds):
    from_node = node_list.get_node_by_coords(node_coords[row])
    to_node = node_list.get_node_by_coords(node_coords[column])
    potion_color_idx = adj_mat[row, column] - 1
    potion_dir = (potion_color_idx % 2) * 2 - 1
    potion_dim = potion_color_idx // 2
    if init_potions is not None:
      matched_potions = [p for p in init_potions if p.dimension == potion_dim
                         and p.direction == potion_dir]
    else:
      matched_potions = [stones_and_potions.Potion(-1, potion_dim, potion_dir)]
    for potion in matched_potions:
      edge_list.add_edge(from_node, to_node, potion)
  return Graph(node_list, edge_list)


def possible_constraints() -> List[Constraint]:
  """Returns all lists of constraint strings regardless of graph validity.

  Returns:
    A list of constraints.
  """
  possible_values = ['-1', '1', '*']
  one_axis_constraints = []
  for v1 in possible_values:
    for v2 in possible_values:
      one_axis_constraints.append([v1, v2])

  def insert_x(val: List[str], pos: int) -> List[str]:
    return val[:pos] + ['X'] + val[pos:]

  constraints = []
  for x in one_axis_constraints:
    for y in one_axis_constraints:
      for z in one_axis_constraints:
        constraints.append([insert_x(x, 0), insert_x(y, 1), insert_x(z, 2)])

  all_graphs = [create_graph_from_constraint(constraint)
                for constraint in constraints]

  def all_reachable(g: Graph) -> bool:
    # All nodes must have an edge to them and there must be only one component.
    # It is necessary to check both since connected components only returns
    # components with at least one edge.
    return (len(g.edge_list.edges) == len(g.node_list.nodes) and
            len(g.edge_list.get_connected_components()) == 1)

  return [constr for graph, constr in zip(all_graphs, constraints)
          if all_reachable(graph)]


def no_bottleneck_constraints() -> List[Constraint]:
  return [[['X', '*', '*'], ['*', 'X', '*'], ['*', '*', 'X']]]


def bottleneck1_constraints() -> List[Constraint]:
  return [[['X', '-1', '1'], ['*', 'X', '*'], ['*', '*', 'X']]]


def bottleneck2_constraints() -> List[Constraint]:
  return [[['X', '-1', '*'], ['*', 'X', '*'], ['*', '*', 'X']]]


def bottleneck3_constraints() -> List[Constraint]:
  return [[['X', '-1', '1'], ['*', 'X', '*'], ['*', '-1', 'X']]]


def get_num_constraints(
    constraints: Sequence[Constraint]) -> List[int]:
  """Returns a list of the total number of constraints across axes."""

  def count_constraints(axis_constraints: List[str]) -> int:
    return sum(1 for e in axis_constraints if e not in ['*', 'X'])

  return [sum(count_constraints(e) for e in c) for c in constraints]


def latent_constraint_to_stone_space(
    latent_constraint: Constraint, stone_map: stones_and_potions.StoneMap
) -> Constraint:
  """Converts a constraint specified in latent space to one in stone space."""
  stone_space_constraint = copy.deepcopy(latent_constraint)
  axes = list(range(len(latent_constraint)))
  for axis in axes:
    for other_axis in axes[:axis] + axes[axis + 1:]:
      if stone_map.latent_pos_dir[other_axis] == -1:
        if latent_constraint[axis][other_axis] == '1':
          stone_space_constraint[axis][other_axis] = '-1'
        elif latent_constraint[axis][other_axis] == '-1':
          stone_space_constraint[axis][other_axis] = '1'
  return stone_space_constraint


def graph_distr(
    constraints: Sequence[Constraint]
) -> Dict[Graph, float]:
  """Returns prior over all valid graphs."""
  num_constraints = get_num_constraints(constraints)

  valid_graphs = [
      (create_graph_from_constraint(constraint), constraint_count)
      for constraint, constraint_count in zip(constraints, num_constraints)]

  # Equal probability is given to the set of graphs with each valid number of
  # constraints.
  # In practice this means:
  #   1/4 probability for the constraint 0 case
  #   (1/4)*(1/12) probability for each case with one constraint (12 cases)
  #   (1/4)*(1/48) probability for each case with two constraints (48 cases)
  #   (1/4)*(1/48) probability for each case with three constraints (48 cases)
  graphs_per_constraint = collections.Counter([g[1] for g in valid_graphs])
  prior = [(1.0 / (len(graphs_per_constraint) * graphs_per_constraint[g[1]]))
           for g in valid_graphs]
  valid_graphs = [g[0] for g in valid_graphs]
  return {g: p for g, p in zip(valid_graphs, prior)}


def random_graph(
    distr: Mapping[Graph, float], random_state: np.random.RandomState
) -> Graph:
  graphs = list(distr.keys())
  return graphs[random_state.choice(
      len(graphs), p=np.array(list(distr.values()), dtype=np.float32))]


def cube_edges() -> List[Tuple[int, int]]:
  """Gets pairs of graph indices which are edges of the cube."""
  graph_nodes = all_nodes_in_graph().nodes
  edges = []
  for (i, node_i), (j, node_j) in itertools.combinations(
      enumerate(graph_nodes), 2):
    # Nodes are 1D so the return has 1 element with the non-zero indices.
    diffs = np.nonzero(np.array(node_i.coords) - np.array(node_j.coords))[0]
    # If the nodes differ on just one dimension there may be an edge between
    # them.
    if diffs.size == 1:
      edges.append((i, j))
  return edges


def blank_known_adj_mat() -> np.ndarray:
  """Returns an adjacency matrix with no knowledge of what cube edges exist."""
  # At first we know the structure of the cube so only 3 possible edges from
  # each node.
  # Start with no edges and add possible edge along the edges of a cube.
  num_nodes = len(all_nodes_in_graph().nodes)
  known_adj_mat = NO_EDGE * np.ones((num_nodes, num_nodes), dtype=object)
  for i, j in cube_edges():
    known_adj_mat[i, j] = helpers.UNKNOWN
    known_adj_mat[j, i] = helpers.UNKNOWN
  return known_adj_mat


def known_adj_mat_from_edge_values(edge_vals: Sequence[int]) -> np.ndarray:
  known_adj_mat = blank_known_adj_mat()
  for (start_node, end_node), val in zip(cube_edges(), edge_vals):
    if val not in _POSS_EDGE_VALS:
      raise ValueError('Invalid edge value for known adjacency matrix.')
    known_adj_mat[start_node, end_node] = val
    known_adj_mat[end_node, start_node] = val
  return known_adj_mat


def adj_mat_from_edge_values(edge_vals: Sequence[int]) -> np.ndarray:
  """Gets adjacency matrix with potion indices at entries from key edge values."""
  known_adj_mat = blank_known_adj_mat()
  for (start_node, end_node), val in zip(cube_edges(), edge_vals):
    if not 0 <= val <= stones_and_potions.PerceivedPotion.num_types:
      raise ValueError('Invalid edge value for adjacency matrix.')
    known_adj_mat[start_node, end_node] = val
    if val > 0:
      potion_pair = stones_and_potions.perceived_potion_from_index(
          stones_and_potions.PerceivedPotionIndex(val - 1))
      potion_pair.perceived_dir = -potion_pair.perceived_dir
      known_adj_mat[end_node, start_node] = potion_pair.index() + 1
    else:
      known_adj_mat[end_node, start_node] = 0
  return known_adj_mat


def edge_values_from_adj_mat(adj_mat: np.ndarray) -> List[int]:
  return [adj_mat[start_node, end_node]
          for start_node, end_node in cube_edges()]


class PartialGraph:
  """Partial information about a graph."""

  def __init__(self):
    self.known_adj_mat = blank_known_adj_mat()

  def add_edge(
      self, stone_start: stones_and_potions.LatentStone,
      stone_end: stones_and_potions.LatentStone, val: int
  ) -> None:
    """Adds an edge to the partial graph."""
    # When we discover that an edge either exists or does not exist.
    # Add the edge from this stone in the direction according to potion
    from_ind = stone_start.index()
    to_ind = stone_end.index()
    self.known_adj_mat[from_ind, to_ind] = val
    # Reverse edge must have the same val
    self.known_adj_mat[to_ind, from_ind] = val
    # There is no need to deduce other knowledge from this since it is all
    # captured by the list of valid graphs (e.g. the exclusion of xor)

  def update(self, all_graphs: Sequence[Graph]) -> None:
    """Updates the graph by filling in edges we can deduce."""
    # Get all graphs matching the partial information we have.
    matches = self.matching_graphs(all_graphs)
    edge_vals = []
    adj_mats = [partial_graph_from_graph(match).known_adj_mat
                for match in matches]
    for start_node, end_node in cube_edges():
      # For this edge get all possible values in matching graphs.
      vals = set(adj_mat[start_node, end_node] for adj_mat in adj_mats)
      # If there is only 1 possible value then it set it in the adjacency
      # matrix.
      if len(vals) == 1:
        edge_vals.append(vals.pop())
      else:
        # Otherwise, set unknown.
        edge_vals.append(helpers.UNKNOWN)
    self.known_adj_mat = known_adj_mat_from_edge_values(edge_vals)

  def matching_graphs(
      self, all_graphs: Sequence[Graph], return_indices: bool = False
  ) -> Union[List[Graph], List[int]]:
    """Returns a list of all graphs which match the partial info we have."""
    def matches(graph: Graph):
      """Returns whether the passed graph matches the partial info we have."""
      # g matches our knowledge if all edges in g are plausible according to
      # what we know and if all edges we know are in g
      known_edges = {(v1, v2) for v1, v2 in
                     zip(*np.where(self.known_adj_mat == KNOWN_EDGE))}
      for start_node, end_node_list in graph.edge_list.edges.items():
        for end_node in end_node_list:
          from_ind = start_node.idx
          to_ind = end_node.idx
          if (from_ind, to_ind) in known_edges:
            known_edges.remove((from_ind, to_ind))
            continue
          if self.known_adj_mat[from_ind, to_ind] == NO_EDGE:
            return False
      return not known_edges

    if return_indices:
      return [i for i, g in enumerate(all_graphs) if matches(g)]
    return [g for g in all_graphs if matches(g)]

  def index(self) -> int:
    important_edges = edge_values_from_adj_mat(self.known_adj_mat)
    return int(np.ravel_multi_index(
        tuple(edge_val_to_index(val) for val in important_edges),
        tuple(len(_POSS_EDGE_VALS) for _ in important_edges)))


def partial_graph_from_graph(graph: Graph) -> PartialGraph:
  """Converts a graphs.Graph to a PartialGraph with all information known."""
  known_adj_mat = NO_EDGE * np.ones(
      (len(graph.node_list.nodes), len(graph.node_list.nodes)))
  for start_node, edges in graph.edge_list.edges.items():
    for end_node in edges:
      from_ind = start_node.idx
      to_ind = end_node.idx
      known_adj_mat[from_ind, to_ind] = KNOWN_EDGE
  partial_graph = PartialGraph()
  partial_graph.known_adj_mat = known_adj_mat
  return partial_graph


def partial_graph_from_index(ind: int) -> PartialGraph:
  """Converts from the integer representation to the full type."""
  important_edges = np.unravel_index(
      ind, tuple(len(_POSS_EDGE_VALS) for _ in range(num_edges_in_cube())))
  partial_graph = PartialGraph()
  partial_graph.known_adj_mat = known_adj_mat_from_edge_values(
      [index_to_edge_val(int(e)) for e in important_edges])
  return partial_graph


def partial_graph_from_possibles(possibles: np.ndarray) -> PartialGraph:
  """Constructs a partial graph from the list of possible graphs."""
  partial_graph = PartialGraph()
  if possibles.size == 0:
    return partial_graph
  graph_nodes = possibles[0].node_list.nodes
  # Go through each possible graph and check the status of the important edges
  adj_mats = tuple(partial_graph_from_graph(p).known_adj_mat for p in possibles)
  stacked_adj_mats = np.stack(adj_mats)
  uniques = [[np.unique(stacked_adj_mats[:, i, j]) for i in range(len(
      graph_nodes))] for j in range(len(graph_nodes))]

  for i in range(len(graph_nodes)):
    for j in range(len(graph_nodes)):
      u = uniques[i][j]
      if len(u) == 1:
        partial_graph.known_adj_mat[i, j] = u[0]
      else:
        partial_graph.known_adj_mat[i, j] = helpers.UNKNOWN

  return partial_graph


def graph_with_potions(
    graph: Graph, potions: Sequence[stones_and_potions.Potion]
) -> Graph:
  """Makes a copy of a graph with edges only where we have potions."""
  reduced_edge_list = EdgeList()
  for start_node, edges in graph.edge_list.edges.items():
    for end_node, edge in edges.items():
      potions_required = [
          (i, ec - sc) for i, (sc, ec) in enumerate(zip(
              start_node.coords, end_node.coords)) if sc != ec]
      assert len(potions_required) == 1
      if potions_required[0] in [
          (p.dimension, 2 * p.direction) for p in potions]:
        reduced_edge_list.add_edge(start_node, end_node, edge[1], edge[0])

  return Graph(copy.deepcopy(graph.node_list), reduced_edge_list)
