# Lint as python3
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
"""A tree search ideal observer for alchemy."""

import collections
import copy
import math
from typing import Any, Counter, List, Mapping, MutableMapping, Sequence, Tuple

from dm_alchemy.ideal_observer import helpers
from dm_alchemy.ideal_observer import precomputed_maps
from dm_alchemy.types import graphs
from dm_alchemy.types import helpers as types_helpers
from dm_alchemy.types import stones_and_potions
import numpy as np

# Alias these for readability
AlignedStone = stones_and_potions.AlignedStone
PerceivedStone = stones_and_potions.PerceivedStone
PerceivedPotion = stones_and_potions.PerceivedPotion
LatentStone = stones_and_potions.LatentStone
LatentPotion = stones_and_potions.LatentPotion
StoneMap = stones_and_potions.StoneMap
PotionMap = stones_and_potions.PotionMap
PartialStoneMap = stones_and_potions.PartialStoneMap
PartialPotionMap = stones_and_potions.PartialPotionMap
PartialGraph = graphs.PartialGraph

PrecomputedMaps = precomputed_maps.PrecomputedMaps

# We use indices in place of actual types for speed
AlignedStoneIndex = stones_and_potions.AlignedStoneIndex
PerceivedPotionIndex = stones_and_potions.PerceivedPotionIndex
LatentStoneIndex = stones_and_potions.LatentStoneIndex
LatentPotionIndex = stones_and_potions.LatentPotionIndex
StoneMapIndex = stones_and_potions.StoneMapIndex
PotionMapIndex = stones_and_potions.PotionMapIndex

Action = Tuple[AlignedStoneIndex, PerceivedPotionIndex]
ActionObjective = Tuple[Action, Any]
SearchResults = MutableMapping[int, ActionObjective]
ActionObjectiveAndSearchResults = Tuple[Action, Any, SearchResults]

END_TRIAL = types_helpers.END_TRIAL


def get_possible_stone_maps(
    stone_map_indices: Sequence[StoneMapIndex],
    aligned_stones: Sequence[AlignedStone]
) -> Tuple[List[int], List[StoneMapIndex]]:
  """Gets possible stone maps (and their indices) consistent with stones passed."""
  # This doesn't need to be especially fast as it just happens once per trial.
  still_in_s = []
  poss_sms = []
  for i, sm in enumerate(stone_map_indices):
    stone_map = stones_and_potions.stone_map_from_index(sm)
    if stone_map.consistent_with_stones(aligned_stones):
      still_in_s.append(i)
      poss_sms.append(sm)
  return still_in_s, poss_sms


class WorldStateDistribution:
  """Distribution over world states."""

  def __init__(
      self, stone_map_distr: Mapping[StoneMapIndex, float],
      potion_map_distr: Mapping[PotionMapIndex, float],
      precomputed: PrecomputedMaps
  ):

    self.stone_map_possible = sorted(stone_map_distr.keys())
    self.potion_map_possible = sorted(potion_map_distr.keys())
    self.partial_potion_map_index = (
        stones_and_potions.partial_potion_map_from_possibles(
            self.potion_map_possible, precomputed.index_to_perm_index).index(
                precomputed.perm_index_to_index))
    self.partial_stone_map_index = (
        stones_and_potions.partial_stone_map_from_possibles(
            self.stone_map_possible).index())

    # Use a bitfield like structure otherwise it takes forever to copy
    self.partial_graph_possible = helpers.list_to_bitfield(
        range(precomputed.graphs_list.shape[0]))
    self.partial_graph_index = (
        precomputed.partial_graph_index_to_possible_index[
            graphs.partial_graph_from_possibles(
                precomputed.graphs_list[self.get_possible_graphs()]).index()])

    self.poss_world_states = np.zeros((
        len(potion_map_distr), len(stone_map_distr),
        precomputed.graph_index_distr.shape[0]), dtype=np.float)
    for potion_map_index, p1 in enumerate(potion_map_distr.values()):
      for stone_map_index, p2 in enumerate(stone_map_distr.values()):
        for graph_index, p3 in enumerate(precomputed.graph_index_distr):
          p = p1 * p2 * p3
          self.poss_world_states[
              potion_map_index, stone_map_index, graph_index] = p

    self.observed_no_effect_bits = 0

  def new_trial(
      self, aligned_stone_indices: Counter[AlignedStoneIndex]
  ) -> None:
    """Updates the world state distribution given the stones perceived.

    The reward indicator on the stones allows us to limit the possible maps from
    stone space to latent space. For example if we see a stone with reward 3
    then it must be at [1, 1, 1] in latent space.

    Args:
      aligned_stone_indices: The stones seen in this trial.
    """
    # The stones seen when we start a new trial could eliminate stone map
    # possibilities if there are multiple.
    aligned_stones = [
        stones_and_potions.aligned_stone_from_index(aligned_stone_index)
        for aligned_stone_index in aligned_stone_indices]
    still_in_s, poss_sms = get_possible_stone_maps(
        self.stone_map_possible, aligned_stones)
    assert still_in_s, 'Stones seen in trial are impossible.'

    self.stone_map_possible = poss_sms
    self.partial_stone_map_index = (
        stones_and_potions.partial_stone_map_from_possibles(
            self.stone_map_possible).index())
    self.poss_world_states = self.poss_world_states[:, still_in_s, :]
    total_prob = self.poss_world_states.sum()
    self.poss_world_states /= total_prob

  def get_possible_graphs(self) -> List[int]:
    return helpers.bitfield_to_list(self.partial_graph_possible)

  def potions_equivalent(
      self, p1: PerceivedPotionIndex, p2: PerceivedPotionIndex,
      s: AlignedStoneIndex, precomputed: PrecomputedMaps
  ) -> bool:
    """If the potions effect on the stone are equivalent in this belief state.

    This is the case if we have the same knowledge about the potions effect and
    the same number of these potions and their counterparts on the same
    dimension. In this case the calculation of the expected reward must be
    exactly the same. This does not imply that the actual effect of the potions
    will be the same.

    For example, if we have no knowledge of the perceptual mapping and graph and
    we have one red potion and one green potion then the calculation will
    include terms for the probability that the red potion acts on each of the
    directed edges in latent space. The calculation for the green potion would
    have all of the same terms.

    Args:
      p1: The first potion.
      p2: The second potion.
      s: The stone they will be applied to.
      precomputed: Precomputed maps used for speed.

    Returns:
      True if they are equivalent.
    """
    latent_dims = precomputed.possible_latent_dims[
        p1, self.partial_potion_map_index[0]]
    latent_dims2 = precomputed.possible_latent_dims[
        p2, self.partial_potion_map_index[0]]
    if latent_dims != latent_dims2:
      return False
    for latent_dim in latent_dims:
      this_could_stay_still1, latent_dirs1 = precomputed.possible_latent_dirs[
          self.partial_potion_map_index[1], self.partial_stone_map_index,
          latent_dim, p1, s]
      this_could_stay_still2, latent_dirs2 = precomputed.possible_latent_dirs[
          self.partial_potion_map_index[1], self.partial_stone_map_index,
          latent_dim, p2, s]
      if this_could_stay_still1 != this_could_stay_still2:
        return False
      if latent_dirs1 != latent_dirs2:
        return False
    return True

  def update_possible(
      self, stone_index: AlignedStoneIndex,
      potion_index: PerceivedPotionIndex,
      result_index: AlignedStoneIndex,
      precomputed: PrecomputedMaps
  ) -> None:
    """Updates which possibilities are consistent with this observation.

    Args:
      stone_index: The initial stone.
      potion_index: The potion applied to the stone.
      result_index: The resulting stone.
      precomputed: Precomputed maps used for speed.
    """
    # Update which potion maps are possible
    poss_p = precomputed.poss_p_maps[stone_index, potion_index, result_index]
    if poss_p is not None:
      self.partial_potion_map_index = precomputed.partial_potion_map_update[
          stone_index, potion_index, result_index,
          self.partial_potion_map_index[0], self.partial_potion_map_index[1]]
      self.potion_map_possible, still_in_p = helpers.sorted_intersection(
          self.potion_map_possible, poss_p)
      self.poss_world_states = self.poss_world_states[still_in_p, :, :]

    # Update which stone maps are possible
    poss_s = precomputed.poss_s_maps[stone_index, potion_index, result_index]
    if poss_s is not None:
      self.partial_stone_map_index = precomputed.partial_stone_map_update[
          stone_index, result_index, self.partial_stone_map_index]
      self.stone_map_possible, still_in_s = helpers.sorted_intersection(
          self.stone_map_possible, poss_s)
      self.poss_world_states = self.poss_world_states[:, still_in_s, :]

    # Update which graphs are possible
    update_graphs_possible = False
    if stone_index != result_index:
      update_graphs_possible = True
      self.partial_graph_index = precomputed.partial_graph_update[
          precomputed.drop_reward[stone_index],
          precomputed.drop_reward[result_index]][self.partial_graph_index]

    if not update_graphs_possible:
      missing_edge = precomputed.missing_edge_no_change[
          self.partial_stone_map_index, self.partial_potion_map_index[0],
          self.partial_potion_map_index[1], potion_index,
          precomputed.drop_reward[stone_index]]
      if missing_edge != -1:
        update_graphs_possible = True
        self.partial_graph_index = precomputed.update_partial_graph_no_change[
            self.partial_graph_index, missing_edge]

    if update_graphs_possible:
      new_graphs_possible = precomputed.partial_graph_to_matching_graphs[
          self.partial_graph_index]
      remaining_graphs_possible = (self.partial_graph_possible &
                                   new_graphs_possible)
      # Work out the position of the eliminated slices in poss_world_states.
      still_in_g = []
      ind = 0
      for i in range(precomputed.graphs_list.shape[0]):
        check = 1 << i
        poss_check = self.partial_graph_possible & check
        if remaining_graphs_possible & check and poss_check:
          still_in_g.append(ind)
        if poss_check:
          ind += 1
      self.partial_graph_possible = remaining_graphs_possible

      self.poss_world_states = self.poss_world_states[:, :, still_in_g]

    # If stone map is known then get info about which actions will have no
    # effect because they take the stone out of the latent cube
    stone_map_index = precomputed.partial_stone_map_to_stone_map[
        self.partial_stone_map_index]
    if stone_map_index != -1:
      self.observed_no_effect_bits |= precomputed.no_effect_from_partial_chem[
          stone_map_index, self.partial_potion_map_index[0],
          self.partial_potion_map_index[1]]

  def possible_outcomes(
      self, perceived_potion_index: PerceivedPotionIndex,
      aligned_stone_index: AlignedStoneIndex, precomputed: PrecomputedMaps
  ) -> Tuple[List[AlignedStoneIndex], bool]:
    """Gets a list of outcomes we could see applying this potion to this stone.

    Args:
      perceived_potion_index: The potion we apply.
      aligned_stone_index: The stone we apply it to.
      precomputed: Precomputed maps used for speed.

    Returns:
      A list of possible outcomes and a boolean saying whether one of them is
      the stone remaining the same.
    """
    outcomes = []
    could_stay_still = False
    for latent_dim in precomputed.possible_latent_dims[
        perceived_potion_index, self.partial_potion_map_index[0]]:
      # latent_dir may not be possible if the reward for the stone is already at
      # max or min. If you know stone position in latent space on latent_dim
      # then latent_dir can only be the opposite so you don't need to consider
      # the reward going the other way even if you don't know what direction the
      # potion acts in.
      this_could_stay_still, latent_dirs = precomputed.possible_latent_dirs[
          self.partial_potion_map_index[1], self.partial_stone_map_index,
          latent_dim, perceived_potion_index, aligned_stone_index]
      # Could stay still due to going outside the cube.
      could_stay_still |= this_could_stay_still
      for latent_dir in latent_dirs:
        result = precomputed.react_result[
            aligned_stone_index, latent_dim, (latent_dir + 1) // 2,
            self.partial_graph_index]
        if result != helpers.IMPOSSIBLE:
          outcomes.append(result)
      if latent_dirs:
        # Could stay still due to edge not existing
        this_edge_exists = precomputed.edge_exists[
            self.partial_graph_index, precomputed.drop_reward[
                aligned_stone_index], latent_dim]
        # If either we know the edge isn't there or we are not sure if the edge
        # is there then could stay still.
        if this_edge_exists != graphs.KNOWN_EDGE:
          could_stay_still = True

    if could_stay_still:
      outcomes.append(aligned_stone_index)
    return outcomes, could_stay_still

  def action_and_outcome(
      self, stone_index: AlignedStoneIndex,
      potion_index: PerceivedPotionIndex, result_index: AlignedStoneIndex,
      precomputed: PrecomputedMaps, bit_mask: int
  ) -> float:
    """Updates the world state distribution given we saw this observation."""

    # Eliminate whole slices of the world state distribution if possible given
    # the new information.
    self.update_possible(stone_index, potion_index, result_index, precomputed)

    # If the stone changed as a result of applying the potion then all
    # information gained removes whole slices otherwise we must remove
    # combinations of potion map, stone map and graph.
    if stone_index == result_index:
      for potion_i, potion_map_index in enumerate(self.potion_map_possible):
        for stone_i, stone_map_index in enumerate(self.stone_map_possible):
          graphs_bitfield = precomputed.graphs_with_edge[
              stone_map_index, potion_map_index, stone_index, potion_index]

          # If there are no graphs with an edge between the stone and result
          # then continue.
          if graphs_bitfield == 0:
            continue

          # Graphs in the list are not possible because they contain the edge so
          # the stone should have changed but didn't.
          not_possible = []
          ind = 0
          for i in range(precomputed.graphs_list.shape[0]):
            check = 1 << i
            poss_check = self.partial_graph_possible & check
            if graphs_bitfield & check and poss_check:
              not_possible.append(ind)
            if poss_check:
              ind += 1

          self.poss_world_states[potion_i, stone_i, not_possible] = 0.0
      self.observed_no_effect_bits |= bit_mask

    # Re-normalise
    total_prob = self.poss_world_states.sum()
    if total_prob > 0.0:
      self.poss_world_states /= total_prob
    return total_prob

  def __len__(self):
    return len(self.poss_world_states)

  def update_stone_map(self, new_to_old: StoneMap) -> None:
    """If we assumed the wrong rotation we may need to swap stone map dims."""
    # Change partial stone map.
    partial_stone_map = stones_and_potions.partial_stone_map_from_index(
        self.partial_stone_map_index)
    # If the partial stone map is not completely known at this point then it is
    # completely unknown since any 2 bits of information would be enough to
    # completely determine the rotation.
    if any(c == types_helpers.UNKNOWN
           for c in partial_stone_map.latent_pos_dir):
      assert all(c == types_helpers.UNKNOWN
                 for c in partial_stone_map.latent_pos_dir)
    else:
      partial_stone_map.chain(new_to_old)
    self.partial_stone_map_index = partial_stone_map.index()
    # Change poss stone maps.
    old_stone_map_possible = copy.deepcopy(self.stone_map_possible)
    old_stone_map_to_new_stone_map = {}
    for stone_map_index in self.stone_map_possible:
      stone_map = stones_and_potions.stone_map_from_index(stone_map_index)
      stone_map.chain(new_to_old)
      old_stone_map_to_new_stone_map[stone_map_index] = stone_map.index()
    self.stone_map_possible = sorted(
        old_stone_map_to_new_stone_map[stone_map_index]
        for stone_map_index in self.stone_map_possible)
    new_stone_map_to_index = {
        stone_map_index: i
        for i, stone_map_index in enumerate(self.stone_map_possible)}
    old_index_to_new_index = [
        new_stone_map_to_index[old_stone_map_to_new_stone_map[stone_map]]
        for stone_map in old_stone_map_possible]
    # Change poss world states.
    old_poss_world_states = copy.deepcopy(self.poss_world_states)
    for old_index, new_index in enumerate(old_index_to_new_index):
      self.poss_world_states[new_index] = old_poss_world_states[old_index]
    # Change observed no effect.
    old_observed_no_effect_bits = copy.deepcopy(self.observed_no_effect_bits)
    self.observed_no_effect_bits = 0
    for old_index in range(stones_and_potions.AlignedStone.num_dir_assignments):
      aligned_stone = stones_and_potions.aligned_stone_from_index(
          AlignedStoneIndex(old_index))
      new_index = new_to_old.apply(aligned_stone).index()
      for potion_index in range(stones_and_potions.PerceivedPotion.num_types):
        old_mask = 1 << (old_index * PerceivedPotion.num_types) + potion_index
        masked = old_observed_no_effect_bits & old_mask
        if masked:
          new_mask = 1 << (new_index * PerceivedPotion.num_types) + potion_index
          self.observed_no_effect_bits |= new_mask


def init_world_state_distribution(
    precomputed: PrecomputedMaps
) -> WorldStateDistribution:
  """Creates an initial world state distribution from observed stones."""
  # Initialise the ideal observer based on the stones and potions you can see
  return WorldStateDistribution(
      stones_and_potions.stone_map_distr(precomputed.stone_maps),
      stones_and_potions.potion_map_distr(precomputed.potion_maps),
      precomputed)


def stone_potion_bit_mask(
    stone_index: AlignedStoneIndex, potion_index: PerceivedPotionIndex,
    precomputed: PrecomputedMaps
) -> int:
  """Returns a mask for the bit representing a stone potion pair."""
  stone_part = precomputed.drop_reward[stone_index] * PerceivedPotion.num_types
  return 1 << (stone_part + potion_index)


class BeliefState:
  """Belief the ideal observer has about stones, potions, world and reward.

  The belief state consists of a set of perceived stones, a set of perceived
  potions, a distribution over world states, and a reward so far.
  """

  possible_partial_graph_num_bits = None

  def __init__(self, precomputed: PrecomputedMaps):

    # These should be set by calling new_trial
    self.aligned_stones: Counter[AlignedStoneIndex] = collections.Counter()
    self.perceived_potions: Counter[PerceivedPotionIndex] = (
        collections.Counter())
    self.world_state_distribution = init_world_state_distribution(precomputed)

  def representative_potions(
      self, stone_index: AlignedStoneIndex, precomputed: PrecomputedMaps
  ) -> List[PerceivedPotionIndex]:
    """Gets a representative set of potions for this stone and belief state.

    Some potions will be equivalent if we don't know what they do and we have
    the same number of them and their counterparts on the same perceptual
    dimension. For each equivalence set we return one potion as a representative
    of the set.

    Args:
      stone_index: The stone to apply potions to.
      precomputed: Precomputed maps used for speed.

    Returns:
      A representative set of potions.
    """
    potion_to_count = [0 for _ in range(PerceivedPotion.num_types)]
    for p1 in self.perceived_potions:
      p2 = precomputed.potion_to_pair[p1]
      c1 = self.perceived_potions[p1]
      c2 = self.perceived_potions[p2]
      potion_to_count[p1] = (c1, c2)
    representative_potions = []
    for p2 in self.perceived_potions:
      equiv = False
      for p1 in representative_potions:
        b1 = (self.world_state_distribution.observed_no_effect_bits &
              precomputed.potion_masks[p1]) >> p1
        b2 = (self.world_state_distribution.observed_no_effect_bits &
              precomputed.potion_masks[p2]) >> p2
        if (potion_to_count[p1] == potion_to_count[p2] and
            (b1 == b2) and
            self.world_state_distribution.potions_equivalent(
                p1, p2, stone_index, precomputed)):
          equiv = True
          break
      if not equiv:
        representative_potions.append(p2)
    return representative_potions

  def _remove_stone(self, stone_index: AlignedStoneIndex) -> None:
    if stone_index in self.aligned_stones:
      self.aligned_stones[stone_index] -= 1
    if self.aligned_stones[stone_index] == 0:
      del self.aligned_stones[stone_index]

  def _add_stone(self, stone_index: AlignedStoneIndex) -> None:
    self.aligned_stones.update([stone_index])

  def _remove_potion(self, potion_index: PerceivedPotionIndex) -> None:
    self.perceived_potions.subtract([potion_index])
    if self.perceived_potions[potion_index] == 0:
      del self.perceived_potions[potion_index]

  def possible_actions(
      self, precomputed: PrecomputedMaps
  ) -> List[Tuple[AlignedStoneIndex, PerceivedPotionIndex]]:
    """Gets representative list of possible actions which have an effect."""
    # Use -1, -1 to represent ending and putting all stones in the cauldron or
    # throwing them away. If we consider this action first then in the event
    # that it has the same expected reward as using a potion we will take this
    # action instead. This gives more intuitive behaviour, for example if we
    # have the best stone we won't transform it to something less good and then
    # transform it back.
    poss_actions = [(AlignedStoneIndex(END_TRIAL),
                     PerceivedPotionIndex(END_TRIAL))]
    for s in self.aligned_stones:
      # Don't consider potions if we have observed that they have no effect.
      potions = [p for p in self.representative_potions(s, precomputed)
                 if not (self.world_state_distribution.observed_no_effect_bits &
                         stone_potion_bit_mask(s, p, precomputed))]
      poss_actions.extend([(s, p) for p in potions])

    return poss_actions

  def use_potion(
      self, stone_index: AlignedStoneIndex,
      potion_index: PerceivedPotionIndex,
      result_index: AlignedStoneIndex
  ) -> None:
    """Uses the potion on the current stone.

    Args:
      stone_index: The stone used in the potion.
      potion_index: The potion applied to the stone.
      result_index: The result observed.
    """
    # Remove the used potion
    self._remove_potion(potion_index)
    # If the stone has not changed then we don't need to do anything else.
    if stone_index == result_index:
      return
    # Remove the initial stone and replace with the result
    self._remove_stone(stone_index)
    self._add_stone(result_index)

  def action_and_outcome(
      self, stone_index: AlignedStoneIndex,
      potion_index: PerceivedPotionIndex, result_index: AlignedStoneIndex,
      precomputed: PrecomputedMaps, bit_mask: int
  ) -> float:
    """Updates the belief state given the action and observation.

    Args:
      stone_index: The stone used in the potion.
      potion_index: The potion in which the stone is used.
      result_index: The resulting stone.
      precomputed: Precomputed maps used for speed.
      bit_mask: Mask on observed no effect for the stone and potion passed.

    Returns:
      The probability given our prior belief state of this observation.
    """
    self.use_potion(stone_index, potion_index, result_index)

    # Update the world state distribution
    total_prob = self.world_state_distribution.action_and_outcome(
        stone_index, potion_index, result_index, precomputed, bit_mask)

    return total_prob

  def new_trial(
      self, aligned_stones: Counter[AlignedStoneIndex],
      perceived_potions: Counter[PerceivedPotionIndex]
  ) -> None:
    self.aligned_stones = copy.deepcopy(aligned_stones)
    self.perceived_potions = copy.deepcopy(perceived_potions)
    self.world_state_distribution.new_trial(aligned_stones)

  def to_bitfield(self) -> int:
    """Converts to a bitfield to cache results."""

    def perceived_potions_to_bits(
        perceived_potions: Mapping[PerceivedPotionIndex, int]
    ) -> Tuple[int, int]:
      """Converts the set of perceived potions to a bitfield."""
      local_int_rep = 0
      for potion_type, count in perceived_potions.items():
        local_int_rep |= (count << (
            PerceivedPotion.count_num_bits * potion_type))
        if count > PerceivedPotion.max_present:
          raise ValueError('Too many potions present.')
        if potion_type >= PerceivedPotion.num_types:
          raise ValueError('Invalid potion type.')
      return local_int_rep, (
          PerceivedPotion.num_types * PerceivedPotion.count_num_bits)

    def aligned_stones_to_bits(
        aligned_stones: Mapping[AlignedStoneIndex, int]
    ) -> Tuple[int, int]:
      """Converts the set of perceived stones to a bitfield."""
      local_int_rep = 0
      stone_number = 0
      for stone_type, count in sorted(aligned_stones.items()):
        for _ in range(count):
          local_int_rep |= (stone_type << (
              AlignedStone.num_bits * stone_number))
          stone_number += 1
        if stone_type >= stones_and_potions.AlignedStone.num_types:
          raise ValueError('Invalid stone type')
      if stone_number > AlignedStone.max_present:
        raise ValueError('Too many stones present.')
      return local_int_rep, AlignedStone.max_present * AlignedStone.num_bits

    all_things = [
        perceived_potions_to_bits(self.perceived_potions),
        aligned_stones_to_bits(self.aligned_stones),
        (self.world_state_distribution.observed_no_effect_bits,
         LatentStone.num_types * PerceivedPotion.num_types),
        (self.world_state_distribution.partial_potion_map_index[0],
         PartialPotionMap.num_bits_axis),
        (self.world_state_distribution.partial_potion_map_index[1],
         PartialPotionMap.num_bits_dir),
        (self.world_state_distribution.partial_stone_map_index,
         PartialStoneMap.num_bits),
        (self.world_state_distribution.partial_graph_index,
         BeliefState.possible_partial_graph_num_bits)
    ]

    return helpers.pack_to_bitfield(all_things)

  def __repr__(self) -> str:
    # Convert the observed_no_effect bitfield to a matrix before printing.
    observed_no_effect = np.zeros(
        (stones_and_potions.LatentStone.num_types,
         stones_and_potions.PerceivedPotion.num_types))
    for pe_st in range(stones_and_potions.LatentStone.num_types):
      for pe_po in range(stones_and_potions.PerceivedPotion.num_types):
        bit_num = (pe_st * stones_and_potions.PerceivedPotion.num_types) + pe_po
        observed_no_effect[pe_st, pe_po] = (
            self.world_state_distribution.observed_no_effect_bits &
            (1 << bit_num))

    return (
        'BeliefState(aligned_stones={aligned_stones}, '
        'perceived_potions={perceived_potions}, '
        'observed_no_effect={observed_no_effect}, '
        'poss_world_states={poss_world_states}, '
        'partial_potion_map_index={partial_potion_map_index}, '
        'partial_stone_map_index={partial_stone_map_index}, '
        'partial_graph_index={partial_graph_index}, '
        'partial_graph_possible={partial_graph_possible}, '
        'stone_map_possible={stone_map_possible}, '
        'potion_map_possible={potion_map_possible}, '
        'num_world_states = {num_world_states}, '.format(
            aligned_stones=self.aligned_stones,
            perceived_potions=self.perceived_potions,
            observed_no_effect=types_helpers.str_np_array_construct(
                observed_no_effect),
            poss_world_states=types_helpers.str_np_array_construct(
                self.world_state_distribution.poss_world_states),
            partial_potion_map_index=(
                self.world_state_distribution.partial_potion_map_index),
            partial_stone_map_index=(
                self.world_state_distribution.partial_stone_map_index),
            partial_graph_index=(
                self.world_state_distribution.partial_graph_index),
            partial_graph_possible=(
                self.world_state_distribution.partial_graph_possible),
            stone_map_possible=self.world_state_distribution.stone_map_possible,
            potion_map_possible=(
                self.world_state_distribution.potion_map_possible),
            num_world_states=len(self.world_state_distribution)))

  def update_stone_map(
      self, new_to_old: StoneMap
  ) -> None:
    """If we assumed the wrong rotation we may need to swap stone map dims."""
    # Change poss stone maps.
    old_aligned_stones = {
        stones_and_potions.aligned_stone_from_index(stone): count
        for stone, count in self.aligned_stones.items()}
    new_aligned_stones = collections.Counter({
        AlignedStone(stone.reward, new_to_old.apply(
            stone).latent_coords).index(): count
        for stone, count in old_aligned_stones.items()})
    self.aligned_stones = new_aligned_stones
    self.world_state_distribution.update_stone_map(new_to_old)

  @property
  def num_world_states(self) -> int:
    return np.where(self.world_state_distribution.poss_world_states)[0].size

  @property
  def num_potion_maps(self) -> int:
    return len(self.world_state_distribution.potion_map_possible)

  @property
  def num_stone_maps(self) -> int:
    return len(self.world_state_distribution.stone_map_possible)

  @property
  def num_graphs(self) -> int:
    return len(self.world_state_distribution.get_possible_graphs())

  def partial_potion_map(
      self, index_to_perm_index: np.ndarray
  ) -> PartialPotionMap:
    return stones_and_potions.partial_potion_map_from_index(
        self.world_state_distribution.partial_potion_map_index,
        index_to_perm_index)

  def partial_stone_map(self) -> PartialStoneMap:
    return stones_and_potions.partial_stone_map_from_index(
        self.world_state_distribution.partial_stone_map_index)

  def partial_graph(
      self, possible_partial_graph_indices: np.ndarray
  ) -> PartialGraph:
    return graphs.partial_graph_from_index(
        possible_partial_graph_indices[
            self.world_state_distribution.partial_graph_index])


class BeliefStateWithRotation:
  """Belief state over chem including rotations."""

  def __init__(self, precomputed: precomputed_maps.PrecomputedMaps):
    self.belief_state = BeliefState(precomputed)
    self.possible_rotations = stones_and_potions.possible_rotations()
    self.rotation = None
    # We need to know 1 stone which is consistent with the selected rotation.
    self._observed_stone = None
    self._rotation_to_angles = (
        lambda rotation: tuple(stones_and_potions.rotation_to_angles(rotation)))
    stone_map_indices = [
        sm.index() for sm in stones_and_potions.possible_stone_maps()]
    self._stone_maps_for_rotation = {
        self._rotation_to_angles(rotation): copy.deepcopy(stone_map_indices)
        for rotation in stones_and_potions.possible_rotations()}

  def _update_given_stones(
      self, perceived_stones: Sequence[PerceivedStone]
  ) -> None:
    """Updates the possible rotations and belief state given observed stones."""
    if not perceived_stones:
      raise ValueError(
          'Must pass perceived stones to update possible rotations.')

    # Given the stones we see can we eliminate some possible rotations.
    valid_rotations = []
    for rotation in self.possible_rotations:
      # For a rotation to be possible all stones have to go to corners of the
      # cube and the change in latent variables has to be consistent with the
      # change in reward (i.e. at least one stone map gives the observed
      # rewards).
      aligned_stones = []
      rotation_valid = True
      for stone in perceived_stones:
        valid, coords = stones_and_potions.aligns(stone, rotation)
        if valid:
          aligned_stones.append(stones_and_potions.aligned_stone_from_coords(
              coords, stone.reward))
        else:
          rotation_valid = False
          break

      if rotation_valid:
        stone_maps = self._stone_maps_for_rotation[self._rotation_to_angles(
            rotation)]
        _, possible_stone_maps = get_possible_stone_maps(
            stone_maps, aligned_stones)
        self._stone_maps_for_rotation[self._rotation_to_angles(
            rotation)] = possible_stone_maps
        if possible_stone_maps:
          valid_rotations.append(rotation)
    assert valid_rotations, 'No rotation is valid.'
    self.possible_rotations = valid_rotations
    if self.rotation is None:
      self.rotation = self.possible_rotations[0]
      self._observed_stone = stones_and_potions.align(
          perceived_stones[0], self.rotation)
    elif not stones_and_potions.rotations_equal(
        self.rotation, self.possible_rotations[0]):
      new_to_old = stones_and_potions.get_new_mapping_to_old_mapping(
          self.rotation, self.possible_rotations[0], self._observed_stone)
      self.belief_state.update_stone_map(new_to_old)
      self.rotation = self.possible_rotations[0]
      self._observed_stone = stones_and_potions.align(
          perceived_stones[0], self.rotation)

  def new_trial(
      self, perceived_stones: Counter[PerceivedStone],
      perceived_potions: Counter[PerceivedPotion]
  ) -> None:
    """Updates belief state given that new trial has started."""
    self._update_given_stones(list(perceived_stones.keys()))
    aligned_stones = collections.Counter(
        {stones_and_potions.align(stone, self.rotation).index(): count
         for stone, count in perceived_stones.items()})
    perceived_potion_indices = collections.Counter(
        {potion.index(): count for potion, count in perceived_potions.items()})
    self.belief_state.new_trial(aligned_stones, perceived_potion_indices)

  def action_and_outcome(
      self, stone: PerceivedStone, potion: PerceivedPotion,
      result: PerceivedStone, precomputed: PrecomputedMaps
  ) -> float:
    self._update_given_stones([result])
    stone_index = stones_and_potions.align(stone, self.rotation).index()
    result_index = stones_and_potions.align(result, self.rotation).index()
    bit_mask = stone_potion_bit_mask(stone_index, potion.index(), precomputed)
    return self.belief_state.action_and_outcome(
        stone_index, potion.index(), result_index, precomputed, bit_mask)

  @property
  def num_world_states(self) -> int:
    return self.belief_state.num_world_states

  @property
  def num_potion_maps(self) -> int:
    return self.belief_state.num_potion_maps

  @property
  def num_stone_maps(self) -> int:
    return self.belief_state.num_stone_maps

  @property
  def num_graphs(self) -> int:
    return self.belief_state.num_graphs

  def partial_potion_map(
      self, index_to_perm_index: np.ndarray
  ) -> PartialPotionMap:
    return self.belief_state.partial_potion_map(index_to_perm_index)

  def partial_stone_map(self) -> PartialStoneMap:
    return self.belief_state.partial_stone_map()

  def partial_graph(
      self, possible_partial_graph_indices: np.ndarray
  ) -> PartialGraph:
    return self.belief_state.partial_graph(possible_partial_graph_indices)


def search(
    belief_state: BeliefState,
    search_results: SearchResults,
    bonus: int, precomputed: PrecomputedMaps,
    depth: int = 0,
    minimise_world_states: bool = False
) -> ActionObjective:
  """Searches iteratively over actions and outcomes to find expected reward.

  Conducts a depth first search over the DAG of available actions and the
  possible outcomes. The reward for a latent stone is assumed to be the sum of
  the latent values plus the passed bonus if all latent values are positive. We
  do not deal with reward functions with arbitrary coefficient vectors and
  offsets.

  Args:
    belief_state: The current belief state.
    search_results: A cache of previously computed results mapping the belief
      state as a bitfield to the best action and expected reward.
    bonus: The extra reward we get by reaching the stone of reward 3.
    precomputed: Precomputed maps used for speed.
    depth: Number of actions taken in this search to reach this belief state.
    minimise_world_states: Let the objective be to minimise the number of world
      states at the end of the trial instead of to maximise the accumulated
      reward. Actions selected will not necessarily produce reward but will
      narrow down the possible chemistries, e.g. given a maximum value stone on
      the first trial and a potion we would use the potion to find out the
      effect even though it could reduce the value of the stone.

  Returns:
    The best action and maximum expected reward achievable from this state.
  """

  # If we have searched from this belief state before return the cached result.
  belief_state_bitfield = belief_state.to_bitfield()
  if belief_state_bitfield in search_results:
    return search_results[belief_state_bitfield]

  # For all possible actions consider all possible outcomes and then search from
  # the outcome.
  action_rewards = {}
  for stone_index, potion_index in belief_state.possible_actions(precomputed):
    # This action means use or discard all stones.
    if potion_index == END_TRIAL:
      # Ending the trial so get the actual number of world states
      if minimise_world_states:
        expected_num_world_states = np.where(
            belief_state.world_state_distribution.poss_world_states)[0].size
      else:
        raw_reward_count = [(precomputed.stone_to_reward[s], c) for s, c in
                            belief_state.aligned_stones.items()]
        reward_per_stone_type = [
            0.0 if reward < 0 else c * (reward + bonus)
            if reward == stones_and_potions.max_reward() else c * reward
            for reward, c in raw_reward_count]
        action_reward = sum(reward_per_stone_type)
        best_action_depth = depth
    else:
      bit_mask = stone_potion_bit_mask(stone_index, potion_index, precomputed)
      if belief_state.world_state_distribution.observed_no_effect_bits & bit_mask:
        continue
      # Using a potion on a stone could lead to a number of possible outcomes
      # with various probabilities. The ideal observer must calculate what
      # reward it can expect to obtain for each of these.
      if minimise_world_states:
        # The expected number of world states when the trial ends
        expected_num_world_states = 0
      else:
        action_reward = 0.0
        best_action_depth = 0.0
      poss_outcomes, could_stay_still = (
          belief_state.world_state_distribution.possible_outcomes(
              potion_index, stone_index, precomputed))
      if len(poss_outcomes) == 1:
        # If there is only one possibility and it is staying still then there is
        # no need to search this action as it will have no effect.
        if could_stay_still:
          continue
        new_game_state = copy.deepcopy(belief_state)
        new_game_state.use_potion(stone_index, potion_index, poss_outcomes[0])
        _, objective = search(
            new_game_state, search_results, bonus, precomputed, depth + 1,
            minimise_world_states)
        if minimise_world_states:
          expected_num_world_states -= objective
        else:
          search_reward, neg_search_action_depth = objective
          action_reward += search_reward
          best_action_depth -= neg_search_action_depth
      else:
        for outcome in poss_outcomes:
          new_game_state = copy.deepcopy(belief_state)
          prob = new_game_state.action_and_outcome(
              stone_index, potion_index, outcome, precomputed, bit_mask)
          if prob > 0:
            _, objective = search(
                new_game_state, search_results, bonus, precomputed, depth + 1,
                minimise_world_states)
            if minimise_world_states:
              expected_num_world_states -= prob * objective
            else:
              search_reward, neg_search_action_depth = objective
              action_reward += prob * search_reward
              best_action_depth -= prob * neg_search_action_depth
    # Store the expected reward and the negative search depth so that when we
    # maximise reward, if 2 actions have the same expected reward then we will
    # take the action with the minimum search depth. This prevents us taking
    # actions which do not harm our expected reward but do not help us.
    if minimise_world_states:
      action_rewards[
          (stone_index, potion_index)] = -expected_num_world_states
    else:
      action_rewards[(stone_index, potion_index)] = (
          action_reward, -best_action_depth)

  result = max(action_rewards.items(), key=lambda a: a[1])

  search_results[belief_state_bitfield] = result
  return result


def ideal_observer(
    init_game_state: BeliefState,
    search_results: SearchResults,
    bonus: int, precomputed: PrecomputedMaps, minimise_world_states: bool
) -> ActionObjectiveAndSearchResults:
  """Runs the ideal observer given a set of stones and potions.

  This runs an exhaustive search from the initial state over possible actions
  and possible outcomes of those actions. It returns which action to take, the
  expected reward and a set of search results for belief states encountered.

  Args:
    init_game_state: The initial belief state of the system.
    search_results: Previously computed action and reward for belief states.
    bonus: The additional reward for getting the best stone.
    precomputed: Precomputed maps used for speed.
    minimise_world_states: Let the objective be to minimise the number of world
      states at the end of the trial instead of to maximise the accumulated
      reward.

  Returns:
    The best action to take, the expected reward and a set of search results for
    belief states encountered.
  """
  # Set the number of bits required to fit all possible partial graph indices.
  BeliefState.possible_partial_graph_num_bits = math.ceil(math.log2(
      len(precomputed.partial_graph_index_to_possible_index)))

  # Run the search over all possible next actions
  action, objective = search(
      init_game_state, search_results, bonus, precomputed,
      minimise_world_states=minimise_world_states)
  return action, objective, search_results
