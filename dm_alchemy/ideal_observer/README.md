## Ideal Observer

This page explains the alchemy ideal observer, knowledge of how the alchemy task
works is assumed.

The alchemy ideal observer is an algorithm which calculates the expected value
of all possible actions given a set of knowledge about the state of the world.

This knowledge consists of:

-   The perceivable attributes of the set of potions available in the current
    trial and the set of stones available in the current trial. For stones this
    includes the reward that would be obtained by placing the stone in the
    cauldron and a position in perceptual space ( $$\in \{-1, 1\}^3$$ ), i.e. if
    we see a small round blue stone we will say it is at $$[-1, -1, 1]$$ in
    stone perceptual space coordinates for example (following the scheme in
    unity the axes are in the order colour, size, roundness and blue is less
    than red). In general the stone may be at a different set of coordinates in
    latent space and to maximise reward the ideal observer must perform
    experiments which determine the mapping from perceptual space to latent
    space. Although we allow 45 degree rotations this does not affect our
    ability to map perceptual attributes to points on this cube. For potions we
    observe the potion colour which we can map to a dimension and direction in
    potion perceptual space because we know that pairs of potion colours are
    always opposite directions on the same dimension. Again the ideal observer
    will have to perform experiments to determine the mapping from potion
    perceptual space to latent space.

-   A set of world states and associated probabilities where a world state is a:

    -   Mapping from potion perceptual space to latent space (`PotionMap`).
    -   Mapping from stone perceptual space to latent space (`StoneMap`).
    -   Graph describing which edges exist in latent space.

    The probability of each world state begins as the prior which is the
    frequency that each is drawn from the set. As we make observations or
    hypothesize them we can eliminate some world states and renormalise the
    probabilities of the remaining ones.

The algorithm then consists of an exhaustive depth first search over possible
actions (over which we maximise) and possible outcomes (over which we take the
expectation).

## Code Layout

The code is split into a file which contains the main ideal observer algorithm
(`ideal_observer.py`) and files for code relating to stones and potions
(`stones_and_potions.py`) and code relating to graphs (`graphs.py`). For speed
we convert types into indices (e.g. there are 109 possible graphs so we replace
the actual graphs with the numbers 0-108) and precompute maps for certain
functions which go from all possible input indices to the corresponding output
indices. The conversions to and from indices are defined next to the type
definitions. The computation of the maps is done in a separate file to avoid
cluttering up the main files (`precomputed_maps.py`).

In `stones_and_potions.py` we have separate types for what we can perceive
(`PerceivedStone` and `PerceivedPotion`) and their positions in the underlying
latent space (`LatentStone` and `LatentPotion`). It is important to keep these
as separate types despite their similarity so that we can easily define
interfaces which ensure that we do not leak information that the ideal observer
should not be privy to.
