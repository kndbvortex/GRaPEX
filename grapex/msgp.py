"""Seasonal gradual pattern mining (MSGP-style) on a set of MTS instances."""

import numpy as np
from typing import List, Set, Tuple, Dict
from dataclasses import dataclass


@dataclass(frozen=True)
class GradualItem:
    attribute: int
    variation: str

    def __repr__(self):
        return f"{self.attribute}{self.variation}"


@dataclass
class GradualPattern:
    items: Set[GradualItem]
    regions: List[Tuple[int, int]]
    support: float
    length: int

    def __repr__(self):
        items_str = " ".join(str(item) for item in sorted(self.items, key=lambda x: x.attribute))
        return f"GP({items_str}) | regions={self.regions} | support={self.support:.2f} | length={self.length}"


class MSGPMax:
    def __init__(self, data: np.ndarray, column_names: List[str] = None):
        if data.ndim != 3:
            raise ValueError(f"Expected 3D array (samples, timesteps, channels), got {data.ndim}D")
        self.data = data
        self.n_samples, self.n_timesteps, self.n_channels = data.shape
        self.column_names = column_names or [f"ch_{i}" for i in range(self.n_channels)]

    def _find_covers(self, sequence: np.ndarray, direction: str) -> List[Tuple[int, int]]:
        covers = []
        start = 0

        for t in range(1, len(sequence)):
            valid = (direction == "↑" and sequence[t] > sequence[t-1]) or \
                    (direction == "↓" and sequence[t] < sequence[t-1])

            if not valid:
                if t - start >= 2:
                    covers.append((start, t - 1))
                start = t

        if len(sequence) - start >= 2:
            covers.append((start, len(sequence) - 1))

        return covers

    def _covers_contain_region(self, covers: List[Tuple[int, int]], region: Tuple[int, int]) -> bool:
        r_start, r_end = region
        return any(c_start <= r_start and c_end >= r_end for c_start, c_end in covers)

    def _get_all_covers(self, channel: int, direction: str) -> List[List[Tuple[int, int]]]:
        return [self._find_covers(self.data[s, :, channel], direction) for s in range(self.n_samples)]

    def _find_maximal_regions(self, all_covers: List[List[Tuple[int, int]]],
                              min_samples: int, min_length: int) -> List[Tuple[Tuple[int, int], int]]:
        regions = []

        for t_start in range(self.n_timesteps - min_length + 1):
            for t_end in range(t_start + min_length - 1, self.n_timesteps):
                region = (t_start, t_end)
                count = sum(1 for covers in all_covers if self._covers_contain_region(covers, region))

                if count >= min_samples:
                    regions.append((region, count))

        maximal = []
        for region, count in regions:
            is_maximal = True
            for other_region, other_count in regions:
                if other_region != region and other_count >= count:
                    if other_region[0] <= region[0] and other_region[1] >= region[1]:
                        if other_region[0] < region[0] or other_region[1] > region[1]:
                            is_maximal = False
                            break
            if is_maximal:
                maximal.append((region, count))

        return maximal

    def extract_patterns(self, min_support: float = 0.5, min_length: int = 7) -> List[GradualPattern]:
        if not 0 < min_support <= 1:
            raise ValueError(f"min_support must be in (0, 1], got {min_support}")

        min_samples = max(1, int(np.ceil(min_support * self.n_samples)))

        item_regions: Dict[GradualItem, List[Tuple[Tuple[int, int], int]]] = {}

        for ch in range(self.n_channels):
            for direction in ["↑", "↓"]:
                item = GradualItem(ch, direction)
                all_covers = self._get_all_covers(ch, direction)
                maximal_regions = self._find_maximal_regions(all_covers, min_samples, min_length)
                if maximal_regions:
                    item_regions[item] = maximal_regions

        patterns = []

        for item, regions_with_count in item_regions.items():
            for region, count in regions_with_count:
                pattern = GradualPattern(
                    items={item},
                    regions=[region],
                    support=count / self.n_samples,
                    length=region[1] - region[0] + 1
                )
                patterns.append(pattern)

        combined = self._combine_patterns(item_regions, min_samples)
        patterns.extend(combined)

        patterns = self._remove_duplicates(patterns)
        patterns.sort(key=lambda p: (-len(p.items), -p.length, -p.support))

        return patterns

    def _remove_duplicates(self, patterns: List[GradualPattern]) -> List[GradualPattern]:
        unique = {}
        for p in patterns:
            key = (frozenset(p.items), p.regions[0])
            if key not in unique or p.support > unique[key].support:
                unique[key] = p

        result = list(unique.values())

        final = []
        for p in result:
            dominated = False
            for other in result:
                if p is other:
                    continue
                if p.items == other.items and p.support <= other.support:
                    p_start, p_end = p.regions[0]
                    o_start, o_end = other.regions[0]
                    if o_start <= p_start and o_end >= p_end and (o_start < p_start or o_end > p_end):
                        dominated = True
                        break
            if not dominated:
                final.append(p)

        return final

    def _combine_patterns(self, item_regions: Dict[GradualItem, List[Tuple[Tuple[int, int], int]]],
                          min_samples: int) -> List[GradualPattern]:
        combined = []
        items_list = list(item_regions.keys())

        for i, item1 in enumerate(items_list):
            for item2 in items_list[i+1:]:
                if item1.attribute == item2.attribute:
                    continue

                for region1, count1 in item_regions[item1]:
                    for region2, count2 in item_regions[item2]:
                        overlap_start = max(region1[0], region2[0])
                        overlap_end = min(region1[1], region2[1])

                        if overlap_end - overlap_start + 1 >= 2:
                            overlap_region = (overlap_start, overlap_end)

                            covers1 = self._get_all_covers(item1.attribute, item1.variation)
                            covers2 = self._get_all_covers(item2.attribute, item2.variation)

                            count = sum(
                                1 for c1, c2 in zip(covers1, covers2)
                                if self._covers_contain_region(c1, overlap_region) and
                                   self._covers_contain_region(c2, overlap_region)
                            )

                            if count >= min_samples:
                                pattern = GradualPattern(
                                    items={item1, item2},
                                    regions=[overlap_region],
                                    support=count / self.n_samples,
                                    length=overlap_end - overlap_start + 1
                                )
                                combined.append(pattern)

        return combined
