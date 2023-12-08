from typing import List, Any, Set


def add_items_to_group(items: List[Any], groups: List[Set[Any]]):
    """
    Grouping items based on their relationships
    @param items: List of items that has to be in the same group
    @param groups: List of groups that we will add the items to, should be an empty list by default
    """
    reference_group = {}
    for g_id, group in enumerate(groups):
        for fragment_id in items:
            if fragment_id in group and g_id not in reference_group:
                reference_group[g_id] = group

    if len(reference_group) > 0:
        reference_ids = list(reference_group.keys())
        for fragment_id in items:
            reference_group[reference_ids[0]].add(fragment_id)
        for g_id in reference_ids[1:]:
            for fragment_id in reference_group[g_id]:
                reference_group[reference_ids[0]].add(fragment_id)
            del groups[g_id]
    else:
        groups.append(set(items))
