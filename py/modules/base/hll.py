from typing import List

def union_hll(hll_sets: List[str], cur_hll) -> str:
    "Union two or more HLL sets and return the combined HLL set"
    hll_values = ", ".join(f"('{hll_set}'::hll)" for hll_set in hll_sets)
    db_query = f"""
        SELECT hll_union_agg(s.hll_set)
        FROM (
        VALUES
            {hll_values}
        ) s(hll_set);
    """
    cur_hll.execute(db_query)
    # return results as first from tuple
    return cur_hll.fetchone()[0]

def cardinality_hll(hll_set, cur_hll) -> int:
    "Calculate the cardinality of a HLL set and return as int"
    cur_hll.execute(f"SELECT hll_cardinality('{hll_set}')::int;")
    return int(cur_hll.fetchone()[0])