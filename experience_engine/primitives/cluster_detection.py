"""Cluster detection via BFS — pure-symbol and wild-aware variants.

Clusters are connected groups of the same standard symbol that meet the
minimum size threshold. Wilds bridge adjacent standard-symbol groups and
can participate in multiple clusters of different symbol types (R-WILD-5).

Detection is a pure function — no mutation of the board.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from ..config.schema import BoardConfig, MasterConfig
from .board import Board, Position, orthogonal_neighbors
from .symbols import Symbol, is_standard, is_wild


@dataclass(frozen=True, slots=True)
class Cluster:
    """A detected cluster of connected same-symbol positions.

    Payout fields (base_payout, total_payout) are NOT stored here — they
    belong to the Paytable module (SRP). Cluster is a pure detection result.
    """

    # The standard symbol forming this cluster
    symbol: Symbol
    # Standard-symbol positions in the cluster
    positions: frozenset[Position]
    # Wild positions participating in this cluster
    wild_positions: frozenset[Position]
    # Total size = standard + wild positions (C-COMPAT-1)
    size: int


def detect_clusters(board: Board, config: MasterConfig) -> list[Cluster]:
    """Detect all qualifying clusters on the board using wild-aware BFS.

    Algorithm:
    1. Find connected components of each standard symbol (ignoring wilds)
    2. For each wild, note which symbol components it neighbors
    3. Merge same-symbol components that share a wild (C-COMPAT-2)
    4. Wilds count toward cluster size (C-COMPAT-1)
    5. Wilds can participate in clusters of different symbol types (R-WILD-5)
    6. Filter by min_cluster_size
    """
    board_config = config.board
    symbol_config = config.symbols
    min_size = board_config.min_cluster_size

    # Step 1: find pure-symbol connected components
    # component_id → (symbol, set of positions)
    components: dict[int, tuple[Symbol, set[Position]]] = {}
    # position → component_id for standard symbol positions
    pos_to_comp: dict[Position, int] = {}
    comp_id = 0

    visited: set[Position] = set()
    for reel in range(board_config.num_reels):
        for row in range(board_config.num_rows):
            pos = Position(reel, row)
            if pos in visited:
                continue
            sym = board.get(pos)
            if sym is None or not is_standard(sym, symbol_config):
                continue

            # BFS for this symbol
            component_positions: set[Position] = set()
            queue: deque[Position] = deque([pos])
            visited.add(pos)
            while queue:
                current = queue.popleft()
                component_positions.add(current)
                pos_to_comp[current] = comp_id
                for neighbor in orthogonal_neighbors(current, board_config):
                    if neighbor not in visited and board.get(neighbor) is sym:
                        visited.add(neighbor)
                        queue.append(neighbor)

            components[comp_id] = (sym, component_positions)
            comp_id += 1

    # Step 2: find wild positions and which components they neighbor
    wild_positions: list[Position] = []
    # wild_pos → set of component IDs it neighbors
    wild_neighbors: dict[Position, set[int]] = {}
    for reel in range(board_config.num_reels):
        for row in range(board_config.num_rows):
            pos = Position(reel, row)
            sym = board.get(pos)
            if sym is not None and is_wild(sym):
                wild_positions.append(pos)
                neighbor_comp_ids: set[int] = set()
                for neighbor in orthogonal_neighbors(pos, board_config):
                    if neighbor in pos_to_comp:
                        neighbor_comp_ids.add(pos_to_comp[neighbor])
                wild_neighbors[pos] = neighbor_comp_ids

    # Step 3: merge same-symbol components that share a wild (C-COMPAT-2)
    # Use union-find to track merges
    parent: dict[int, int] = {cid: cid for cid in components}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for wild_pos in wild_positions:
        neighbor_ids = wild_neighbors.get(wild_pos, set())
        # Group neighbor components by symbol
        symbol_groups: dict[Symbol, list[int]] = defaultdict(list)
        for cid in neighbor_ids:
            sym, _ = components[cid]
            symbol_groups[sym].append(cid)

        # Merge components of the same symbol that share this wild
        for sym, comp_ids in symbol_groups.items():
            for i in range(1, len(comp_ids)):
                union(comp_ids[0], comp_ids[i])

    # Step 4: collect merged components
    merged: dict[int, tuple[Symbol, set[Position]]] = {}
    for cid, (sym, positions) in components.items():
        root = find(cid)
        if root not in merged:
            merged[root] = (sym, set())
        merged[root][1].update(positions)

    # Step 5: assign wilds to merged components (R-WILD-5 — multi-participation)
    # A wild participates in a cluster if it neighbors any component in that merged group
    wild_assignments: dict[int, set[Position]] = defaultdict(set)
    for wild_pos in wild_positions:
        neighbor_ids = wild_neighbors.get(wild_pos, set())
        # Find which merged groups this wild neighbors
        assigned_roots: set[int] = set()
        for cid in neighbor_ids:
            root = find(cid)
            assigned_roots.add(root)
        for root in assigned_roots:
            wild_assignments[root].add(wild_pos)

    # Step 6: build Cluster objects, filtering by min_cluster_size
    clusters: list[Cluster] = []
    for root, (sym, positions) in merged.items():
        wilds = frozenset(wild_assignments.get(root, set()))
        total_size = len(positions) + len(wilds)
        if total_size >= min_size:
            clusters.append(Cluster(
                symbol=sym,
                positions=frozenset(positions),
                wild_positions=wilds,
                size=total_size,
            ))

    return clusters


def detect_components(
    board: Board, symbol: Symbol, board_config: BoardConfig
) -> list[frozenset[Position]]:
    """Find all connected components of a single symbol (no wild awareness).

    Returns components regardless of size — useful for WFC propagator checks.
    """
    visited: set[Position] = set()
    result: list[frozenset[Position]] = []

    for reel in range(board_config.num_reels):
        for row in range(board_config.num_rows):
            pos = Position(reel, row)
            if pos in visited or board.get(pos) is not symbol:
                continue

            component: set[Position] = set()
            queue: deque[Position] = deque([pos])
            visited.add(pos)
            while queue:
                current = queue.popleft()
                component.add(current)
                for neighbor in orthogonal_neighbors(current, board_config):
                    if neighbor not in visited and board.get(neighbor) is symbol:
                        visited.add(neighbor)
                        queue.append(neighbor)

            result.append(frozenset(component))

    return result


def max_component_size(
    board: Board,
    symbol: Symbol,
    board_config: BoardConfig,
    extra: frozenset[Position] | None = None,
    wild_positions: frozenset[Position] | None = None,
) -> int:
    """Largest connected component size for a symbol, optionally treating
    extra positions as if they also held that symbol.

    Used by WFC propagator to check whether placing a symbol would create
    a component exceeding the threshold.

    wild_positions — when provided, wilds are treated as same-symbol during
    BFS traversal, matching detect_clusters() union-find semantics.
    """
    extra_set = extra or frozenset()
    wild_set = wild_positions or frozenset()
    visited: set[Position] = set()
    max_size = 0

    for reel in range(board_config.num_reels):
        for row in range(board_config.num_rows):
            pos = Position(reel, row)
            if pos in visited:
                continue
            if board.get(pos) is not symbol and pos not in extra_set:
                continue

            size = 0
            queue: deque[Position] = deque([pos])
            visited.add(pos)
            while queue:
                current = queue.popleft()
                size += 1
                for neighbor in orthogonal_neighbors(current, board_config):
                    if neighbor not in visited:
                        # Wild positions count as same-symbol — prevents
                        # wild-mediated extensions past the cluster threshold
                        if (board.get(neighbor) is symbol
                                or neighbor in extra_set
                                or neighbor in wild_set):
                            visited.add(neighbor)
                            queue.append(neighbor)

            max_size = max(max_size, size)

    return max_size
