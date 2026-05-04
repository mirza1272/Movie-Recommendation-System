from collections import deque
import heapq
def bfs_search(movies, top_n=10):
    visited = []
    queue = deque(range(len(movies)))
    seen = set()
    while queue and len(visited) < top_n:
        idx = queue.popleft()
        if idx not in seen:
            seen.add(idx)
            visited.append(idx)
    return movies.iloc[visited]

def dfs_search(movies, top_n=10):
    visited = []
    stack = list(range(len(movies) - 1, -1, -1))
    seen = set()
    while stack and len(visited) < top_n:
        idx = stack.pop()
        if idx not in seen:
            seen.add(idx)
            visited.append(idx)
    return movies.iloc[visited]

def astar_search(movies, top_n=10):
    if "heuristic_score" not in movies.columns:
        return movies.head(top_n)
    heap = []
    for idx, row in movies.iterrows():
        heapq.heappush(heap, (-row["heuristic_score"], idx))
    result_indices = []
    while heap and len(result_indices) < top_n:
        _, idx = heapq.heappop(heap)
        result_indices.append(idx)
    return movies.loc[result_indices]

def run_search(movies, algorithm="A*", top_n=10):
    if algorithm == "BFS":
        return bfs_search(movies, top_n)
    elif algorithm == "DFS":
        return dfs_search(movies, top_n)
    else:
        return astar_search(movies, top_n)
