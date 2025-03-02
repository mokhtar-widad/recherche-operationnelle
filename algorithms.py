import random
import networkx as nx
from collections import deque

def generate_random_graph(num_nodes, edge_probability, min_weight, max_weight):
    G = nx.Graph()
    nodes = list(range(num_nodes))
    G.add_nodes_from(nodes)
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if random.random() < edge_probability:
                weight = random.randint(min_weight, max_weight)
                G.add_edge(i, j, weight=weight)
    return G

def welsh_powell(G):
    nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
    colors = {}
    for node in nodes:
        used_colors = set(colors.get(neighbor) for neighbor in G.neighbors(node) if neighbor in colors)
        for color in range(len(G)):
            if color not in used_colors:
                colors[node] = color
                break
    return colors

def kruskal(G):
    edges = sorted(G.edges(data=True), key=lambda t: t[2].get('weight', 1))
    mst = []
    total_cost = 0
    uf = {node: node for node in G.nodes()}

    def find(item):
        if uf[item] != item:
            uf[item] = find(uf[item])
        return uf[item]

    def union(x, y):
        uf[find(x)] = find(y)

    num_nodes = len(G.nodes())
    for u, v, attr in edges:
        if find(u) != find(v):
            union(u, v)
            mst.append((u, v, attr['weight']))
            total_cost += attr['weight']
            if len(mst) == num_nodes - 1:
                break

    return mst, total_cost

def generate_transportation_problem(m, n):
    supply = [random.randint(50, 200) for _ in range(m)]
    demand = [random.randint(30, 150) for _ in range(n)]
    
    # Ajuster l'offre ou la demande pour équilibrer le problème
    total_supply = sum(supply)
    total_demand = sum(demand)
    if total_supply > total_demand:
        demand[-1] += total_supply - total_demand
    elif total_demand > total_supply:
        supply[-1] += total_demand - total_supply
    
    costs = [[random.randint(10, 100) for _ in range(n)] for _ in range(m)]
    return supply, demand, costs

def format_transportation_table(supply, demand, costs):
    m, n = len(supply), len(demand)
    table = [[""] * (n + 2) for _ in range(m + 2)]
    
    # En-têtes de colonnes
    table[0][0] = "S/D"
    for j in range(n):
        table[0][j + 1] = f"D{j+1}"
    table[0][-1] = "Supply"
    
    # En-têtes de lignes et données
    for i in range(m):
        table[i + 1][0] = f"S{i+1}"
        for j in range(n):
            table[i + 1][j + 1] = str(costs[i][j])
        table[i + 1][-1] = str(supply[i])
    
    # Ligne de demande
    table[-1][0] = "Demand"
    for j in range(n):
        table[-1][j + 1] = str(demand[j])
    
    return table

def northwest_corner(supply, demand):
    m, n = len(supply), len(demand)
    result = [[0 for _ in range(n)] for _ in range(m)]
    i, j = 0, 0
    
    while i < m and j < n:
        quantity = min(supply[i], demand[j])
        result[i][j] = quantity
        supply[i] -= quantity
        demand[j] -= quantity
        if supply[i] == 0:
            i += 1
        if demand[j] == 0:
            j += 1
    
    return result

def least_cost(supply, demand, costs):
    m, n = len(supply), len(demand)
    result = [[0 for _ in range(n)] for _ in range(m)]
    
    while sum(supply) > 0 and sum(demand) > 0:
        min_cost = float('infinity')
        min_i, min_j = -1, -1
        for i in range(m):
            for j in range(n):
                if supply[i] > 0 and demand[j] > 0 and costs[i][j] < min_cost:
                    min_cost = costs[i][j]
                    min_i, min_j = i, j
        
        quantity = min(supply[min_i], demand[min_j])
        result[min_i][min_j] = quantity
        supply[min_i] -= quantity
        demand[min_j] -= quantity
    
    return result

def calculate_total_cost(solution, costs):
    return sum(solution[i][j] * costs[i][j] for i in range(len(costs)) for j in range(len(costs[0])))

def dijkstra(G, start):
    distances = {node: float('infinity') for node in G.nodes()}
    distances[start] = 0
    pq = [(0, start)]
    predecessors = {start: None}

    while pq:
        current_distance, current_node = min(pq)
        pq.remove((current_distance, current_node))

        for neighbor in G.neighbors(current_node):
            weight = G[current_node][neighbor]['weight']
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                pq.append((distance, neighbor))

    return distances, predecessors

def bellman_ford(G, start):
    distances = {node: float('infinity') for node in G.nodes()}
    distances[start] = 0
    predecessors = {node: None for node in G.nodes()}

    for _ in range(len(G) - 1):
        for u, v, attr in G.edges(data=True):
            weight = attr['weight']
            if distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                predecessors[v] = u

    for u, v, attr in G.edges(data=True):
        if distances[u] + attr['weight'] < distances[v]:
            raise ValueError("Le graphe contient un cycle de poids négatif")

    return distances, predecessors

def metra_potential(tasks):
    G = nx.DiGraph()
    for task, info in tasks.items():
        G.add_node(task, duration=info['duration'])
        for pred in info['predecessors']:
            G.add_edge(pred, task)

    early = {task: 0 for task in tasks}
    for task in nx.topological_sort(G):
        early[task] = max([early[pred] + tasks[pred]['duration'] for pred in G.predecessors(task)] or [0])

    late = {task: max(early.values()) for task in tasks}
    for task in reversed(list(nx.topological_sort(G))):
        late[task] = min([late[succ] - tasks[task]['duration'] for succ in G.successors(task)] or [late[task]])

    return early, late

def ford_fulkerson(G, source, sink):
    def bfs(G, source, sink, parent):
        visited = {node: False for node in G.nodes()}
        queue = deque([source])
        visited[source] = True
        while queue:
            u = queue.popleft()
            for v in G.neighbors(u):
                if not visited[v] and G[u][v]['capacity'] > G[u][v].get('flow', 0):
                    queue.append(v)
                    visited[v] = True
                    parent[v] = u
                    if v == sink:
                        return True
        return False

    parent = {node: None for node in G.nodes()}
    max_flow = 0

    while bfs(G, source, sink, parent):
        path_flow = float('infinity')
        s = sink
        while s != source:
            path_flow = min(path_flow, G[parent[s]][s]['capacity'] - G[parent[s]][s].get('flow', 0))
            s = parent[s]
        
        max_flow += path_flow

        v = sink
        while v != source:
            u = parent[v]
            G[u][v]['flow'] = G[u][v].get('flow', 0) + path_flow
            G[v][u]['flow'] = G[v][u].get('flow', 0) - path_flow
            v = parent[v]

    return max_flow

def stepping_stone(supply, demand, costs, initial_solution):
    m, n = len(supply), len(demand)

    # Fonction pour trouver le chemin Stepping Stone
    def find_stepping_stone_path(i, j):
        path = [(i, j)]  # Initialiser le chemin avec la case actuelle
        row_used, col_used = set(), set()

        # Recherche du chemin via DFS
        def dfs(i, j, direction):
            if direction == 'row':
                for jj in range(n):
                    if jj != j and initial_solution[i][jj] > 0 and (i, jj) not in path:
                        path.append((i, jj))
                        result = dfs(i, jj, 'col')
                        if result:
                            return result
                        path.pop()
            else:
                for ii in range(m):
                    if ii != i and initial_solution[ii][j] > 0 and (ii, j) not in path:
                        path.append((ii, j))
                        result = dfs(ii, j, 'row')
                        if result:
                            return result
                        path.pop()
            return None
        
        # Démarrer la recherche du chemin à partir de la ligne ou colonne
        return dfs(i, j, 'row') or dfs(i, j, 'col')
    
    # L'algorithme Stepping Stone
    while True:
        max_reduction = 0
        max_i, max_j = -1, -1
        
        # Recherche du maximum de réduction possible pour chaque cellule
        for i in range(m):
            for j in range(n):
                if initial_solution[i][j] == 0:
                    path = find_stepping_stone_path(i, j)
                    if path:
                        reduction = sum((-1)**k * costs[x][y] for k, (x, y) in enumerate(path))
                        if reduction < max_reduction:
                            max_reduction = reduction
                            max_i, max_j = i, j
        
        # Si aucune amélioration n'est trouvée, on sort
        if max_reduction >= 0:
            break
        
        # Appliquer la valeur minimale (theta) sur le chemin trouvé
        path = find_stepping_stone_path(max_i, max_j)
        theta = min(initial_solution[i][j] for k, (i, j) in enumerate(path) if k % 2 == 1)
        
        # Mise à jour de la solution
        for k, (i, j) in enumerate(path):
            initial_solution[i][j] += theta if k % 2 == 0 else -theta
    
    return initial_solution



def solve_transportation_problem(supply, demand, costs):
    # Apply Northwest Corner method
    nw_solution = northwest_corner(supply.copy(), demand.copy())
    nw_cost = calculate_total_cost(nw_solution, costs)
    print(f"Total cost (Northwest Corner): {nw_cost}")
    
    # Apply Least Cost method
    lc_solution = least_cost(supply.copy(), demand.copy(), costs)
    lc_cost = calculate_total_cost(lc_solution, costs)
    print(f"Total cost (Least Cost): {lc_cost}")
    
    # Choose the solution with the lowest cost
    if nw_cost <= lc_cost:
        initial_solution = nw_solution
        initial_cost = nw_cost
        method = "Northwest Corner"
    else:
        initial_solution = lc_solution
        initial_cost = lc_cost
        method = "Least Cost"
    
    # Apply Stepping Stone method to the chosen solution
    optimized_solution = stepping_stone(supply, demand, costs, initial_solution)
    optimized_cost = calculate_total_cost(optimized_solution, costs)
    
    return {
        "initial_method": method,
        "initial_solution": initial_solution,
        "initial_cost": initial_cost,
        "optimized_solution": optimized_solution,
        "optimized_cost": optimized_cost
    }

