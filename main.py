import tkinter as tk
from tkinter import ttk, scrolledtext
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import string

from algorithms import (welsh_powell, kruskal, northwest_corner, dijkstra, 
                        bellman_ford, least_cost, metra_potential, 
                        ford_fulkerson, stepping_stone, generate_random_graph,
                        generate_transportation_problem, format_transportation_table,
                        calculate_total_cost, solve_transportation_problem)

class AlgorithmGUI:
    def __init__(self, master):
        self.master = master
        master.title("Algorithmes de Recherche Opérationnelle - EMSI")
        master.geometry("1000x700")
        master.configure(bg="#2C3E50")

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#2C3E50")
        style.configure("TButton", background="#3498DB", foreground="white", font=('Arial', 12, 'bold'))
        style.map("TButton", background=[('active', '#2980B9')])
        style.configure("TLabel", background="#2C3E50", foreground="white", font=('Arial', 14))

        self.main_frame = ttk.Frame(master, padding="20")
        self.main_frame.pack(expand=True, fill="both")

        self.title_label = ttk.Label(self.main_frame, text="Algorithmes RO et Théorie des Graphes", 
                                     font=("Arial", 24, "bold"), foreground="#ECF0F1")
        self.title_label.pack(pady=20)

        self.algorithms = [
            "Welsh-Powell", "Kruskal", "Nord-Ouest et Moindre Coût", "Dijkstra",
            "Bellman-Ford", "Potentiel METRA", "Ford-Fulkerson"
        ]

        self.buttons_frame = ttk.Frame(self.main_frame)
        self.buttons_frame.pack(fill="x", pady=10)

        for i, algo in enumerate(self.algorithms):
            btn = ttk.Button(self.buttons_frame, text=algo, 
                             command=lambda a=algo: self.show_algorithm(a))
            btn.grid(row=i//3, column=i%3, padx=10, pady=5, sticky="ew")

        for i in range(3):
            self.buttons_frame.columnconfigure(i, weight=1)

    def show_algorithm(self, algorithm):
        new_window = tk.Toplevel(self.master)
        new_window.title(algorithm)
        new_window.geometry("900x700")
        new_window.configure(bg="#34495E")

        content = ttk.Frame(new_window, padding="20")
        content.pack(expand=True, fill="both")

        ttk.Label(content, text=algorithm, font=("Arial", 20, "bold"), foreground="#ECF0F1").pack(pady=10)

        result_text = scrolledtext.ScrolledText(content, wrap=tk.WORD, width=80, height=20, 
                                                font=("Courier", 12))
        result_text.pack(pady=10, padx=10, expand=True, fill="both")

        run_button = ttk.Button(content, text="Exécuter l'algorithme", 
                                command=lambda: self.run_algorithm(algorithm, result_text, new_window))
        run_button.pack(pady=10)

    def get_num_nodes(self):
        dialog = tk.Toplevel(self.master)
        dialog.title("Nombre de sommets")
        dialog.geometry("300x150")
        dialog.configure(bg="#34495E")

        ttk.Label(dialog, text="Entrez le nombre de sommets :", font=("Arial", 12), 
                  background="#34495E", foreground="white").pack(pady=10)
        num_nodes_var = tk.IntVar()
        ttk.Entry(dialog, textvariable=num_nodes_var, font=("Arial", 12)).pack(pady=5)

        def confirm():
            dialog.destroy()

        ttk.Button(dialog, text="Confirmer", command=confirm).pack(pady=10)
        dialog.wait_window()
        return num_nodes_var.get()

    def run_algorithm(self, algorithm, result_text, window):
        result_text.delete('1.0', tk.END)
        result_text.insert(tk.END, f"Exécution de l'algorithme {algorithm}...\n\n")

        if algorithm in ["Welsh-Powell", "Kruskal", "Dijkstra", "Bellman-Ford"]:
            num_nodes = self.get_num_nodes()

        if algorithm == "Welsh-Powell":
            G = generate_random_graph(num_nodes, 0.5, 1, 10)
            coloring = welsh_powell(G)
            result_text.insert(tk.END, f"Coloration : {coloring}\n")
            self.draw_graph(G, coloring, window, algorithm="Welsh-Powell")

        elif algorithm == "Kruskal":
            edge_probability = 0.7
            min_weight = 1
            max_weight = 20
            G = generate_random_graph(num_nodes, edge_probability, min_weight, max_weight)
            mst, total_cost = kruskal(G)
            result_text.insert(tk.END, f"Nombre de sommets : {num_nodes}\n")
            result_text.insert(tk.END, f"Nombre d'arêtes sélectionnées : {len(mst)}\n")
            result_text.insert(tk.END, f"Coût total minimum : {total_cost}\n")
            result_text.insert(tk.END, "Arêtes de l'arbre couvrant minimum :\n")
            for u, v, weight in mst:
                result_text.insert(tk.END, f"({u}, {v}) : poids {weight}\n")
            self.draw_graph(G, mst, window, algorithm="Kruskal")

        elif algorithm == "Nord-Ouest et Moindre Coût":
            m, n = 3, 3
            supply, demand, costs = generate_transportation_problem(m, n)
            table = format_transportation_table(supply, demand, costs)
            result_text.insert(tk.END, "Tableau initial du problème de transport :\n")
            for row in table:
                result_text.insert(tk.END, " | ".join(f"{cell:^10}" for cell in row) + "\n")
            
            # Calcul avec la méthode Nord-Ouest
            nw_solution = northwest_corner(supply.copy(), demand.copy())
            nw_cost = calculate_total_cost(nw_solution, costs)
            result_text.insert(tk.END, f"\nCoût total (Méthode Nord-Ouest) : {nw_cost}\n")

            # Calcul avec la méthode du Moindre Coût
            lc_solution = least_cost(supply.copy(), demand.copy(), costs)
            lc_cost = calculate_total_cost(lc_solution, costs)
            result_text.insert(tk.END, f"Coût total (Méthode du Moindre Coût) : {lc_cost}\n")

        elif algorithm == "Dijkstra":
            G = generate_random_graph(num_nodes, 0.7, 1, 10)
            start = list(G.nodes())[0]
            distances, predecessors = dijkstra(G, start)
            result_text.insert(tk.END, "Distances depuis le sommet de départ :\n")
            for node, distance in distances.items():
                result_text.insert(tk.END, f"Distance de {start} à {node} : {distance}\n")

        elif algorithm == "Bellman-Ford":
            edge_probability = 0.4
            min_weight = 1
            max_weight = 10
            G = generate_random_graph(num_nodes, edge_probability, min_weight, max_weight)
            start = list(G.nodes())[0]
            distances, predecessors = bellman_ford(G, start)
            result_text.insert(tk.END, "Distances depuis le sommet de départ :\n")
            for node, distance in distances.items():
                result_text.insert(tk.END, f"Distance de {start} à {node} : {distance}\n")

        elif algorithm == "Potentiel METRA":
            tasks = {
                'A': {'duration': random.randint(1, 5), 'predecessors': []},
                'B': {'duration': random.randint(1, 5), 'predecessors': ['A']},
                'C': {'duration': random.randint(1, 5), 'predecessors': ['A']},
                'D': {'duration': random.randint(1, 5), 'predecessors': ['B', 'C']},
                'E': {'duration': random.randint(1, 5), 'predecessors': ['C']},
                'F': {'duration': random.randint(1, 5), 'predecessors': ['D', 'E']}
            }

            # Calculer les dates au plus tôt et au plus tard avec l'algorithme METRA
            try:
                early, late = metra_potential(tasks)
                result_text.insert(tk.END, f"Dates au plus tôt : {early}\n")
                result_text.insert(tk.END, f"Dates au plus tard : {late}\n")
            except Exception as e:
                result_text.insert(tk.END, f"Erreur dans le calcul de METRA : {e}\n")
                return

            # Création et visualisation du graphe
            G = nx.DiGraph()

            # Ajouter les nœuds avec les durées comme attribut
            for task, info in tasks.items():
                G.add_node(task, duration=info['duration'])

            # Ajouter les arêtes basées sur les prédécesseurs
            for task, info in tasks.items():
                for pred in info['predecessors']:
                    G.add_edge(pred, task)

            # Dessiner le graphe avec le titre correspondant
            self.draw_graph(
                G,
                window=window,
                title="Graphe Potentiel METRA",
                algorithm="Potentiel METRA"
            )

        elif algorithm == "Ford-Fulkerson":
            G = nx.DiGraph()
            edges = [('s', 'a'), ('s', 'c'), ('a', 'b'), ('a', 'd'), ('b', 't'), ('c', 'd'), ('d', 't')]
            for u, v in edges:
                G.add_edge(u, v, capacity=random.randint(1, 10))
            source, sink = 's', 't'
            max_flow = ford_fulkerson(G, source, sink)
            result_text.insert(tk.END, f"Flux maximum : {max_flow}\n")

    def draw_graph(self, G, highlight=None, window=None, title="", algorithm=""):
        fig, ax = plt.subplots(figsize=(12, 10))
        pos = nx.spring_layout(G, k=1.5, iterations=50)
        node_size = 1000
        font_size = 14
        edge_width = 2
        if algorithm == "Welsh-Powell" and isinstance(highlight, dict):
            node_colors = [highlight.get(node, 0) for node in G.nodes()]
            nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors, 
                    node_size=node_size, font_size=font_size, font_weight='bold', cmap=plt.cm.Set3)
        elif algorithm == "Kruskal" and highlight:
            nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', 
                    node_size=node_size, font_size=font_size, font_weight='bold')
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=highlight, edge_color='r', width=edge_width)
        else:
            nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', 
                    node_size=node_size, font_size=font_size, font_weight='bold')
        plt.title(title, fontsize=20)
        plt.axis('off')
        if window:
            for widget in window.winfo_children():
                if isinstance(widget, tk.Canvas):
                    widget.destroy()
            canvas = FigureCanvasTkAgg(fig, master=window)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        else:
            plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    gui = AlgorithmGUI(root)
    root.mainloop()