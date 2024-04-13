import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import csv

def calculate_path_length_with_strategy(G, strategy, alpha=1.0, num_trials=100):
    """
    Calculate average path length using a specific strategy.
    Strategies include 'random_walk', 'max_degree', 'min_degree', 'preferential_attachment'.
    Alpha parameter is used for the preferential attachment strategy.
    """
    np.random.seed(42)  # For reproducibility
    total_path_length = 0
    
    for _ in range(num_trials):
        if strategy == 'random_walk':
            total_path_length += modified_random_walk(G)
        else:
            total_path_length += path_length_with_degree_strategy(G, strategy, alpha)
            
    return total_path_length / num_trials

def modified_random_walk(G):
    """
    Perform a modified random walk that avoids revisiting the same edge consecutively and computes the path length correctly.
    """
    nodes = list(G.nodes())
    start_node = np.random.choice(nodes)
    current_node = start_node
    previous_node = None
    path_length = 0

    while True:
        neighbors = [n for n in G.neighbors(current_node) if n != previous_node]
        if not neighbors:
            break  # End the walk if no other neighbors
        next_node = np.random.choice(neighbors)
        path_length += 1
        if next_node == start_node:
            break  # End the walk if it returns to the start node
        previous_node, current_node = current_node, next_node

    return path_length

def path_length_with_degree_strategy(G, strategy, alpha):
    """
    Calculate path length using degree-based strategies correctly.
    """
    path_lengths = []
    for _ in range(100):  # For each trial
        nodes = list(G.nodes())
        start_node = np.random.choice(nodes)
        current_node = start_node
        total_steps = 0
        
        while total_steps < 1000:  # Limit the number of steps to avoid infinite loop
            if strategy == 'max_degree':
                next_node = max(G.neighbors(current_node), key=G.degree)
            elif strategy == 'min_degree':
                next_node = min(G.neighbors(current_node), key=G.degree)
            elif strategy == 'preferential_attachment':
                neighbors = list(G.neighbors(current_node))
                degrees = np.array([G.degree(n)**alpha for n in neighbors])
                probabilities = degrees / np.sum(degrees)
                next_node = np.random.choice(neighbors, p=probabilities)
            total_steps += 1
            if next_node == start_node:
                break
            current_node = next_node
        path_lengths.append(total_steps)

    return np.mean(path_lengths)

if __name__ == '__main__':
    N = 1000
    strategy = 'preferential_attachment'
    alpha_values = np.linspace(0.5, 3.0, 11)
    results = []

    # Save results to CSV
    with open('alpha_variation_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['alpha', 'Average Path Length'])
        
        for alpha in alpha_values:
            G = nx.barabasi_albert_graph(N, 2)
            D = calculate_path_length_with_strategy(G, strategy, alpha=alpha)
            results.append((alpha, D))
            writer.writerow([alpha, D])
            
    # Visualization
    alphas, path_lengths = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, path_lengths, marker='o')
    plt.xlabel('Alpha')
    plt.ylabel('Average Path Length')
    plt.title(f'Average Path Length vs. Alpha for {strategy} Strategy')
    plt.grid(True)
    plt.show()