#https://web.archive.org/web/20110523210824/http://www.nils-haldenwang.de/computer-science/computational-intelligence/genetic-algorithm-vs-0-1-knapsack
#https://www.kdnuggets.com/2023/01/knapsack-problem-genetic-programming-python.html

import random

class Item:
    def __init__(self, name, weight, value):
        self.name, self.weight, self.value = name, float(weight), float(value)

def load_items(filename):
    return [Item(*line.strip().split(',')) for line in open(filename) if line.strip()]

def genetic_knapsack(items, capacity, pop_size=100, generations=50):
    population = [[random.randint(0,1) for _ in items] for _ in range(pop_size)]
    best_fitness = 0
    
    for generation in range(generations):
       
        fitness_scores = []
        for chrom in population:
            total_weight = sum(items[i].weight * chrom[i] for i in range(len(items)))
            total_value = sum(items[i].value * chrom[i] for i in range(len(items)))
            fitness_scores.append(total_value if total_weight <= capacity else 0)
        
    
        new_population = []
        for _ in range(pop_size):
          
            p1, p2 = random.sample(population, 2)
            parent1 = p1 if fitness_scores[population.index(p1)] > fitness_scores[population.index(p2)] else p2
            p1, p2 = random.sample(population, 2)
            parent2 = p1 if fitness_scores[population.index(p1)] > fitness_scores[population.index(p2)] else p2
            

            if random.random() < 0.7:
                point = random.randint(1, len(items)-1)
                child = parent1[:point] + parent2[point:]
            else:
                child = parent1 if random.random() < 0.5 else parent2
            
         
            for i in range(len(child)):
                if random.random() < 1/len(items):
                    child[i] = 1 - child[i]
            
            new_population.append(child)
        
        population = new_population
        current_best = max(fitness_scores)
        if current_best > best_fitness:
            best_fitness = current_best
    
    best_solution = max(population, key=lambda c: sum(items[i].value * c[i] for i in range(len(items))) 
                      if sum(items[i].weight * c[i] for i in range(len(items))) <= capacity else 0)
    return best_solution

#base
items = []
for file in ["lada1.txt", "lada2.txt", "lada3.txt", "lada4.txt", "lada5.txt"]:
    items.extend(load_items(file))

capacity = float(input("Hátizsák kapacitása (kg): "))
solution = genetic_knapsack(items, capacity)

print("\nOptimális loot:")
total_weight = total_value = 0
for i, selected in enumerate(solution):
    if selected:
        item = items[i]
        print(f"- {item.name}: {item.weight}kg, {item.value:.0f} érték")
        total_weight += item.weight
        total_value += item.value

print(f"\nÖsszesen: {total_weight:.1f}kg / {capacity}kg")
print(f"Teljes érték: {total_value:.0f}")
print(f"Kihasználtság: {(total_weight/capacity*100):.1f}%")