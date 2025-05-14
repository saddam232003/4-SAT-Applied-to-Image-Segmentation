import numpy as np
import cv2
import matplotlib.pyplot as plt
from pysat.formula import CNF
from pysat.solvers import Minisat22

def generate_synthetic_retina_mask(size=128):
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size//2, size//2), size//3, 1, -1)
    return mask

def corrupt_mask(mask, missing_ratio=0.3):
    corrupted = mask.copy()
    coords = np.argwhere(mask == 1)
    np.random.shuffle(coords)
    n_missing = int(len(coords) * missing_ratio)
    for y, x in coords[:n_missing]:
        corrupted[y, x] = 0
    return corrupted

def mask_to_variables(mask):
    h, w = mask.shape
    return np.arange(1, h * w + 1).reshape(h, w)

def build_4sat_clauses(mask, var_map):
    h, w = mask.shape
    cnf = CNF()
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            v = var_map[y, x]
            neighbors = [var_map[y-1, x], var_map[y+1, x], var_map[y, x-1], var_map[y, x+1]]
            # If 3+ neighbors are foreground, then center likely is too
            known = [n for n in neighbors if mask.flatten()[n - 1] != 0]
            if len(known) >= 3:
                clause = [v] + known[:3]  # Add positive clause
                cnf.append(clause)
    return cnf

def solve_sat(cnf, var_map):
    solver = Minisat22(bootstrap_with=cnf.clauses)
    if solver.solve():
        model = solver.get_model()
        model_set = set(model)
        sat_mask = np.zeros_like(var_map, dtype=np.uint8)
        for y in range(var_map.shape[0]):
            for x in range(var_map.shape[1]):
                var = var_map[y, x]
                if var in model_set:
                    sat_mask[y, x] = 1
        return sat_mask
    else:
        print("Unsatisfiable!")
        return np.zeros_like(var_map, dtype=np.uint8)

def visualize_results(original, partial, predicted):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ['Original', 'Corrupted', 'SAT-Reconstructed']
    for ax, img, title in zip(axes, [original, partial, predicted], titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def dice_score(pred, target):
    intersection = np.sum(pred * target)
    return (2. * intersection) / (np.sum(pred) + np.sum(target) + 1e-8)

# === RUN EXPERIMENT ===
original_mask = generate_synthetic_retina_mask()
corrupted_mask = corrupt_mask(original_mask)
var_map = mask_to_variables(corrupted_mask)
cnf = build_4sat_clauses(corrupted_mask, var_map)
reconstructed_mask = solve_sat(cnf, var_map)

# === VISUALIZE ===
visualize_results(original_mask, corrupted_mask, reconstructed_mask)

# === METRIC ===
print("Dice Score:", round(dice_score(reconstructed_mask, original_mask), 4))
