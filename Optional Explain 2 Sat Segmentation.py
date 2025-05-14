import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk
from pysat.solvers import Minisat22

# Step 1: Generate synthetic medical image
def generate_medical_image(size=(128, 128), radius=30):
    image = np.zeros(size)
    rr, cc = disk((size[0]//2, size[1]//2), radius)
    image[rr, cc] = 1
    return image

# Step 2: Simulate partial mask with missing labels
def create_partial_mask(mask, missing_ratio=0.3):
    partial_mask = mask.copy()
    foreground_coords = np.array(np.nonzero(mask)).T
    num_foreground = len(foreground_coords)
    missing_pixels = int(missing_ratio * num_foreground)
    np.random.shuffle(foreground_coords)
    for i in range(missing_pixels):
        y, x = foreground_coords[i]
        partial_mask[y, x] = 0
    return partial_mask

# Step 3: SAT-based reconstruction
def sat_reconstruct(partial_mask):
    reconstructed = partial_mask.copy()
    unknown_coords = np.array(np.nonzero(partial_mask == 0)).T
    var_map = {tuple(coord): i+1 for i, coord in enumerate(unknown_coords)}
    
    clauses = []
    for coord, var in var_map.items():
        y, x = coord
        neighbors = [(y-1,x), (y+1,x), (y,x-1), (y,x+1)]
        neighbor_vars = []
        for ny, nx in neighbors:
            if 0 <= ny < partial_mask.shape[0] and 0 <= nx < partial_mask.shape[1]:
                if partial_mask[ny, nx] == 1:
                    clauses.append([var])  # Encourage positive
                elif partial_mask[ny, nx] == 0 and (ny, nx) in var_map:
                    neighbor_vars.append(var_map[(ny, nx)])
        if len(neighbor_vars) >= 3:
            clauses.append([var] + neighbor_vars[:3])  # 4-SAT style

    solver = Minisat22()
    for clause in clauses:
        solver.add_clause(clause)
    
    solution = solver.solve()
    model = solver.get_model()

    if model:
        for coord, var in var_map.items():
            if var in model:
                reconstructed[coord] = 1
    return reconstructed

# Step 4: Evaluation
def calculate_metrics(gt, pred):
    intersection = np.logical_and(gt, pred)
    union = np.logical_or(gt, pred)
    iou = np.sum(intersection) / np.sum(union)
    dice = 2 * np.sum(intersection) / (np.sum(gt) + np.sum(pred))
    return iou, dice

# Step 5: Visualization
def visualize_masks(original, partial, reconstructed):
    fig, axs = plt.subplots(1, 4, figsize=(14, 4))
    axs[0].imshow(original, cmap='gray')
    axs[0].set_title("Original Mask")
    axs[1].imshow(partial, cmap='gray')
    axs[1].set_title("Partial Mask")
    axs[2].imshow(reconstructed, cmap='gray')
    axs[2].set_title("SAT Reconstructed")
    axs[3].imshow(original != reconstructed, cmap='hot')
    axs[3].set_title("Error Map")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Main pipeline
original = generate_medical_image()
partial = create_partial_mask(original, 0.3)
reconstructed = sat_reconstruct(partial)
iou, dice = calculate_metrics(original, reconstructed)

visualize_masks(original, partial, reconstructed)

print(f"IoU: {iou:.4f}")
print(f"Dice Coefficient: {dice:.4f}")
