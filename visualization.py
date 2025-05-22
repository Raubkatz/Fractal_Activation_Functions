import torch
import numpy as np
from matplotlib.animation import FuncAnimation

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib   
matplotlib.use('Agg')

SEED = 42

class ModulatedBlancmange(nn.Module):
    
    def __init__(self, n_terms=30, a=0.75):
        super().__init__()
        self.n_terms = n_terms
        self.a = a

    def forward(self, x):
        y = torch.zeros_like(x)
        for n in range(self.n_terms):
            factor = 2 ** n
            modulation = torch.tanh(self.a * factor * x)
            ax = self.a * torch.sqrt(torch.abs(x))
            y += modulation * torch.abs(x * factor % 2 - 1 * ax) / factor
        return y / 2
    
class ModifiedWeierstrass(nn.Module):

    def __init__(self, n_terms=30, a=0.75):
        super().__init__()
        self.n_terms = n_terms
        self.a = a

    # PyTorch version of the ModifiedWeierstrass forward method
    def forward(self, x):
        b = 3  # You may want to make b a parameter

        w = torch.zeros_like(x, dtype=torch.float64)
        for n in range(self.n_terms):
            w += ((-1) ** n) * (self.a ** n) * torch.cos((b ** n) * np.pi * x)

        combined_function = w * torch.exp(-0.75 * torch.abs(x)) + torch.relu(x)
        return combined_function.to(torch.float32)
    
def plot_curve(curve, x_vals):
    y_vals = curve(x_vals)
    plt.figure()
    plt.plot(x_vals.numpy(), y_vals.detach().numpy(), label='ModifiedWeierstrass')
    plt.title('Modified Weierstrass Curve')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig("plots/modified_weierstrass_curve.png")


    
# Plot the ModifiedWeierstrass curve
if __name__ == "__main__":
    # Generate x values
    x_vals = torch.linspace(0, 1, 1000)
    mb = ModifiedWeierstrass()
    #plot_curve(mb, x_vals)

    

# Define the neural network
class SimpleNet(nn.Module):
    def __init__(self, activation_function: type = nn.ReLU, size: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, size),
            activation_function(),
            nn.Linear(size, size),
            activation_function(),
            nn.Linear(size, 1)
        )

    def forward(self, x):
        return self.net(x)
    

TARGET_FUNCTIONS = {
    #"linear": lambda x: x,
    #"quadratic": lambda x: x ** 2,
    "sine": lambda x: torch.sin(2 * np.pi * x),
    #"exp": lambda x: torch.exp(x),
    #"log": lambda x: torch.log(x + 1e-5),  # Avoid log(0)
    #"modulated_blancmange": ModulatedBlancmange(),
    #"modified_weierstrass": ModifiedWeierstrass(),
}
ACTIVATION_FUNCTIONS = {
    'ReLU': nn.ReLU,
    #'Tanh': nn.Tanh,
    'Sigmoid': nn.Sigmoid,
    #'LeakyReLU': nn.LeakyReLU,
    #'ELU': nn.ELU,
    #'SELU': nn.SELU,
    #'GELU': nn.GELU,
    #'Softplus': nn.Softplus,
    'modulated_blancmange': ModulatedBlancmange,
    'modified_weierstrass': ModifiedWeierstrass,
}
NN_SIZES = [8]#2, 4, 16, 32, 64]

    


def set_seed(seed):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

def generate_plot():
    for nn_size in NN_SIZES:
        for tf_name, target_funciton in tqdm(TARGET_FUNCTIONS.items()):
            # Create folder for this target function if it doesn't exist
            target_dir = f'plots/{tf_name}'
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            for af_name, activation_function in ACTIVATION_FUNCTIONS.items():
                # Set seed for reproducibility
                set_seed(SEED)

                # Generate data
                x = np.linspace(0, 1, 200).reshape(-1, 1)
                y = target_funciton(torch.tensor(x, dtype=torch.float32)).numpy()

                x_tensor = torch.tensor(x, dtype=torch.float32)
                y_tensor = torch.tensor(y, dtype=torch.float32)

                # Initialize network, loss, optimizer
                net = SimpleNet(activation_function=activation_function)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(net.parameters(), lr=0.01)

                # For animation: store predictions at intervals
                predictions = []
                losses = []
                weight_trajectory = []

                # Train the network
                EPOCHS = 1000
                INTERVAL = 10  # Store every 10 EPOCHS
                for epoch in range(EPOCHS):
                    optimizer.zero_grad()
                    outputs = net(x_tensor)
                    loss = criterion(outputs, y_tensor)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                    if epoch % INTERVAL == 0 or epoch == EPOCHS - 1:
                        with torch.no_grad():
                            pred = net(x_tensor).detach().numpy()
                            predictions.append(pred)
                        
                             # Store current weights
                            current_weights = [p.data.clone() for p in net.parameters()]
                            weight_trajectory.append(current_weights)

                combined_animation_path = f'{target_dir}/combined_{af_name}_{nn_size}_{tf_name}.gif'
                create_combined_animation(
                    net, criterion, x_tensor, y_tensor, x, y, 
                    weight_trajectory, predictions, 
                    af_name, tf_name, nn_size, INTERVAL,
                    combined_animation_path
                )

def get_pca_dims(weight_trajectory, original_weights, model):
    # Calculate all parameter displacements during training
    all_displacements = []
    for weights in weight_trajectory:
        # Flatten and concatenate all parameter differences
        displacement = torch.cat([
            (w - original_weights[i]).flatten() 
            for i, w in enumerate(weights)
        ])
        all_displacements.append(displacement)
    
    # Stack all displacements
    all_disps_tensor = torch.stack(all_displacements)
    
    # Center the data for PCA
    mean_disp = torch.mean(all_disps_tensor, dim=0)
    centered_disps = all_disps_tensor - mean_disp
    
    # Compute principal components using SVD
    try:
        U, S, V = torch.svd(centered_disps)
        
        # Use the top two principal components as directions
        direction1 = []
        direction2 = []
        idx = 0
        for i, p in enumerate(model.parameters()):
            shape = p.data.shape
            size = p.data.numel()
            direction1.append(V[:, 0][idx:idx+size].reshape(shape))
            direction2.append(V[:, 1][idx:idx+size].reshape(shape))
            idx += size
            
        return direction1, direction2
    except Exception as e:
        return None, None

def create_combined_animation(model, criterion, x_tensor, y_tensor, x_plot, y_plot, 
                              weight_trajectory, predictions, af_name, tf_name, nn_size, 
                              interval, save_path):
    """
    Create side-by-side animation of loss landscape and function fitting
    """
    set_seed(SEED)

    # Store original weights
    original_weights = [p.data.clone() for p in model.parameters()]

    direction1, direction2 = get_pca_dims(weight_trajectory, original_weights, model)
    
    
    if direction1 is not None and direction2 is not None:
        print(f"PCA successful")
    else:
        print(f"PCA failed. Falling back to random directions")
        # Generate two random directions
        direction1 = [torch.randn_like(p) for p in model.parameters()]
        direction2 = [torch.randn_like(p) for p in model.parameters()]
    
    # Normalize directions
    d1_norm = torch.sqrt(sum(torch.sum(d**2) for d in direction1))
    d2_norm = torch.sqrt(sum(torch.sum(d**2) for d in direction2))
    
    for i in range(len(direction1)):
        direction1[i] /= d1_norm
        direction2[i] /= d2_norm
    
    # Project weight trajectory onto the two directions
    trajectory_coords = []
    trajectory_losses = []
    
    for weights in weight_trajectory:
        # Calculate displacement from original weights
        displacement = [w - original_weights[i] for i, w in enumerate(weights)]
        
        # Project displacement onto direction1 and direction2
        alpha = sum(torch.sum(d1 * d) for d1, d in zip(direction1, displacement))
        beta = sum(torch.sum(d2 * d) for d2, d in zip(direction2, displacement))
        
        # Set weights and compute loss
        for idx, w in enumerate(model.parameters()):
            w.data = weights[idx].clone()
        
        outputs = model(x_tensor)
        loss = criterion(outputs, y_tensor).item()
        
        trajectory_coords.append((alpha.item(), beta.item()))
        trajectory_losses.append(loss)

    # Extract x and y coordinates from trajectory
    trajectory_x, trajectory_y = zip(*trajectory_coords)
    
    # Determine adaptive plot limits
    min_x, max_x = min(trajectory_x), max(trajectory_x)
    min_y, max_y = min(trajectory_y), max(trajectory_y)
    
    # Add padding (20% of range on each side)
    x_range = max_x - min_x
    y_range = max_y - min_y
    
    # Use the larger range to ensure square aspect ratio for direction space
    max_range = max(x_range, y_range)
    padding = max_range * 0.2
    
    # If range is very small, use a minimum size
    if max_range < 0.1:
        padding = 0.5
    
    # Create grid of points for loss landscape with adaptive limits
    alpha_min = min_x - padding
    alpha_max = max_x + padding
    beta_min = min_y - padding
    beta_max = max_y + padding
    
    alpha_range = np.linspace(alpha_min, alpha_max, 21)
    beta_range = np.linspace(beta_min, beta_max, 21)
    loss_grid = np.zeros((len(alpha_range), len(beta_range)))
    
    # Compute loss values for grid
    for i, alpha in enumerate(alpha_range):
        for j, beta in enumerate(beta_range):
            # Update weights
            for idx, w in enumerate(model.parameters()):
                w.data = original_weights[idx] + alpha * direction1[idx] + beta * direction2[idx]
            
            # Compute loss
            outputs = model(x_tensor)
            loss = criterion(outputs, y_tensor).item()
            loss_grid[i, j] = loss

    # Project weight trajectory onto the two directions
    trajectory_coords = []
    trajectory_losses = []
    
    for weights in weight_trajectory:
        # Calculate displacement from original weights
        displacement = [w - original_weights[i] for i, w in enumerate(weights)]
        
        # Project displacement onto direction1 and direction2
        alpha = sum(torch.sum(d1 * d) for d1, d in zip(direction1, displacement))
        beta = sum(torch.sum(d2 * d) for d2, d in zip(direction2, displacement))
        
        # Set weights and compute loss
        for idx, w in enumerate(model.parameters()):
            w.data = weights[idx].clone()
        
        outputs = model(x_tensor)
        loss = criterion(outputs, y_tensor).item()
        
        trajectory_coords.append((alpha.item(), beta.item()))
        trajectory_losses.append(loss)
    
    # Restore original weights
    for idx, w in enumerate(model.parameters()):
        w.data.copy_(original_weights[idx])
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(20, 8))
    
    # Loss landscape subplot
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot surface
    alpha_mesh, beta_mesh = np.meshgrid(alpha_range, beta_range)
    surface = ax1.plot_surface(alpha_mesh, beta_mesh, loss_grid, cmap='viridis', alpha=0.75, edgecolor='none')

    # Add a small vertical line (stem) from each trajectory point to the surface
    # to help with depth perception
    stem_lines = []
    for i in range(len(trajectory_coords)):
        x, y = trajectory_coords[i]
        loss = trajectory_losses[i]
        # This creates a thin line connecting the point to the surface
        stem, = ax1.plot([x, x], [y, y], [0, loss], 'r-', alpha=0.3, linewidth=0.5)
        stem_lines.append(stem)
        stem.set_visible(False)  # Initially hidden

    # Add colorbar
    fig.colorbar(surface, ax=ax1, shrink=0.5, aspect=5, label='Loss value')
    
    # Setup for loss landscape animation
    line, = ax1.plot([], [], [], 'r-', linewidth=2, label='Optimization path')
    point = ax1.scatter([], [], [], color='red', s=100, marker='o')
    
    ax1.set_xlabel('Direction 1')
    ax1.set_ylabel('Direction 2')
    ax1.set_zlabel('Loss')
    ax1.set_title(f'Loss Landscape: {af_name}')
    
    # Function fitting subplot
    ax2 = fig.add_subplot(122)
    line_true, = ax2.plot(x_plot, y_plot, label=f'{tf_name} function')
    line_pred, = ax2.plot(x_plot, predictions[0], label=f'{af_name} prediction', color='orange')
    ax2.set_ylim(min(y_plot) - 0.2, max(y_plot) + 0.2)
    ax2.set_title(f'Function Approximation')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend()
    
    fig.suptitle(f'{af_name} (size {nn_size}) learning {tf_name} function - Epoch 0', fontsize=16)
    
    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point._offsets3d = ([], [], [])
        line_pred.set_ydata(np.zeros_like(x_plot))
        ax1.view_init(elev=30, azim=45)  # Set initial view angle
        return line, point, line_pred
    
    def animate(i):
        # Loss landscape trajectory
        x_data = trajectory_x[:i+1]
        y_data = trajectory_y[:i+1]
        z_data = trajectory_losses[:i+1]
        
        line.set_data(x_data, y_data)
        line.set_3d_properties(z_data)

        # Get the exact current position
        current_x = x_data[-1]
        current_y = y_data[-1]
        current_z = z_data[-1]  # This is the actual loss value at this point
        
        # Position the point at the current position
        point._offsets3d = ([current_x], [current_y], [current_z])

        # Show stem line for current point
        for j, stem in enumerate(stem_lines):
            stem.set_visible(j == i)
        
        # Function prediction
        line_pred.set_ydata(predictions[i])

        ax1.view_init(elev=30, azim=45+i/2)  # Set view angle for the loss landscape
        
        # Update overall title
        fig.suptitle(f'{af_name} (size {nn_size}) learning {tf_name} function - Epoch {i*interval}', fontsize=16)
        
        return line, point, line_pred
    
    ani = FuncAnimation(fig, animate, frames=len(predictions), 
                        init_func=init, interval=100, blit=True)
    
    # Add tight layout to ensure subplots are well-spaced
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to make room for suptitle
    
    # Save animation
    ani.save(save_path, writer='pillow', fps=10)
    plt.close(fig)


if __name__ == "__main__":
    # Create the plots directory if it doesn't exist
    import os
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    generate_plot()