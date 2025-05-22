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
    def __init__(self, activation_function: type = nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 8),
            activation_function(),
            nn.Linear(8, 8),
            activation_function(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)
    

TARGET_FUNCTIONS = {
    "linear": lambda x: x,
    "quadratic": lambda x: x ** 2,
    "sine": lambda x: torch.sin(2 * np.pi * x),
    "exp": lambda x: torch.exp(x),
    "log": lambda x: torch.log(x + 1e-5),  # Avoid log(0)
    "modulated_blancmange": ModulatedBlancmange(),
    "modified_weierstrass": ModifiedWeierstrass(),
}

ACTIVATION_FUNCTIONS = {
    'ReLU': nn.ReLU,
    'Tanh': nn.Tanh,
    'Sigmoid': nn.Sigmoid,
    'LeakyReLU': nn.LeakyReLU,
    'ELU': nn.ELU,
    'SELU': nn.SELU,
    'GELU': nn.GELU,
    'Softplus': nn.Softplus,
    'modulated_blancmange': ModulatedBlancmange,
    'modified_weierstrass': ModifiedWeierstrass,
}

    


def set_seed(seed):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

def generate_plot():
    for tf_name, target_funciton in tqdm(TARGET_FUNCTIONS.items()):
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

            # Train the network
            EPOCHS = 1000
            INTERVAL = 10  # Store every 10 EPOCHS
            for epoch in range(EPOCHS):
                optimizer.zero_grad()
                outputs = net(x_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
                if epoch % INTERVAL == 0 or epoch == EPOCHS - 1:
                    with torch.no_grad():
                        pred = net(x_tensor).detach().numpy()
                        predictions.append(pred)

            # Animation
            fig, ax = plt.subplots()
            line_true, = ax.plot(x, y, label=f'{tf_name} function')
            line_pred, = ax.plot(x, predictions[0], label=f'{af_name} prediction', color='orange')
            ax.set_ylim(-1.2, 1.2)
            ax.legend()

            def animate(i):
                line_pred.set_ydata(predictions[i])
                ax.set_title(f'Epoch {i*INTERVAL}')
                return line_pred,

            ani = FuncAnimation(fig, animate, frames=len(predictions), interval=50, blit=True)
            ani.save(f'plots/{tf_name}/{af_name}_{tf_name}.gif', writer='pillow', fps=20)
            print(f"Animation saved as '{af_name}_{tf_name}.gif'")
            plt.close(fig)


if __name__ == "__main__":
    # Create the plots directory if it doesn't exist
    import os
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    generate_plot()