import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional


class DQN(nn.Module):
    """
    Deep Q-Network for OAM handover decisions.
    
    Takes a state vector as input and outputs Q-values for each possible action.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: List[int] = [128, 128],
        activation: str = "relu"
    ):
        """
        Initialize the DQN model.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_layers: List of hidden layer sizes
            activation: Activation function to use ('relu', 'leaky_relu', 'tanh')
        """
        super(DQN, self).__init__()
        
        # Store activation type for weight initialization
        self.activation = activation
        
        # Input layer
        layers = []
        prev_dim = state_dim
        
        # Hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Add activation function
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.01))
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())  # Default to ReLU
                self.activation = "relu"  # Update activation for consistency
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        # Combine all layers
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        Initialize network weights.
        
        Args:
            module: PyTorch module to initialize
        """
        if isinstance(module, nn.Linear):
            # Use appropriate initialization based on activation function
            if self.activation == "relu":
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            elif self.activation == "leaky_relu":
                nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
            elif self.activation == "tanh":
                # For tanh, use Xavier/Glorot initialization
                nn.init.xavier_normal_(module.weight)
            else:
                # Default to ReLU initialization
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Q-values tensor of shape (batch_size, action_dim)
        """
        return self.model(state)
    
    def save(self, path: str) -> None:
        """
        Save model weights to file.
        
        Args:
            path: Path to save the model
        """
        torch.save(self.state_dict(), path)
    
    def load(self, path: str, device: torch.device) -> None:
        """
        Load model weights from file.
        
        Args:
            path: Path to load the model from
            device: Device to load the model to
        """
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
        self.eval() 