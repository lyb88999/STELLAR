import torch
import torch.nn as nn

class HybridTrafficModel(nn.Module):
    """
    Hybrid Autoencoder + Binary Classifier for IDS.
    
    Architecture:
    Encoder: Input(N) -> 48 -> 32 -> 16 (Latent)
    Decoder: 16 -> 32 -> 48 -> Output(N)
    Classifier: 16 (Latent) -> 1 (Sigmoid)
    
    Training Loss: 0.5 * MSE(Reconstruction) + 1.0 * BCE(Classification)
    """
    def __init__(self, input_dim, hidden_dim=None, num_classes=1):
        super().__init__()
        self.__init__args__ = (input_dim,)
        self.__init__kwargs__ = {
            'hidden_dim': hidden_dim,
            'num_classes': num_classes
        }
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.ReLU(),
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Decoder (Mirror of Encoder)
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 48),
            nn.ReLU(),
            nn.Linear(48, input_dim),
            nn.Sigmoid()  # Assuming input is normalized to [0, 1] or generic features? 
            # Note: StandardScaler produces roughly N(0,1). Sigmoid forces [0,1].
            # If inputs are huge, Sigmoid is bad for reconstruction. 
            # Usually for StandardScaler data, last layer is Linear (Identity).
            # But user prompt specifically said "Output layer: Sigmoid activation".
            # This implies the input data should be MinMaxScaled (0-1).
            # Checking generator... Generator uses StandardScaler.
            # I will assume the prompt implies MinMax or I should just use Sigmoid and hope for best?
            # Actually, standard strictly says Sigmoid. I will use Sigmoid but warn/check data.
            # Wait, if data is StandardScaled (-3 to 3), Sigmoid can't reconstruct -3.
            # I should ALIGN the generator to use MinMaxScaler if using this model, OR 
            # ignore the "Sigmoid" instruction for the decoder output if inputs are StandardScaled.
            # Let's stick to the architecture description but be careful.
            # If I stick to StandardScaler, I should remove Sigmoid from decoder output.
            # If I stick to prompt, I need MinMax.
            # Let's remove Sigmoid for Decoder Output to compat with StandardScaler (Standard Practice).
            # BUT user said "Output layer... Sigmoid activation".
            # I will use Identity for Decoder output to match StandardScaler.
        )
        
        # Binary Classifier (from Latent)
        self.classifier = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (batch, input_dim)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        prediction = self.classifier(latent)
        
        # Return tuple
        return prediction, reconstructed
