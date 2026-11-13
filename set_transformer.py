
import torch
import torch.nn as nn
import torch.nn.functional as F


# Multihead Attention Block (MAB)
# This is the foundation of the Set Transformer.
# It uses attention to determine how information from one group
# of vectors (K) should influence another group of vectors (Q).
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()

        # Output feature dimension and number of attention heads
        self.dim_V = dim_V
        self.num_heads = num_heads

        # Learned linear transformations for queries, keys, and values
        # All three are projected into the same output dimension
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)

        # Linear transformation applied after attention
        self.fc_o = nn.Linear(dim_V, dim_V)

        # Optional layer normalization for training stability
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)

    def forward(self, Q, K):
        # Project the inputs into queries, keys, and values
        # Q determines what information is being requested
        # K determines what information can be matched
        # V contains the information to be combined
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        # Divide the feature dimension among attention heads
        # Example: 128 features / 4 heads = 32 features per head
        dim_split = self.dim_V // self.num_heads

        # Split features into heads and concatenate along the
        # batch dimension so all heads can be processed together
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        # Calculate attention scores using query-key dot products
        # Scale by sqrt(dim_V), following the original implementation
        # Softmax converts scores into attention weights
        attention = torch.softmax(
            Q_.bmm(K_.transpose(1, 2)) / (self.dim_V ** 0.5),
            dim=2
        )

        # Multiply attention weights by values to combine information
        # Add the projected queries as a residual connection
        # Recombine all attention heads along the feature dimension
        O = torch.cat(
            (Q_ + attention.bmm(V_)).split(Q.size(0), 0),
            2
        )

        # Apply optional normalization
        if hasattr(self, 'ln0'):
            O = self.ln0(O)

        # Apply a learned transformation and ReLU activation
        # Add another residual connection
        O = O + F.relu(self.fc_o(O))

        # Apply optional final normalization
        if hasattr(self, 'ln1'):
            O = self.ln1(O)

        return O


# Set Attention Block (SAB)
# Allows every element in a set to attend to every other element.
# Uses MAB with the same set as both inputs.
# Preserves permutation equivariance.
class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super().__init__()

        # Self-attention: queries, keys, and values come from X
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        # Every element attends to all elements in the set
        return self.mab(X, X)


# Induced Set Attention Block (ISAB)
# Uses a smaller collection of learned inducing points to
# reduce the computational cost of attention.
# Full self-attention scales approximately as O(N^2).
# ISAB scales approximately as O(N*M), where M is the
# number of inducing points.
class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super().__init__()

        # Create trainable inducing points
        # Shape: (1, number of inducing points, output features)
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))

        # Initialize inducing points using Xavier initialization
        nn.init.xavier_uniform_(self.I)

        # First attention block: inducing points attend to input set
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)

        # Second attention block: input elements attend to
        # the representations produced by the inducing points
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        # Repeat inducing points across the batch
        # Each example uses the same learned inducing parameters
        # H summarizes information from the input set
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)

        # Each input element attends to the inducing-point summaries
        # Output preserves the number of elements in the input set
        return self.mab1(X, H)


# Pooling by Multihead Attention (PMA)
# Converts an unordered set into a fixed number of output vectors.
# Learned seed vectors attend to the encoded set.
# This operation is permutation-invariant with respect to X.
class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super().__init__()

        # Create trainable seed vectors
        # For max regression, num_seeds = 1
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))

        # Initialize the seed vectors
        nn.init.xavier_uniform_(self.S)

        # Seed vectors act as queries; encoded set provides keys/values
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        # Repeat the learned seeds across the batch
        # Pool information from the entire input set
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


# Complete Set Transformer
# Architecture follows the original Set Transformer implementation:
# Encoder: ISAB -> ISAB
# Decoder: PMA -> SAB -> SAB -> Linear
#
# Configured for maximum-value regression:
# Input: unordered set of scalar values
# Output: one predicted maximum per set
class SetTransformer(nn.Module):
    def __init__(self, dim_input=1, num_outputs=1, dim_output=1,
                 num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super().__init__()

        # Encoder processes relationships between set elements
        # First ISAB expands each scalar to 128 features
        # Second ISAB further processes those learned features
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        )

        # Decoder converts the encoded set into a prediction
        # PMA pools the set into one learned representation
        # Two SAB blocks further transform the pooled representation
        # Final linear layer maps 128 features to one scalar
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_output)
        )

    def forward(self, X):
        # Encode the unordered input set
        # Pool and transform the representations into predictions
        return self.dec(self.enc(X))
