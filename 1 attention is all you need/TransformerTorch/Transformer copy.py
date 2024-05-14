import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

# MultiHeadAttention is a core component of the Transformer architecture.
# It allows the model to jointly attend to information from different representation subspaces at different positions.
# -- MultiHeadAttention enables the model to process parts of the input data in parallel and differently, improving efficiency and effectiveness.
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure the model dimension is divisible by the number of heads to equally distribute the data.
        # -- The division ensures that each head can equally share the model's dimension, avoiding dimension mismatch errors.
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model  # Total dimension of the model
        self.num_heads = num_heads  # Number of attention heads
        self.d_k = d_model // num_heads  # Dimension of each head

        # Linear transformations for queries, keys, and values
        # -- These transformations project the input into different subspaces for queries, keys, and values.
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        # Final linear transformation after concatenating attention heads
        # -- This layer combines the outputs from all heads back into a single matrix.
        self.W_o = nn.Linear(d_model, d_model)

        # Choose the appropriate device (GPU or CPU)
        # -- This ensures that the model uses GPU if available for faster computation.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Scaled dot-product attention mechanism
    # -- This function calculates attention scores and applies them to the value vectors.
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Compute the dot products of the query with all keys, divide each by sqrt(d_k), and apply a softmax function
        # to obtain the weights on the values.
        # -- The division by sqrt(d_k) helps in stabilizing gradients during training.
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # Apply masking if provided (for padded elements and future tokens)
        # -- Masking ensures that the model does not consider certain positions in the input, typically for padding or future tokens.
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        # Multiply the weights by the values to get the output of the attention mechanism.
        # -- This step aggregates the information from different positions based on the computed attention scores.
        output = torch.matmul(attn_probs, V)
        return output
    
    # Split the embeddings into multiple heads for multi-head attention
    # -- This function reshapes the input tensor to separate heads, allowing parallel processing.
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        # Reshape into (batch_size, num_heads, seq_length, d_k)
        # -- The reshaping facilitates separate attention computations for each head.
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    # Combine the multi-head attention outputs back into a single matrix
    # -- This function reverses the operation of split_heads, combining the separate head outputs.
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        # Reshape back to (batch_size, seq_length, d_model)
        # -- The output is reshaped back to the original embedding dimension.
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    # Forward pass of the multi-head attention layer
    # -- This function defines the operations for processing inputs through the multi-head attention mechanism.
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split into heads
        # -- Linear transformations are first applied to queries, keys, and values, followed by splitting into multiple heads.
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Apply scaled dot-product attention to each head
        # -- Each head independently computes scaled dot-product attention.
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        # Concatenate heads and apply final linear layer
        # -- The outputs from all heads are concatenated and passed through a final linear transformation.
        output = self.W_o(self.combine_heads(attn_output))
        return output

# Position-wise feed-forward network used in each encoder and decoder layer
# -- This network applies two linear transformations with a ReLU activation in between, processing each position identically and independently.
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        # First linear transformation increases dimension from d_model to d_ff
        # -- This expansion layer increases the dimensionality to introduce non-linearity and aid in learning complex patterns.
        self.fc1 = nn.Linear(d_model, d_ff)
        # ReLU activation function
        # -- ReLU helps introduce non-linearity into the model and speeds up training.
        self.relu = nn.ReLU()
        # Second linear transformation reduces dimension back to d_model
        # -- This layer reduces the dimensionality back to that of the model, aligning with the Transformer's architecture.
        self.fc2 = nn.Linear(d_ff, d_model)

    # Forward pass of the feed-forward network
    # -- This function defines how the input data flows through the feed-forward network.
    def forward(self, x):
        # Apply first linear transformation, ReLU, then second linear transformation
        # -- The input data is transformed, activated, and then transformed again in sequence.
        return self.fc2(self.relu(self.fc1(x)))

# Positional encoding adds information about the relative or absolute position of the tokens in the sequence.
# The positional encodings have the same dimension as the embeddings, so that the two can be summed.
# -- Positional encodings are crucial for the model to understand the order of tokens, as the Transformer architecture itself does not have recurrence or convolution.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix of zeros with shape (max_seq_length, d_model)
        # -- This matrix will hold the positional encodings for all positions up to max_seq_length.
        pe = torch.zeros(max_seq_length, d_model)
        # Create a tensor representing the position indices (0 to max_seq_length-1)
        # -- This tensor helps in calculating the positional encoding for each position.
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        # Compute the positional encoding values using sine and cosine functions
        # -- Sine and cosine functions are used to generate positional encodings, providing a unique signal for each position.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register pe as a buffer that is not a model parameter
        # -- This registration makes 'pe' a persistent state of the module for use in training and inference.
        self.register_buffer('pe', pe.unsqueeze(0))
        
    # Forward pass adds positional encoding to input embeddings
    # -- This function defines how positional encodings are added to embeddings, enhancing them with positional information.
    def forward(self, x):
        # Add positional encoding to each input embedding
        # -- The addition of positional encodings to embeddings allows the model to use position information.
        return x + self.pe[:, :x.size(1)]

# Encoder layer consists of multi-head attention and position-wise feed-forward network
# -- Each encoder layer processes the input sequence using self-attention and then passes it through a feed-forward network.
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        # Multi-head attention layer
        # -- This layer allows the encoder to focus on different parts of the input sequence independently.
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # Position-wise feed-forward network
        # -- This network processes each position in the sequence independently and identically.
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        # Layer normalization applied before the residual connection and after the dropout
        # -- Normalization helps stabilize the training process by normalizing the layer inputs.
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout layer applied to the output of sub-layers
        # -- Dropout is a regularization technique that helps prevent overfitting by randomly zeroing out some outputs.
        self.dropout = nn.Dropout(dropout)
        
    # Forward pass of the encoder layer
    # -- This function defines the sequence of operations for data passing through the encoder layer.
    def forward(self, x, mask):
        # Apply self-attention
        # -- Self-attention allows the layer to focus on different parts of the input for better understanding.
        attn_output = self.self_attn(x, x, x, mask)
        # Add residual connection and apply layer normalization
        # -- The residual connection helps in training deeper models by allowing gradients to flow through the network.
        x = self.norm1(x + self.dropout(attn_output))
        # Apply feed-forward network
        # -- The feed-forward network processes the data further after attention.
        ff_output = self.feed_forward(x)
        # Add residual connection and apply layer normalization
        # -- Another residual connection and normalization step to stabilize and improve training.
        x = self.norm2(x + self.dropout(ff_output))
        return x

# Decoder layer consists of two multi-head attention layers and one position-wise feed-forward network
# -- The decoder layer uses self-attention to process its input and cross-attention to focus on relevant parts of the encoder output.
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        # Self-attention layer for the decoder
        # -- Self-attention in the decoder helps it focus on different parts of its input.
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # Cross-attention layer where queries come from previous decoder layer, and keys and values come from the encoder
        # -- Cross-attention allows the decoder to focus on relevant parts of the encoder output.
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        # Position-wise feed-forward network
        # -- Similar to the encoder, this network processes the decoder's attention output further.
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        # Layer normalization applied before the residual connection and after the dropout
        # -- Normalization steps are crucial for maintaining stability in the model's training process.
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # Dropout layer applied to the output of sub-layers
        # -- Dropout helps in regularizing the model, preventing overfitting.
        self.dropout = nn.Dropout(dropout)
        
    # Forward pass of the decoder layer
    # -- This function defines the sequence of operations for data passing through the decoder layer.
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Apply self-attention
        # -- The decoder first processes its input using self-attention.
        attn_output = self.self_attn(x, x, x, tgt_mask)
        # Add residual connection and apply layer normalization
        # -- A residual connection is added to help gradients flow and stabilize training.
        x = self.norm1(x + self.dropout(attn_output))
        # Apply cross-attention
        # -- Cross-attention focuses on the encoder output, using it to inform the decoding process.
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        # Add residual connection and apply layer normalization
        # -- Another set of residual connection and normalization to process the cross-attention output.
        x = self.norm2(x + self.dropout(attn_output))
        # Apply feed-forward network
        # -- The feed-forward network processes the combined attention outputs further.
        ff_output = self.feed_forward(x)
        # Add residual connection and apply layer normalization
        # -- The final residual connection and normalization in the decoder layer.
        x = self.norm3(x + self.dropout(ff_output))
        return x

# Transformer model combining multiple encoder and decoder layers
# -- The Transformer model architecture combines these layers to process input sequences into output sequences.
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        # Embedding layers for source and target languages
        # -- Embeddings convert input token indices into vectors of continuous values, each representing a token's features.
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        # Positional encoding layer
        # -- Positional encodings provide a way for the model to incorporate information about the position of tokens in the sequence.
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # Stack of encoder layers
        # -- Multiple layers allow the model to learn complex patterns in the data.
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        # Stack of decoder layers
        # -- Each decoder layer processes the encoder's output and the previous decoder output to generate the next output token.
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        # Final linear layer that projects the output of the decoder to the target vocabulary size
        # -- This layer converts the decoder's output into probabilities over the target vocabulary.
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        # Dropout layer applied to embeddings
        # -- Applying dropout to embeddings can help in preventing overfitting on particular words.
        self.dropout = nn.Dropout(dropout)

        # Choose the appropriate device (GPU or CPU)
        # -- Using the right device can significantly speed up training times.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate masks for source and target sequences
    # -- Masks help the model ignore padding tokens and maintain causality in the decoder.
    def generate_mask(self, src, tgt):
        # Create a mask for the source sequence to ignore padding tokens (0)
        # -- Source mask prevents the model from processing padding tokens, which carry no meaningful information.
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        # Create a mask for the target sequence to ignore padding tokens and prevent attending to future tokens
        # -- Target mask ensures that during training, the decoder can only attend to past tokens, preserving causality.
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        # Create a mask to prevent positions from attending to subsequent positions (for causality)
        # -- The no-peek mask prevents the decoder from cheating by looking at future tokens.
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(self.device)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    # Forward pass of the Transformer model
    # -- This function orchestrates the entire sequence of processing the input through the model to produce the output.
    def forward(self, src, tgt):
        # Generate masks for source and target
        # -- Masks are generated to be applied during the attention computations.
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        # Apply embeddings and positional encoding to source and target
        # -- The input tokens are converted to embeddings, which are then enhanced with positional information.
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        # Pass through each encoder layer
        # -- The encoder processes the source input sequentially through each layer, refining the representation at each step.
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        # Pass through each decoder layer
        # -- The decoder refines its output at each layer, using both its previous outputs and the encoder's output.
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        # Apply final linear transformation
        # -- The final output of the decoder is transformed into a set of scores for each token in the target vocabulary.
        output = self.fc(dec_output)
        return output