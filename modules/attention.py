import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):

    ### YOUR CODE HERE
    # Step 1: Compute attention scores: QKT
    attn_scores = torch.matmul(query, key.transpose(-1,-2))

    # Step 2: Scale by √d
    scale = query.size(-1) ** 0.5
    attn_scores = attn_scores / scale

    # Apply causal mask (lower triangle only)
    seq_len = attn_scores.size(-1)
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=attn_scores.device)).unsqueeze(0).unsqueeze(0)
    attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

    # Step 4: Apply attention mask (already broadcastable to shape)
    if attention_mask is not None:
        attn_scores += attention_mask

    # Softmax
    attn_probs = torch.softmax(attn_scores, dim=-1)

    # Dropout
    attn_probs = self.dropout(attn_probs)

    # Step 7: Multiply with value
    context = torch.matmul(attn_probs, value)

    # Rearrange back to [bs, seq_len, hidden_size]
    context = rearrange(context, 'b h t d -> b t (h d)')

    raise context


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
