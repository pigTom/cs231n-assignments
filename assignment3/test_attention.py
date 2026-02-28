import torch
import numpy as np

from cs231n.transformer_layers import MultiHeadAttention

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

torch.manual_seed(231)

# Choose dimensions such that they are all unique for easier debugging:
# Specifically, the following values correspond to N=1, H=2, T=3, E//H=4, and E=8.
batch_size = 1
sequence_length = 3
embed_dim = 8
attn = MultiHeadAttention(embed_dim, num_heads=2)
attn.eval()

# Self-attention.
data = torch.randn(batch_size, sequence_length, embed_dim)
self_attn_output = attn(query=data, key=data, value=data)

# Masked self-attention.
mask = torch.randn(sequence_length, sequence_length) < 0.5
masked_self_attn_output = attn(query=data, key=data, value=data, attn_mask=mask)

# Attention using two inputs.
other_data = torch.randn(batch_size, sequence_length, embed_dim)
attn_output = attn(query=data, key=other_data, value=other_data)

expected_self_attn_output = np.asarray([[
 [-0.2494,  0.1396,  0.4323, -0.2411, -0.1547,  0.2329, -0.1936,
           -0.1444],
          [-0.1997,  0.1746,  0.7377, -0.3549, -0.2657,  0.2693, -0.2541,
           -0.2476],
          [-0.0625,  0.1503,  0.7572, -0.3974, -0.1681,  0.2168, -0.2478,
           -0.3038]]])

expected_masked_self_attn_output = np.asarray([[
 [-0.1347,  0.1934,  0.8628, -0.4903, -0.2614,  0.2798, -0.2586,
           -0.3019],
          [-0.1013,  0.3111,  0.5783, -0.3248, -0.3842,  0.1482, -0.3628,
           -0.1496],
          [-0.2071,  0.1669,  0.7097, -0.3152, -0.3136,  0.2520, -0.2774,
           -0.2208]]])

expected_attn_output = np.asarray([[
 [-0.1980,  0.4083,  0.1968, -0.3477,  0.0321,  0.4258, -0.8972,
           -0.2744],
          [-0.1603,  0.4155,  0.2295, -0.3485, -0.0341,  0.3929, -0.8248,
           -0.2767],
          [-0.0908,  0.4113,  0.3017, -0.3539, -0.1020,  0.3784, -0.7189,
           -0.2912]]])

print('self_attn_output error: ', rel_error(expected_self_attn_output, self_attn_output.detach().numpy()))
print('masked_self_attn_output error: ', rel_error(expected_masked_self_attn_output, masked_self_attn_output.detach().numpy()))
print('attn_output error: ', rel_error(expected_attn_output, attn_output.detach().numpy()))

# Check if errors are within tolerance
self_attn_error = rel_error(expected_self_attn_output, self_attn_output.detach().numpy())
masked_self_attn_error = rel_error(expected_masked_self_attn_output, masked_self_attn_output.detach().numpy())
attn_error = rel_error(expected_attn_output, attn_output.detach().numpy())

print(f'\nSelf-attention error: {self_attn_error}')
print(f'Masked self-attention error: {masked_self_attn_error}')
print(f'Attention error: {attn_error}')

if self_attn_error < 1e-4 and masked_self_attn_error < 1e-4 and attn_error < 1e-4:
    print('\nAll tests passed!')
else:
    print('\nSome tests failed!')