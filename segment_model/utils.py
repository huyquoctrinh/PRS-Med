import torch
def create_attention_mask(input_ids):
    """
    Create attention mask for the input_ids.
    """
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    attention_mask[input_ids == -100] = 0
    return attention_mask
