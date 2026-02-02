from thesis_project.models.keyword_spotting import KWSBase, KWSDynamic


def compute_macs_base_model(base_model: KWSBase, time_steps: int, rank_pr_stack: list) -> int:
    """
    Compute MACs for the base model given time steps and ranks per stack.

    Args:
        base_model: The base model instance
        time_steps: Number of time steps in the input
        rank_pr_stack: List of ranks for each stack in the backbone

    Returns:
        Total MACs for the base model
    """

    assert len(rank_pr_stack) == len(base_model.backbone), \
    "Length of rank_pr_stack must match number of stacks in the backbone."

    # total MACs across all stacks = (MACs 1 rank layer) * 2 layers per block * n_blocks_pr_stack * sum of ranks across stacks
    macs = time_steps * (base_model.cfg.backbone.n_channels_ext + base_model.cfg.backbone.n_channels_int)
    macs *= 2 * base_model.cfg.backbone.n_blocks_pr_stack
    macs *= sum(rank_pr_stack)
    return int(macs)

    # # loop version for reference and clarity, DO NOT DELETE
    # macs_base_model = 0
    # for rank in rank_pr_stack:
    #     single_layer_macs = time_steps * rank * (base_model.cfg.backbone.n_channels_ext + base_model.cfg.backbone.n_channels_int)
    #     single_block_macs = single_layer_macs * 2  # two layers per block
    #     single_stack_macs = single_block_macs * base_model.cfg.backbone.n_blocks_pr_stack
    #     macs_base_model += single_stack_macs
    # return macs_base_model

def compute_avg_rank_from_macs(base_model: KWSBase, time_steps: int, macs: int, num_stacks: int) -> float:
    """
    Compute the average rank needed per stack to achieve target MACs.

    Args:
        base_model: The base model instance
        time_steps: Number of time steps in the input
        macs: Target number of MACs
        num_stacks: Number of stacks in the backbone
    
    Returns:
        Average rank needed per stack
    """    
    # MACs = num_stacks * time_steps * avg_rank * (n_channels_ext + n_channels_int) * 2 * n_blocks_pr_stack
    # Solving for avg_rank:
    avg_rank = macs / (num_stacks * time_steps * (base_model.cfg.backbone.n_channels_ext + base_model.cfg.backbone.n_channels_int) * 2 * base_model.cfg.backbone.n_blocks_pr_stack)
    
    return avg_rank
