import argparse
import itertools
from pathlib import Path
import json
from typing import List, Optional
from datetime import datetime
import sys

# Add parent directory to path to import thesis_project
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from thesis_project.models.keyword_spotting.base_schema import KeyWordSpottingBaseConfig
from thesis_project.models.keyword_spotting.dynamic_schema import (
    DynamicTrainingConfig
)
from thesis_project.training.key_word_spotting import setup_and_train_dynamic_model
from thesis_project.training.key_word_spotting.configs.sweep_standard_config import get_sweep_base_config
from thesis_project.training.key_word_spotting.loss_functions import (
    BatchMeanMSELoss,
    PerSampleMSELoss,
    AvgRankLoss,
    OneSidedMSELoss,
    CEWeightedRankLoss,
    AsymmetricMSELoss,
    CEGatedLoss,
)


def create_sweep_configs(
    loss_types: List[str],
    rank_weights: List[float],
    rank_targets: Optional[List[float]] = None,
    pool_reduction_factors: List[int] = [2],
):
    """
    Create all combinations of hyperparameters for sweep.
    Smart handling: losses without REQUIRES_TARGET skip rank_targets loop.
    
    Args:
        loss_types: List of loss modes (e.g., ['avg_rank'])
        rank_weights: List of rank loss weights (e.g., [10, 20, 30])
        rank_targets: List of target ranks for base_model_rank_reference (e.g., [10, 20, 30])
        pool_reduction_factors: List of pool reduction factors (e.g., [2, 4, 8])
    """
    
    # Map loss names to classes to check REQUIRES_TARGET
    loss_class_map = {
        "batch_mean_mse": BatchMeanMSELoss,
        "per_sample_mse": PerSampleMSELoss,
        "avg_rank": AvgRankLoss,
        "one_sided_mse": OneSidedMSELoss,
        "ce_weighted_rank": CEWeightedRankLoss,
        "asymmetric_mse": AsymmetricMSELoss,
        "ce_gated": CEGatedLoss,
    }
    
    # Router modes: global (1 output) or per_stack (3 outputs)
    router_modes = ['global', 'per_stack']
    
    configs = []
    for loss_type in loss_types:
        loss_class = loss_class_map.get(loss_type)
        requires_target = getattr(loss_class, 'REQUIRES_TARGET', False) if loss_class else False
        
        for rank_weight in rank_weights:
            for pool_reduction_factor in pool_reduction_factors:
                # If loss doesn't require target, only create one config per router mode
                if not requires_target:
                    for router_mode in router_modes:
                        use_global_rank = router_mode == 'global'
                        num_rank_outputs = 1 if use_global_rank else 3
                        
                        config = {
                            'loss_type': loss_type,
                            'rank_weight': rank_weight,
                            'rank_target': 15.0,  # Set a default value (won't be used by loss)
                            'router_mode': router_mode,
                            'use_global_rank': use_global_rank,
                            'num_rank_outputs': num_rank_outputs,
                            'pool_reduction_factor': pool_reduction_factor,
                        }
                        configs.append(config)
                else:
                    # Loss requires target, loop over all rank_targets
                    for rank_target in rank_targets:
                        for router_mode in router_modes:
                            use_global_rank = router_mode == 'global'
                            num_rank_outputs = 1 if use_global_rank else 3
                            
                            config = {
                                'loss_type': loss_type,
                                'rank_weight': rank_weight,
                                'rank_target': rank_target,
                                'router_mode': router_mode,
                                'use_global_rank': use_global_rank,
                                'num_rank_outputs': num_rank_outputs,
                                'pool_reduction_factor': pool_reduction_factor,
                            }
                            configs.append(config)
    
    return configs


def run_experiment(config: dict, sweep_dir: Path, exp_number: int, base_config: dict):
    """Run a single experiment with given config."""
    print(f"\n{'='*80}")
    print(f"Experiment {exp_number}: {config}")
    print(f"{'='*80}\n")
    
    try:
        # Override sweep parameters in the base config
        base_config['router'].use_global_rank = config['use_global_rank']
        base_config['router'].num_rank_outputs = config['num_rank_outputs']
        base_config['router'].pool_reduction_factor = config['pool_reduction_factor']
        base_config['loss'].rank_loss_weight = config['rank_weight']
        base_config['loss'].rank_loss_mode = config['loss_type']
        base_config['loss'].base_model_rank_reference = config['rank_target']
        
        # Create full config
        full_config = DynamicTrainingConfig(
            seed=base_config['seed'],
            router=base_config['router'],
            dynamic_model=base_config['dynamic_model'],
            data_loader=base_config['data_loader'],
            loss=base_config['loss'],
            training=base_config['training'],
            noise_train=base_config['noise_train'],
            noise_eval=base_config['noise_eval'],
        )
        
        # Load base model config
        BASE_MODEL_DIR = Path(full_config.dynamic_model.base_model_dir)
        base_model_cfg = KeyWordSpottingBaseConfig(**json.loads((BASE_MODEL_DIR / "base_config.json").read_text()))
        
        # Create run directory within sweep folder
        run_name = f"exp_{exp_number:03d}"
        run_dir = sweep_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save sweep-specific config
        sweep_config_path = run_dir / "sweep_config.json"
        with open(sweep_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Run training pipeline (this handles everything else)
        model, epoch_logs, step_logs = setup_and_train_dynamic_model(
            full_config=full_config,
            base_model_cfg=base_model_cfg,
            run_dir=run_dir,
        )
        
        print(f"\nExperiment {exp_number} completed successfully!")
        
        # Extract final metrics
        final_metrics = {}
        if epoch_logs is not None and len(epoch_logs) > 0:
            last_epoch = epoch_logs[-1]
            final_metrics = {
                'train_loss': last_epoch.get('train_loss'),
                'train_acc': last_epoch.get('train_acc'),
                'best_train_loss': last_epoch.get('best_train_loss'),
                'best_train_acc': last_epoch.get('best_train_acc'),
                'train_expected_rank': last_epoch.get('train_expected_rank'),
                'val_loss': last_epoch.get('val_loss'),
                'val_acc': last_epoch.get('val_acc'),
                'best_val_loss': last_epoch.get('best_val_loss'),
                'best_val_acc': last_epoch.get('best_val_acc'),
                'val_expected_rank': last_epoch.get('val_expected_rank'),
            }
            print(f"Final metrics: {final_metrics}")
        
        return True, final_metrics
        
    except Exception as e:
        print(f"Experiment {exp_number} failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False, {}


def main():
    parser = argparse.ArgumentParser(
        description='Sweep hyperparameters for dynamic routing model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic sweep with avg_rank loss
  python sweep_dynamic.py --loss-types avg_rank --rank-weights 10 20 30
  
  # Custom rank targets
  python sweep_dynamic.py --loss-types avg_rank --rank-weights 20 --rank-targets 10 15 20
  
  # Try different loss with multiple weights
  python sweep_dynamic.py --loss-types asymmetric_mse --rank-weights 5 10 15 20
  
  # Sweep over multiple pool reduction factors
  python sweep_dynamic.py --loss-types avg_rank --rank-weights 10 --pool-reduction-factor 2 4 8
  
  # Use custom config file
  python sweep_dynamic.py --config my_config.json --loss-types avg_rank --rank-weights 10
  
  Note: Custom config files should be Python modules with get_sweep_base_config() function
        """
    )
    
    parser.add_argument(
        '--loss-types',
        nargs='+',
        required=True,
        choices=['avg_rank', 'batch_mean_mse', 'per_sample_mse', 'ce_weighted_rank', 'asymmetric_mse', 'one_sided_mse', 'ce_gated'],
        help='Loss types to sweep over'
    )
    parser.add_argument(
        '--rank-weights',
        nargs='+',
        type=float,
        default=[1.0, 5.0, 10.0, 15.0, 20.0],
        help='Rank loss weights to test (default: [1, 5, 10, 15, 20])'
    )
    parser.add_argument(
        '--rank-targets',
        nargs='+',
        type=float,
        default=[10.0, 20.0, 30.0],
        help='Target ranks for base_model_rank_reference (default: [10.0, 20.0, 30.0])'
    )
    parser.add_argument(
        '--pool-reduction-factor',
        nargs='+',
        type=int,
        default=[2],
        help='Pool reduction factor(s) for router. Can specify multiple values to sweep over. Set to 1 for no pooling (default: [2])'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to custom config Python module (default: uses sweep_standard_config.py)'
    )
    parser.add_argument(
        '--sweep-name',
        type=str,
        default=None,
        help='Name for this sweep (default: timestamp)'
    )
    parser.add_argument(
        '--use-subset',
        action='store_true',
        help='Use subset of data for quick testing (default: False)'
    )
    parser.add_argument(
        '--subset-size',
        type=int,
        default=1000,
        help='Size of data subset when --use-subset is enabled (default: 1000)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print configs without running experiments'
    )
    
    args = parser.parse_args()
    
    # Load base config
    if args.config:
        # Import custom config module
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_config", args.config)
        custom_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_config)
        base_config = custom_config.get_sweep_base_config(
            use_subset=args.use_subset,
            subset_size=args.subset_size
        )
        print(f"Loading custom config from: {args.config}")
    else:
        # Use default config
        base_config = get_sweep_base_config(
            use_subset=args.use_subset,
            subset_size=args.subset_size
        )
        print("Using default sweep_standard_config")
    
    # Generate all config combinations
    configs = create_sweep_configs(
        loss_types=args.loss_types,
        rank_weights=args.rank_weights,
        rank_targets=args.rank_targets,
        pool_reduction_factors=args.pool_reduction_factor,
    )
    
    print(f"\nGenerated {len(configs)} experiment configurations")
    
    if args.use_subset:
        print(f"Running in SUBSET MODE with {args.subset_size} samples per experiment")
    
    # Create sweep directory
    if args.sweep_name:
        sweep_name = args.sweep_name
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        sweep_name = f"sweep_{timestamp}"
    
    sweep_dir = Path("model_runs/dynamic") / sweep_name
    
    if args.dry_run:
        print(f"\nDRY RUN - Would create sweep directory: {sweep_dir}")
        print(f"\nConfigs that would be run:")
        for i, config in enumerate(configs, 1):
            print(f"\n{i}. {json.dumps(config, indent=2)}")
        return
    
    sweep_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSweep directory: {sweep_dir}")
    
    # Save sweep metadata (convert Pydantic models to dicts)
    base_config_serializable = {
        'seed': base_config['seed'],
        'router': base_config['router'].model_dump(),
        'dynamic_model': base_config['dynamic_model'].model_dump(),
        'data_loader': base_config['data_loader'].model_dump(),
        'loss': base_config['loss'].model_dump(),
        'training': base_config['training'].model_dump(),
        'noise_train': base_config['noise_train'].model_dump(),
        'noise_eval': base_config['noise_eval'].model_dump(),
    }
    
    sweep_metadata = {
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'base_config': base_config_serializable,
        'num_experiments': len(configs),
        'experiment_configs': configs,
    }
    with open(sweep_dir / "sweep_metadata.json", 'w') as f:
        json.dump(sweep_metadata, f, indent=2)
    
    # Run experiments
    results = []
    for i, config in enumerate(configs, 1):
        success, metrics = run_experiment(config, sweep_dir, i, base_config)
        results.append({
            'exp_number': i,
            'config': config,
            'success': success,
            'metrics': metrics
        })
    
    # Save results summary
    with open(sweep_dir / "results_summary.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SWEEP SUMMARY")
    print(f"{'='*80}")
    successful = sum(1 for r in results if r['success'])
    print(f"Completed: {successful}/{len(results)} experiments")
    print(f"Results saved to: {sweep_dir}")
    
    if successful < len(results):
        print("\nFailed experiments:")
        for r in results:
            if not r['success']:
                print(f"  - Exp {r['exp_number']}: {r['config']}")


if __name__ == "__main__":
    main()
