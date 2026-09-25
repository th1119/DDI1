import os
import logging
import json
import torch


def initialize_experiment(params, file_name=None):
    """Initialize experiment directory, logger, and parameter archive."""
    params.main_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    global_exps_dir = os.path.join(params.main_dir, 'experiments')
    os.makedirs(global_exps_dir, exist_ok=True)

    params.exp_dir = os.path.join(global_exps_dir, params.experiment_name)
    os.makedirs(params.exp_dir, exist_ok=True)

    if getattr(params, 'rank', 0) == 0:
        log_path = os.path.join(params.exp_dir, "log_train.txt")
        file_handler = logging.FileHandler(log_path)
        logger = logging.getLogger()

        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

        logger.info('============ Initialized logger ============')
        log_params = {k: v for k, v in vars(params).items() if not k.startswith('_') and not callable(v)}
        logger.info('\t '.join('%s: %s' % (k, str(v)) for k, v in sorted(log_params.items())))
        logger.info('============================================')

        with open(os.path.join(params.exp_dir, "params.json"), 'w') as fout:
            serializable_params = {}
            for k, v in vars(params).items():
                try:
                    json.dumps(v)
                    serializable_params[k] = v
                except TypeError:
                    serializable_params[k] = str(v)
            json.dump(serializable_params, fout, indent=4)


def initialize_model(params, model):
    """
    Model initialization / checkpoint loading (adapted for state_dict saving convention).
    """
    model_filename = f'best_graph_classifier_{params.iFold}.pth'
    model_path = os.path.join(params.exp_dir, model_filename)

    # Always build model skeleton first (ensures structure matches current architecture)
    classifier = model(params).to(device=params.device)

    # Attempt to load weights
    if params.load_model and os.path.exists(model_path):
        logging.info(f'Loading existing model state_dict from {model_path}')
        try:
            checkpoint = torch.load(model_path, map_location=params.device)

            # Compatible with multiple checkpoint formats
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
            elif hasattr(checkpoint, 'state_dict'):
                state_dict = checkpoint.state_dict()
            else:
                state_dict = checkpoint

            # Filter shape-mismatched parameters, load only fully matching entries
            model_dict = classifier.state_dict()
            filtered_state_dict = {}
            skipped_keys = []
            for k, v in state_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    filtered_state_dict[k] = v
                else:
                    skipped_keys.append(k)
            if skipped_keys:
                logging.info(f'Skipped {len(skipped_keys)} keys due to shape mismatch: {skipped_keys[:10]}...')
            classifier.load_state_dict(filtered_state_dict, strict=False)
            logging.info('Model weights loaded successfully (shape-matched).')
        except Exception as e:
            logging.warning(f'Checkpoint load failed: {e}. Using freshly initialized model.')
            classifier = model(params).to(device=params.device)
    else:
        logging.info('No existing model found. Initializing new model..')

    return classifier
