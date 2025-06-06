# Run main functions from command line
# First, navigate to folder and activate virtual environment
# Mac/Linux: source venv/bin/activate
# Windows: venv\Scripts\activate
#
# Usage: python run.py <function> <args>
#
# Functions:
#   - reflex: Train model for reflex prediction task
#   - proto: Train model for proto-language prediction task
#
# Args:
#   - feat_vec_dim: dimension of feature vectors
#   - hidden_dim: hidden dimension of model
#   - nhead: number of attention heads
#   - nlayers_enc: number of encoder layers
#   - nlayers_dec: number of decoder layers
#   - dropout: dropout rate
#   - learning_rate: learning rate
#   - n_epochs: number of epochs
#   - num_steps: number of steps
#   - batch_size: batch size
#   - missing_prop: proportion of missing data (e.g. 0.50 or 0.50,0.30)
#   - pretrain: whether to pretrain or load existing model
#   - fine_tune: fine-tune on individual language families
#   - fine_tune_epochs: number of fine-tuning epochs
#   - fine_tune_steps: number of fine-tuning steps
#   - fine_tune_batch_size: fine-tuning batch size
#   - fine_tune_lr: fine-tuning learning rate
#   - model_suffix: suffix for model name
#   - print_loss: whether to print loss during training (quickly fills up the screen)
#
# Additional args for proto prediction:
#   - n_folds: number of cross-validation folds
#   - fine_tune_batch_size_80: fine-tuning batch size for 80% data
#   - force_pretrain: whether to force pretraining even if model exists
#   - use_pretrain_data: whether to use pretrained model (or pretrain it) or only train on task-specific data
#   - zero_shot: whether to use zero-shot learning (no fine-tuning on the proto-languages)

import sys
import os

# Map of datasets to language families for proto-language reconstruction
proto_map = {'Burmish':'ProtoBurmish', 'Purus':'ProtoPurus',
                'Lalo':'ProtoLalo', 'Bai':'ProtoBai', 'Karen':'ProtoKaren',
                'Romance':'Latin'}

# List of datasets for fine-tuning reflex prediction model
finetune_language_families = os.listdir("data/reflex_data/data-surprise")

# Default args for reflex prediction task
reflex_args = {
    # model params
    "feat_vec_dim": 39,
    "hidden_dim": 256,
    "nhead": 4,
    "nlayers_enc": 1,
    "nlayers_dec": 2,
    "dropout": 0.15,
    # training params
    "num_steps": 500,
    "n_epochs": 32,
    "batch_size": 48,
    "learning_rate": 5e-4,
    "pretrain": True,
    "fine_tune": True,
    "fine_tune_epochs": 8,
    "fine_tune_steps": 100,
    "fine_tune_batch_size": 48,
    "fine_tune_lr": 1e-4,
    "missing_prop": None,
    "model_suffix": "",
    "print_loss": False
}

proto_args = {
    # Model params
    "feat_vec_dim": 39,
    "hidden_dim": 128,
    "nhead": 4,
    "nlayers_enc": 1,
    "nlayers_dec": 2,
    "dropout": 0.15,
    # training params
    "num_steps": 500,
    "n_epochs": 32,
    "batch_size": 48,
    "learning_rate": 1e-3,
    "use_pretrain_data": True,
    "force_pretrain": False,
    "fine_tune_epochs": 40,
    "fine_tune_lr": 5e-4,
    "fine_tune_batch_size": 48,
    "fine_tune_batch_size_80": 16,
    "missing_prop": None,
    "n_folds": 10,
    "model_suffix": "",
    "zero_shot": False,
    "print_loss": False
    }


def print_help():
    """Prints help message."""
    print("\nUsage: python run.py <function> [arguments]\n")
    print("Available functions:")
    print(" - reflex:   Train the reflex prediction model\n")
    print(" - proto:    Train the proto-language reconstruction model\n")
    print("Arguments (reflex):")
    for key, value in reflex_args.items():
        print(f"  {key} (default: {value})")
    print("\nExample:")
    print("  python run.py reflex nlayers_enc=2 nlayers_dec=3\n")
    print("\n\n")
    print("Arguments (proto):")
    for key, value in proto_args.items():
        print(f"  {key} (default: {value})")
    print("\n\n")
    print("Note:")
    print("You will find the variable 'proto_map' in the script, which maps proto-languages to language families.")
    print("Please do not forget to update this variable if you want to train the model for different proto-languages.\n")
    print("Value None for missing_prop means all proportions (0.10, 0.30, 0.50).")
    sys.exit(0)

# Run main functions from command line
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print_help()
    from code.training import *
    if sys.argv[1] == "reflex":
        # Update args if provided
        for arg in sys.argv[2:]:
            key, value = arg.split('=')
            if key == 'missing_prop':
                reflex_args[key] = value.split(",")
            elif type(reflex_args[key]) == bool:
                reflex_args[key] = value.lower() == 'true'
            else:
                reflex_args[key] = type(reflex_args[key])(value)
        train_reflex(reflex_args, finetune_language_families)
    elif sys.argv[1] == "proto":
        # Update args if provided
        for arg in sys.argv[2:]:
            key, value = arg.split('=')
            if key == 'missing_prop':
                proto_args[key] = value.split(",")
            elif type(proto_args[key]) == bool:
                proto_args[key] = value.lower() == 'true'
            else:
                proto_args[key] = type(proto_args[key])(value)
        train_proto(proto_args, proto_map)
    else:
        print("Function not found.")
        print_help()
