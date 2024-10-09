from modelBuilder import Config, init_my_model, parse_args
from trainTest import train_my_model, test_my_model, class_predict
from modelBuilder import parse_args

if __name__ == "__main__":
    
    args = parse_args()
    # Load configuration from YAML file
    config = Config(args.config)
    # Update config with command-line arguments if provided
    config.update_from_args(args)

    
    model = init_my_model(config)
    if config.SOLVER['mode'] == "train":
        train_my_model(config, model)

    elif config.SOLVER['mode'] == "test":
        test_my_model(config, model)

    elif config.SOLVER['mode'] == "predict":
        class_predict(config, model)
    else:
        print("Invalid mode")
