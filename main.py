from tensorflow.keras.layers import *
import argparse

from unet import build_model, train_model, predict, save_predictions


def main(args):
    print("start main")
    model = get_model()
    print("model is got")

    if args.test:
        path_to_rgb = args.predict_path
        path_to_save = args.save_path

        path_to_checkpoint = args.model_path
        checkpoint_name = path_to_checkpoint + args.checkpoint_name
        model.load_weights(checkpoint_name)

        tifs = predict(model, path_to_rgb)
        save_predictions(tifs, path_to_save)

    elif args.train:
        path_to_train = args.train_path
        path_to_checkpoint = args.model_path

        train_model(path_to_train, path_to_checkpoint, model)


def get_model():
    model_unet = build_model()
    model_unet.compile(optimizer="adam", loss="mae", metrics=["mse"])
    return model_unet


if __name__ == "__main__":
    print("start args parse")
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="train model")
    parser.add_argument("--test", action="store_true", help="generate test predictions")
    parser.add_argument("--model-path", type=str, help="Default is most recent in checkpoint dir",
                        default="/home/s0105/_scratch2/geoposeEnsemble/checkpoints")
    parser.add_argument("--checkpoint-name", type=str, default="Weights_exact_elevation_727.h5")
    parser.add_argument("--train-path", type=str, help="Path to train data (rgb, vflow, agl)",
                        default="/home/s0105/_scratch2/geoposeEnsenble/train_data")
    parser.add_argument("--predict-path", type=str, help="Path to test data (rgb)",
                        default="/home/s0105/_scratch2/geoposeEnsemble/test_data")
    parser.add_argument("--save-path", type=str, help="Path to save predictions (agl)",
                        default="/home/s0105/_scratch2/geoposeEnsemble/out")

    arguments = parser.parse_args()

    print("go to main")
    main(arguments)
