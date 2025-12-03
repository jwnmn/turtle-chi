import numpy as np
import json
from train_mlp_classifier import (
    load_keypoints_dict,
    extract_features,
    MLPBinaryClassifier
)

# load scaler and model weights
def load_mlp_from_npz(model_path, scaler_path, input_dim, hidden_dims=(64,32)):
    # load parameters
    data = np.load(model_path)
    scaler = np.load(scaler_path)

    mean = scaler["mean"]
    std = scaler["std"]

    # create model object
    model = MLPBinaryClassifier(input_dim=input_dim, hidden_dims=hidden_dims)

    # load weights
    model.W1 = data["W1"]
    model.b1 = data["b1"]
    model.W2 = data["W2"]
    model.b2 = data["b2"]
    model.W3 = data["W3"]
    model.b3 = data["b3"]

    return model, mean, std


# main test
if __name__ == "__main__":
    correct_kpts = load_keypoints_dict("dataset/movement_1/correct.json")
    incorrect_kpts = load_keypoints_dict("dataset/movement_1/incorrect.json")

    X_correct = np.array([extract_features(k) for k in correct_kpts])
    X_incorrect = np.array([extract_features(k) for k in incorrect_kpts])

    input_dim = X_correct.shape[1]

    # Load model + scaler
    model, mean, std = load_mlp_from_npz(
        "models/ver3/movement_1_mlp_scratch_ver3.npz",
        "models/ver3/movement_1_scaler_scratch_ver3.npz",
        input_dim
    )

    # Normalize features with same scaler as training
    X_correct_norm = (X_correct - mean) / std
    X_incorrect_norm = (X_incorrect - mean) / std

    print("Correct samples:")
    for i in range(30):
        pred = model.predict(X_correct_norm[i:i+1])[0]
        print(f"Correct sample {i}: Pred={pred}")

    print("Incorrect samples:")
    for i in range(30):
        pred = model.predict(X_incorrect_norm[i:i+1])[0]
        print(f"Incorrect sample {i}: Pred={pred}")