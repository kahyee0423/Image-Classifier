import argparse
from model_utils import load_checkpoint, predict
from data_utils import process_image
import torch

def main():
    parser = argparse.ArgumentParser(description="Predict image class using a trained model.")
    parser.add_argument("input", type=str, help="Path to input image")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--top_k", type=int, default=1, help="Top K most likely classes")
    parser.add_argument("--category_names", type=str, help="JSON file mapping categories to names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    args = parser.parse_args()
    
    # Load model
    model = load_checkpoint(args.checkpoint)
    
    # Process image
    image = process_image(args.input)
    
    # Predict
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    probs, classes = predict(image, model, args.top_k, device)
    
    # Map categories to names if provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        labels = [cat_to_name[str(cls)] for cls in classes]
    else:
        labels = classes
    
    print(f"Predictions: {labels}")
    print(f"Probabilities: {probs}")

if __name__ == "__main__":
    main()
