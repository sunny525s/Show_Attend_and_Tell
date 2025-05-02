import torch
import matplotlib.pyplot as plt
import os
import csv
from config import CHECKPOINT_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def metrics_over_epochs():
  epochs = []
  bleu1s, bleu2s, bleu3s, bleu4s = [], [], [], []
  meteor_scores = []
  val_scores = []
  train_scores = []

  checkpoint_files = sorted(os.listdir(CHECKPOINT_PATH))

  for ckpt_file in checkpoint_files[1:]:
      if ckpt_file.endswith(".pth") or ckpt_file.endswith(".pt"):
          checkpoint = torch.load(f"{CHECKPOINT_PATH}/{ckpt_file}", map_location=device, weights_only=False)
      else:
          continue
      bleu_scores = checkpoint['bleu-score']
      meteor_score = checkpoint['meteor-score']
      val_accuracy = checkpoint['validation-accuracy']
      train_accuracy = checkpoint['training-accuracy']

      epochs.append(checkpoint.get('epoch', len(epochs)))  #  fallback if epoch isn't saved
      bleu1s.append(bleu_scores[0])
      bleu2s.append(bleu_scores[1])
      bleu3s.append(bleu_scores[2])
      bleu4s.append(bleu_scores[3])
      meteor_scores.append(meteor_score)
      val_scores.append(val_accuracy)
      train_scores.append(train_accuracy)

  return epochs, bleu1s, bleu2s, bleu3s, bleu4s, meteor_scores, val_scores,train_scores
  
def generate_plot(epochs, bleu1s, bleu2s, bleu3s, bleu4s, meteor_scores, val_scores, train_scores):
    if not epochs:
      print("No data available to generate plot.")
      return
    plt.figure(figsize=(12, 15))

    plt.subplot(3, 1, 1)
    plt.plot(epochs, bleu1s, label="BLEU-1")
    plt.plot(epochs, bleu2s, label="BLEU-2")
    plt.plot(epochs, bleu3s, label="BLEU-3")
    plt.plot(epochs, bleu4s, label="BLEU-4")

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("BLEU Performance Over Epochs")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(epochs, meteor_scores, label="METEOR",   linestyle="--")
    plt.title('METEOR Score over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('METEOR Score')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(epochs, val_scores, label="Validation Accuracy")
    plt.plot(epochs, train_scores, label="Training Accuracy")

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, val_scores, label="Validation Accuracy")
    plt.plot(epochs, train_scores, label="Training Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Model Performance Over Epochs")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    output_filename = f"results/metric_plots.png"
    try:
        plt.savefig(output_filename)
        print(f"Plot saved to {output_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    # plt.show()

def generate_csv(epochs, bleu1s, bleu2s, bleu3s, bleu4s, meteor_scores, val_scores, train_scores):
    """Writes the provided metrics to a CSV file."""
    if not epochs:
        print("No data available to write to CSV.")
        return

    output_filename = f"results/metrics.csv"
    header = ['Epoch', 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'Train Accuracy', 'Validation Accuracy']

    rows = zip(epochs, bleu1s, bleu2s, bleu3s, bleu4s, meteor_scores, train_scores, val_scores)

    try:
        with open(output_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"CSV data saved to {output_filename}")
    except IOError as e:
        print(f"Error writing CSV file: {e}")


def main():
    epochs, bleu1s, bleu2s, bleu3s, bleu4s, meteor_score, val_scores, train_scores = metrics_over_epochs()

    generate_plot(epochs[1:], bleu1s[1:], bleu2s[1:], bleu3s[1:], bleu4s[1:], meteor_score[1:], val_scores[1:], train_scores[1:])
    generate_csv(epochs[1:], bleu1s[1:], bleu2s[1:], bleu3s[1:], bleu4s[1:], meteor_score[1:], val_scores[1:], train_scores[1:])

if __name__ == "__main__":
    main()
