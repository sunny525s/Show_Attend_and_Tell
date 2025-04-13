import argparse
import json
from collections import Counter
from pathlib import Path


def build_word_dict(captions, min_word_count):
    """
    Builds a word dictionary from caption tokens, filtering by minimum count.
    """
    word_count = Counter(token for caption in captions for token in caption)
    vocab = [word for word, count in word_count.items() if count >= min_word_count]

    word_dict = {word: idx + 4 for idx, word in enumerate(vocab)}
    word_dict['<start>'] = 0
    word_dict['<eos>'] = 1
    word_dict['<unk>'] = 2
    word_dict['<pad>'] = 3

    return word_dict


def tokenize_captions(captions, word_dict, max_length):
    """
    Converts list of tokenized captions to indexed and padded format.
    """
    indexed = []
    for tokens in captions:
        idxs = [word_dict.get(token, word_dict['<unk>']) for token in tokens]
        padded = [word_dict['<start>']] + idxs + [word_dict['<eos>']]
        padded += [word_dict['<pad>']] * (max_length - len(tokens))
        indexed.append(padded)
    return indexed


def generate_json_data(split_path, data_path, max_captions, min_word_count):
    with open(split_path, 'r') as f:
        split_data = json.load(f)

    train_img_paths, val_img_paths = [], []
    train_captions, val_captions = [], []
    max_caption_len = 0

    for img in split_data['images']:
        captions_added = 0
        for sentence in img['sentences']:
            if captions_added >= max_captions:
                break
            captions_added += 1

            img_path = str(Path(data_path) / 'imgs' / img['filename'])
            tokens = sentence['tokens']
            max_caption_len = max(max_caption_len, len(tokens))

            if img['split'] == 'train':
                train_img_paths.append(img_path)
                train_captions.append(tokens)
            elif img['split'] == 'val':
                val_img_paths.append(img_path)
                val_captions.append(tokens)

    word_dict = build_word_dict(train_captions + val_captions, min_word_count)

    train_caption_ids = tokenize_captions(train_captions, word_dict, max_caption_len)
    val_caption_ids = tokenize_captions(val_captions, word_dict, max_caption_len)

    # Output paths
    data_path = Path(data_path)
    with open(data_path / 'word_dict.json', 'w') as f:
        json.dump(word_dict, f)
    with open(data_path / 'train_img_paths.json', 'w') as f:
        json.dump(train_img_paths, f)
    with open(data_path / 'val_img_paths.json', 'w') as f:
        json.dump(val_img_paths, f)
    with open(data_path / 'train_captions.json', 'w') as f:
        json.dump(train_caption_ids, f)
    with open(data_path / 'val_captions.json', 'w') as f:
        json.dump(val_caption_ids, f)

    print(f"Saved JSON files to {data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate caption training data in JSON format')
    parser.add_argument('--split-path', type=str, default='data/flickr8k/dataset.json')
    parser.add_argument('--data-path', type=str, default='data/flickr8k')
    parser.add_argument('--max-captions', type=int, default=5, help='Max captions per image')
    parser.add_argument('--min-word-count', type=int, default=5, help='Minimum word frequency to keep in vocab')
    args = parser.parse_args()

    generate_json_data(
        split_path=args.split_path,
        data_path=args.data_path,
        max_captions=args.max_captions,
        min_word_count=args.min_word_count
    )
