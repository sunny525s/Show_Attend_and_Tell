import json, os
from collections import Counter
import argparse

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

def decode_caption(caption, word_dict):
    """
    Decode given caption using word_dict
    """
    idx_to_word = {idx: word for word, idx in word_dict.items()}
    decoded_caption = []

    for id in caption.numpy():
        if idx_to_word[id] not in ["<start>", "<eos>", "<unk>", "<pad>"]:
            decoded_caption.append(idx_to_word[id])

    return " ".join(decoded_caption)

def tokenize_captions(captions, word_dict, max_length):
    """
    Converts list of tokenized captions to indexed and padded format.
    """
    indexed, caps_len = [], []
    for tokens in captions:
        idxs = [word_dict.get(token, word_dict['<unk>']) for token in tokens]
        padded = [word_dict['<start>']] + idxs + [word_dict['<eos>']]
        padded += [word_dict['<pad>']] * (max_length - len(tokens))
        indexed.append(padded)
        caps_len.append(len(tokens) + 2)
    return indexed, caps_len

def generate_json_data(split_path, data_path, min_word_count, max_caption_len=100):
    """
    Splits data into train, val, test dataset and saves as JSON
    """
    with open(split_path, 'r') as f:
        split_data = json.load(f)

    train_img_paths, val_img_paths, test_img_paths = [], [], []
    train_captions, val_captions, test_captions = [], [], []

    for img in split_data['images']:
        for sentence in img['sentences']:
            img_path = os.path.join(data_path, 'imgs', img['filename'])
            tokens = sentence['tokens']

            if img['split'] == 'train':
                train_img_paths.append(img_path)
                train_captions.append(tokens)
            elif img['split'] == 'val':
                val_img_paths.append(img_path)
                val_captions.append(tokens)
            elif img['split'] == 'test':
                test_img_paths.append(img_path)
                test_captions.append(tokens)

    word_dict = build_word_dict(train_captions + val_captions + test_captions, min_word_count)

    train_caption_ids, train_caption_lens = tokenize_captions(train_captions, word_dict, max_caption_len)
    val_caption_ids, val_caption_lens = tokenize_captions(val_captions, word_dict, max_caption_len)
    test_caption_ids, test_caption_lens = tokenize_captions(test_captions, word_dict, max_caption_len)

    # Output paths
    with open(os.path.join(data_path, 'word_dict.json'), 'w') as f:
        json.dump(word_dict, f)

    with open(os.path.join(data_path, 'train_img_paths.json'), 'w') as f:
        json.dump(train_img_paths, f)
    with open(os.path.join(data_path, 'val_img_paths.json'), 'w') as f:
        json.dump(val_img_paths, f)
    with open(os.path.join(data_path, 'test_img_paths.json'), 'w') as f:
        json.dump(test_img_paths, f)

    with open(os.path.join(data_path, 'train_captions.json'), 'w') as f:
        json.dump(train_caption_ids, f)
    with open(os.path.join(data_path, 'val_captions.json'), 'w') as f:
        json.dump(val_caption_ids, f)
    with open(os.path.join(data_path, 'test_captions.json'), 'w') as f:
        json.dump(test_caption_ids, f)

    with open(os.path.join(data_path, 'train_caption_lens.json'), 'w') as f:
        json.dump(train_caption_lens, f)
    with open(os.path.join(data_path, 'val_caption_lens.json'), 'w') as f:
        json.dump(val_caption_lens, f)
    with open(os.path.join(data_path, 'test_caption_lens.json'), 'w') as f:
        json.dump(test_caption_lens, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate caption training data in JSON format')
    parser.add_argument('--split-path', type=str, default='data/flickr8k/dataset.json')
    parser.add_argument('--data-path', type=str, default='data/flickr8k')
    parser.add_argument('--min-word-count', type=int, default=5, help='Minimum word frequency to keep in vocab')
    parser.add_argument('--max-caption-len', type=int, default=100, help='Maximum length of the caption')
    args = parser.parse_args()

    generate_json_data(
        split_path=args.split_path,
        data_path=args.data_path,
        min_word_count=args.min_word_count,
        max_caption_len=args.max_caption_len
    )
