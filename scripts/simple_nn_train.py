import argparse
import pickle
import unicodedata
import matplotlib.pyplot as plt
from constants import *
from dataset_downloader import *
from simple_nn_model import SimpleNeuralNetwork

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--new", type=int, help="Force to retrain model")
ap.add_argument("-c", "--category", type=str, default="vn", required=False, help="category: digit | vn | en")
args = vars(ap.parse_args())

rs_paths = {
    "vn": RS_VN_CHARS,
}
arg_cat = str.lower(args['category'])
if arg_cat not in rs_paths.keys():
    print("!!!Unknown category. Accepted categories:", ", ".join(rs_paths.keys()))
    exit()

rs_path = rs_paths[arg_cat]
char_list = {
    "vn": VN_CHARS_LIST,
}[arg_cat]
test_path = {
    "vn": TEST_VN_CHARS_PATH,
}[arg_cat]
model_filepath = f"{MODELS_PATH}/simple_nn_model_{arg_cat}.pickle"
LABELS_COUNT = len(char_list)


def show_images(image, labels, num_row=2, num_col=5):
    image_size = int(np.sqrt(image.shape[-1]))
    image = np.reshape(image, (image.shape[0], image_size, image_size))
    fig, axes = plt.subplots(num_row, num_col, figsize=(num_col, num_row))
    for i in range(num_row * num_col):
        ax = axes[i // num_col, i % num_col]
        ax.imshow(image[i], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        print(f"{i}: {np.argmax(labels[i])} - {char_list[np.argmax(labels[i])]}")
    plt.tight_layout()
    plt.show()


def one_hot(arr, k, dtype=np.float32):
    return np.array(arr[:, None] == np.arange(k), dtype)


if args["new"] != 1 and os.path.exists(model_filepath):
    print("... Loading model from file")
    with open(model_filepath, "rb") as f:
        model_file = pickle.load(f)
        simple_nn = model_file["model"]
else:
    print("... Initialize new model and train on new dataset")
    x, y = get_vn_chars_xy(rs_path)
    x = np.reshape(x, [x.shape[0], x.shape[1] * x.shape[2]])
    x = x / 255.0
    examples = y.shape[0]
    y_new = one_hot(y.astype('uint8'), LABELS_COUNT)
    train_size = int(x.shape[0] * 1.0)
    test_size = int(x.shape[0] * 0.8)
    x_train, x_test = x[:train_size], x[test_size:]
    y_train, y_test = y_new[:train_size], y_new[test_size:]
    shuffle_index = np.random.permutation(train_size)
    x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]
    # show_images(x_train, y_train, num_row=2, num_col=5)
    # exit()

    simple_nn = SimpleNeuralNetwork(sizes=[x_train.shape[1], 64, LABELS_COUNT], activation='sigmoid')
    simple_nn.train(x_train, y_train, x_test, y_test, batch_size=32, optimizer='momentum', l_rate=4)

    with open(model_filepath, "wb") as f:
        pickle.dump({"model": simple_nn}, f)
        f.close()
        print("Saved model to file")

# Testing
file_count = 0
match_count = 0
filenames = sorted(os.listdir(test_path))
for filename in filenames:
    if filename.endswith(f".{IMG_EXT}"):
        file_count += 1
        img = cv2.imread(f"{test_path}/{filename}")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        pred_cases = simple_nn.predict(img)
        idx = np.argmax(pred_cases)
        file_label = unicodedata.ucd_3_2_0.normalize('NFC', filename.split(".")[0].split(" ")[1]).encode("utf-8")
        match_str = ""
        if file_label == char_list[idx].encode("utf-8"):
            match_count += 1
            match_str = "     <"
        print(f"pred_idx[{idx}]: {char_list[idx]} | {filename} | acc: {np.max(pred_cases):.2f}{match_str}")
print(f"Matched: {match_count}/{file_count}")
