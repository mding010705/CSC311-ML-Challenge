import numpy as np
import pandas as pd
import random
import sys
import re
import time

stop_words = set(
    ["a", "about", "above", "after", "again", "against", "ain", "all",
     "am", "an", "and", "any", "are", "aren", "aren't", "as", "at", "be",
     "because", "been", "before", "being", "below", "between", "both",
     "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't",
     "do", "does", "doesn", "doesn't", "doing", "don", "don't", "down",
     "during", "each", "few", "for", "from", "further", "had", "hadn",
     "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't",
     "having", "he", "he'd", "he'll", "her", "here", "hers", "herself",
     "he's", "him", "himself", "his", "how", "i", "i'd", "if", "i'll",
     "i'm", "in", "into", "is", "isn", "isn't", "it", "it'd", "it'll",
     "it's", "its", "itself", "i've", "just", "ll", "m", "ma", "me",
     "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my",
     "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of",
     "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves",
     "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she'd",
     "she'll", "she's", "should", "shouldn", "shouldn't", "should've", "so",
     "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs",
     "them", "themselves", "then", "there", "these", "they", "they'd",
     "they'll",
     "they're", "they've", "this", "those", "through", "to", "too", "under",
     "until", "up", "ve", "very", "was", "wasn", "wasn't", "we", "we'd",
     "we'll", "we're", "were", "weren", "weren't", "we've", "what", "when",
     "where", "which", "while", "who", "whom", "why", "will", "with", "won",
     "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "your",
     "you're", "yours", "yourself", "yourselves", "you've", ""])
sub_imdb = pd.read_csv("tmdb_movies_tv.csv")
sub_imdb['title'] = sub_imdb['title'].apply(str)


# --- Helper Functions for Data Cleaning ---

def change_movie(movie: str):
    return re.sub(r'[^\w\s]', '',
                  str(movie).lower().removeprefix("the ").removeprefix(
                      "a ").removeprefix("an ")).replace(" ", "")[:18]


def detect_num_of_ingredients(text: str) -> list[str]:
    text = str(text)
    numbers = re.findall(r"\d+", str(text))
    numbers = [int(num) for num in numbers]

    number_words = ["one", "two", "three", "four", "five", "six", "seven",
                    "eight", "nine", "ten", "eleven", "twelve", "thirteen",
                    "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
                    "nineteen", "twenty"]

    word_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7,
                "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
                "thirteen": 13,
                "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
                "eighteen": 18,
                "nineteen": 19, "twenty": 20}

    numbers += [word_map[word] for word in number_words if word in text.lower()]

    if numbers == []:
        ingredients_list = re.split(r",", text)
        return [len(ingredients_list)]

    return numbers


def detect_price(text: str) -> list[str]:
    numbers = re.findall(r"\d+(?:\.\d+)?", str(text))

    numbers = [float(num) for num in numbers if num != '']

    number_words = ["one", "two", "three", "four", "five", "six", "seven",
                    "eight", "nine", "ten", "eleven", "twelve", "thirteen",
                    "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
                    "nineteen", "twenty"]

    word_map = {"one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0,
                "six": 6.0, "seven": 7.0,
                "eight": 8.0, "nine": 9.0, "ten": 10.0, "eleven": 11.0,
                "twelve": 12.0, "thirteen": 13.0,
                "fourteen": 14.0, "fifteen": 15.0, "sixteen": 16.0,
                "seventeen": 17.0, "eighteen": 18.0,
                "nineteen": 19.0, "twenty": 20.0, "quarter": 0.25, "half": 0.5,
                "thirty": 30.0}

    numbers += [word_map[word] for word in number_words if
                word in str(text).lower()]

    if str(text).lower() == "dollar each":
        numbers = [1.0]

    return numbers


def clean_ingredients(text: str) -> int:
    # Number of ingredients
    if not text:
        return None
    num_ingredients = detect_num_of_ingredients(text)
    if num_ingredients:
        return sum(num_ingredients) / len(num_ingredients)
    else:
        return None


def clean_cost(text: str) -> float:
    # Expected cost
    if not text:
        return None
    cost = detect_price(text)
    if cost:
        return sum(cost) / len(cost)
    else:
        return None


def detect_stopwords(text: str) -> bool:
    words = text.lower().split()
    return any(word in stop_words for word in words)


def detect_capitilized(text: str) -> list[str]:
    return re.findall(r'[A-Z][a-zA-Z]*', text)


# ============================================================================
# from https://github.com/nltk/nltk/blob/develop/nltk/metrics/distance.py
def jaro_similarity(s1, s2):
    # First, store the length of the strings
    # because they will be re-used several times.
    len_s1, len_s2 = len(s1), len(s2)

    # The upper bound of the distance for being a matched character.
    match_bound = max(len_s1, len_s2) // 2 - 1

    # Initialize the counts for matches and transpositions.
    matches = 0  # no.of matched characters in s1 and s2
    transpositions = 0  # no. of transpositions between s1 and s2
    flagged_1 = []  # positions in s1 which are matches to some character in s2
    flagged_2 = []  # positions in s2 which are matches to some character in s1

    # Iterate through sequences, check for matches and compute transpositions.
    for i in range(len_s1):  # Iterate through each character.
        upperbound = min(i + match_bound, len_s2 - 1)
        lowerbound = max(0, i - match_bound)
        for j in range(lowerbound, upperbound + 1):
            if s1[i] == s2[j] and j not in flagged_2:
                matches += 1
                flagged_1.append(i)
                flagged_2.append(j)
                break
    flagged_2.sort()
    for i, j in zip(flagged_1, flagged_2):
        if s1[i] != s2[j]:
            transpositions += 1

    if matches == 0:
        return 0
    else:
        return (
                1
                / 3
                * (
                        matches / len_s1
                        + matches / len_s2
                        + (matches - transpositions // 2) / matches
                )
        )


def jaro_winkler_similarity(s1, s2, p=0.1, max_l=4):
    # To ensure that the output of the Jaro-Winkler's similarity
    # falls between [0,1], the product of l * p needs to be
    # also fall between [0,1].
    # Compute the Jaro similarity
    jaro_sim = jaro_similarity(s1, s2)

    # Initialize the upper bound for the no. of prefixes.
    # if user did not pre-define the upperbound,
    # use shorter length between s1 and s2

    # Compute the prefix matches.
    l = 0
    # zip() will automatically loop until the end of shorter string.
    for s1_i, s2_i in zip(s1, s2):
        if s1_i == s2_i:
            l += 1
        else:
            break
        if l == max_l:
            break
    # Return the similarity value as described in docstring.
    return jaro_sim + (l * p * (1 - jaro_sim))


# ===========================================================================

def detect_movie(text: str) -> list[str]:
    if not text:
        return []
    search = sub_imdb.loc[sub_imdb['title'] == text]
    if search.empty:
        sims = [0] * len(sub_imdb['title'])
        sims = sub_imdb['title'].apply(jaro_similarity, s2=text)
        search = sub_imdb.iloc[[np.argmax(sims)]]

    return search


def detect_drinks(text: str) -> list[str]:
    if not text:
        return []
    text = str(text)
    common_drinks = {'7up': ['fruit', 'soda'], 'coca-cola': ['cola', 'soda'],
                     'iced tea': ['tea', 'soda'], 'pepsi': ['cola', 'soda'],
                     'fanta': ['fruit', 'soda'],
                     'mirinda': ['fruit', 'soda'],
                     'orange juice': ['fruit', 'juice'],
                     'ginger ale': ['ginger', 'soda'],
                     'lemonade': ['fruit', 'juice'],
                     'sprite': ['fruit', 'soda'],
                     'kombucha': ['ginger', 'tea'], 'water': ['water'],
                     'lemon water': ['fruit', 'water'], 'pop': ['soda'],
                     'diet coke': ['cola', 'soda'],
                     'ayran yogurt': ['dairy', 'mideast'],
                     'beer': ['alcohol', 'soda'], 'juice': ['fruit', 'juice'],
                     'wine': ['alcohol', 'wine'],
                     'green tea': ['tea', 'asian', 'water'],
                     'sake': ['alcohol', 'asian'],
                     'tea': ['asian', 'tea', 'water'],
                     'cocktail': ['alcohol', 'liquor'],
                     'bubble tea': ['tea', 'juice', 'dairy'],
                     'crush': ['fruit', 'soda'],
                     'canada dry': ['ginger', 'soda'],
                     'red wine': ['alcohol', 'wine'],
                     'coco cola': ['cola', 'soda'],
                     'alcohol': ['alcohol', 'liquor'],
                     'chocolate milk': ['dairy', 'juice'],
                     'fanta': ['fruit', 'soda'], 'soft drink': ['soda'],
                     'sparkling water': ['water', 'soda'],
                     'mountain dew': ['fruit', 'soda'], 'no drink': ['none'],
                     'alcoholic drink': ['alcohol', 'liquor'],
                     'milk': ['dairy'], 'coke': ['cola', 'soda'],
                     'soda': ['soda'], 'cola': ['cola', 'soda'],
                     'mango lassi': ['fruit', 'dairy'],
                     'soy sauce': ['soy_sauce', 'asian']}
    text = re.sub(r'[^\w\s]', '', text).lower()
    drinks = []
    if detect_stopwords(text):
        querywords = text.split()
        resultwords = [word for word in querywords if
                       word.lower() not in stop_words]
        text = ' '.join(resultwords)
        drinks = [common_drinks[drink] for drink in list(common_drinks) if
                  drink in text]
    if drinks:
        drinks = [x for xs in drinks for x in xs]
        drinks = list(set(drinks))
        return ','.join(drinks)
    else:
        sims = [0] * len(common_drinks)
        for i in range(len(common_drinks)):
            sims[i] = jaro_winkler_similarity(text,
                                              list(common_drinks)[i])
            search = common_drinks[list(common_drinks)[np.argmax(sims)]]
        return ','.join(search)


def clean_drink(text: str) -> str:
    if not text:
        return None
    drinks = detect_drinks(text)
    if drinks:
        return drinks
    else:
        return None


def clean_movie(text: str) -> list:
    movies = detect_movie(text)
    if len(movies) > 0:
        return [movies["title"].values[0], movies["genre_ids"].values[0]]
    else:
        return ["none", ""]


def final_cleaning(fn: str):
    df = pd.read_csv(fn)
    df = df.rename(columns={
        "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)": 'food_complexity',
        'Q2: How many ingredients would you expect this food item to contain?': 'num_ingredients',
        'Q3: In what setting would you expect this food to be served? Please check all that apply': 'serving_setting',
        'Q4: How much would you expect to pay for one serving of this food item?': 'expected_cost',
        'Q5: What movie do you think of when thinking of this food item?': 'related_movie',
        'Q6: What drink would you pair with this food item?': 'paired_drink',
        "Q7: When you think about this food item, who does it remind you of?": 'associated_people',
        'Q8: How much hot sauce would you add to this food item?': 'hot_sauce_level'})
    df["movie_genres"] = ''
    df["num_ingredients"] = df["num_ingredients"].apply(clean_ingredients)
    df["expected_cost"] = df["expected_cost"].apply(clean_cost)
    df["paired_drink"] = df["paired_drink"].apply(clean_drink)
    df["related_movie"] = df["related_movie"].apply(change_movie)
    cleaned_movies = df["related_movie"].apply(clean_movie).to_list()
    df[["related_movie", "movie_genres"]] = pd.DataFrame(cleaned_movies,
                                                         index=df.index)
    df.hot_sauce_level = df.hot_sauce_level.fillna('none')
    df = df.fillna(0)

    # Convert categorical labels to indicators

    x = df.drop(columns=[col for col in df.columns if col.startswith("Label")])
    y_ = df["Label"]
    split_fets = ["serving_setting", "paired_drink", "associated_people",
                  "movie_genres"]
    for fet in split_fets:
        x_dumb = x[fet].str.get_dummies(sep=",")
        x_dumb = x_dumb.add_prefix((fet[fet.index('_') + 1:] + "_"), axis=1)
        x = pd.concat([x, x_dumb], axis=1)
    x = pd.get_dummies(x, columns=["related_movie", "hot_sauce_level"],
                       prefix=["movie", "hot"])

    x_ = x.drop(columns=[col for col in x.columns
                         if col in ["serving_setting", "paired_drink",
                                    "associated_people", "related_movie",
                                    "movie_genres"]])
    df = pd.concat([x_, y_], axis=1)
    fets = pd.read_csv("/clean_results_final.csv")
    learned_fets = list(fets)
    df = pd.concat([fets, df], axis=0, join="outer", sort=False)
    df = df.drop(columns=[col for col in df.columns
                          if col not in learned_fets])
    df = df[1644:]



    df.to_csv("/clean_results_test.csv")
    df.drop(columns=[col for col in df.columns if col.startswith('Unnamed') or col == 'id'], inplace=True)
    df = df.fillna(0)
    X = df.drop(columns=[col for col in df.columns if col.startswith("Label")]).values
    y = df["Label"]
    return X, y


# --- Helper Functions for Splitting ---

def gini_impurity(y):
    """Calculate the Gini impurity for a list/array of labels."""
    m = len(y)
    if m == 0:
        return 0
    counts = {}
    for label in y:
        counts[label] = counts.get(label, 0) + 1
    impurity = 1.0 - sum((count / m) ** 2 for count in counts.values())
    return impurity


def gini_index(y_left, y_right):
    """Calculate the weighted Gini index for a split."""
    m = len(y_left) + len(y_right)
    return (len(y_left) / m) * gini_impurity(y_left) + (
                len(y_right) / m) * gini_impurity(y_right)


def best_split(X, y, feature_indices):
    """For a given set of features, find the best feature and threshold that yields the lowest Gini impurity."""
    best_feature, best_threshold, best_gini = None, None, float('inf')
    for feature in feature_indices:
        values = X[:, feature]
        for threshold in np.unique(values):
            left_indices = values <= threshold
            right_indices = values > threshold
            if sum(left_indices) == 0 or sum(right_indices) == 0:
                continue
            y_left = y[left_indices]
            y_right = y[right_indices]
            current_gini = gini_index(y_left, y_right)
            if current_gini < best_gini:
                best_gini = current_gini
                best_feature = feature
                best_threshold = threshold
    return best_feature, best_threshold, best_gini


def most_common_label(y):
    """Return the most common label from an array."""
    counts = {}
    for label in y:
        counts[label] = counts.get(label, 0) + 1
    return max(counts, key=counts.get)


# --- Decision Tree Implementation ---
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None,
                 right=None, *, value=None):
        self.feature_index = feature_index  # index of feature used for split
        self.threshold = threshold  # threshold value for the split
        self.left = left  # left subtree
        self.right = right  # right subtree
        self.value = value  # class label for leaf node


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features  # number of features to consider at each split
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1]
        if self.max_features is None:
            self.max_features = self.n_features
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if len(set(y)) == 1:
            return Node(value=y[0])
        if len(y) < self.min_samples_split or (
                self.max_depth is not None and depth >= self.max_depth):
            return Node(value=most_common_label(y))
        features = random.sample(range(self.n_features), self.max_features)
        feature_index, threshold, _ = best_split(X, y, features)
        if feature_index is None:
            return Node(value=most_common_label(y))
        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold
        left_node = self._build_tree(X[left_indices], y[left_indices],
                                     depth + 1)
        right_node = self._build_tree(X[right_indices], y[right_indices],
                                      depth + 1)
        return Node(feature_index=feature_index, threshold=threshold,
                    left=left_node, right=right_node)

    def predict_one(self, x):
        node = self.root
        while node.value is None:
            node = node.left if x[
                                    node.feature_index] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])


# --- Random Forest Implementation ---
class RandomForest:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2,
                 max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                max_features=self.max_features)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array(
            [most_common_label(tree_preds[:, i]) for i in range(X.shape[0])])


# --- Prediction Process ---
def predict_all(filename):
    start_time = time.time()  # Start measuring time

    df = pd.read_csv("clean_results_final.csv")
    df.drop(columns=[col for col in df.columns if
                     col.startswith('Unnamed') or col == 'id'], inplace=True)
    df = df.fillna(0)

    target_column = 'Label'
    y_series = df[target_column]
    df.drop(columns=[col for col in df.columns if col.startswith('Label')],
            inplace=True)
    unique_labels = np.unique(y_series)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_series = y_series.map(label_map)

    X_train, y_train = df.values, y_series.values
    np.random.seed(42)

    """
    indices = np.random.permutation(len(y))
    train_size = int(0.7 * len(y))
    val_size = int(0.25 * len(y))
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:]
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    """

    rf_final = RandomForest(n_estimators=100, max_depth=15, min_samples_split=2,
                            max_features=int(np.sqrt(X_train.shape[1])))
    rf_final.fit(X_train, y_train)

    # clean test data
    X_test, y_test = final_cleaning(filename)

    #y_test = y_test.map(label_map).values

    y_test_pred = rf_final.predict(X_test)
    inv_map = {v: k for k, v in label_map.items()}
    y_pred = np.array([inv_map[x] for x in y_test_pred])

    #final_accuracy = np.mean(y_test_pred == y_test)

    #print("\nFinal Test Accuracy:", final_accuracy)

    y_pred.tofile(filename, sep = ",")



if __name__ == "__main__":
    filename = sys.argv[1]
    predict_all(filename)
