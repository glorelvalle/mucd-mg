import json


def load_all_json_yelp(data_name, data_path="yelp_dataset"):
    with open(f"{data_path}/{data_name}.json", encoding="utf-8") as f:
        reviews = json.load(f)
    numReviews = len(reviews)
    print(f"{data_name}: {numReviews} reviews loaded")
    return reviews


def load_by_line_json_yelp(data_name, data_path="yelp_dataset"):
    reviews = []
    with open(f"{data_path}/{data_name}.json", encoding="utf-8") as f:
        f.readline()  # first line '['
        numReviews = 0
        while True:
            numReviews += 1
            line = f.readline().strip()  # Get next line from file
            if line == "]":  # end of file is reached ']'
                print(f"{data_name}: {numReviews} reviews loaded")
                break
            if line[-1] == ",":
                line = line[:-1]
            reviews.append(json.loads(line))
    return reviews


def load_huge_file(data_name, data_path="amazon", limit=8898040):
    reviews = []
    with open(f"{data_path}/{data_name}.json", encoding="utf-8") as f:
        f.readline()  # first line '['
        n_reviews = 0
        while True:
            n_reviews += 1
            line = f.readline().strip()  # Get next line from file
            if n_reviews == limit:
                print(f"{data_name}: {n_reviews} reviews loaded")
                break
            reviews.append(json.loads(line))
    return reviews


def load_aspects(data_name, data_path="aspects"):
    with open(f"{data_path}/{data_name}.csv", encoding="utf-8") as f:
        aspects = {}
        for line in f:
            key, synonymous = line.rstrip("\n").split(",")
            if key in aspects and synonymous not in aspects[key]:
                aspects[key].append(synonymous)
            else:
                aspects[key] = []
    return aspects
