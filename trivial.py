import string

from helpers import DataSet

# Initialize dataset
dataset = DataSet("IMDB Dataset.csv", "sentiment")

# Get reviews and sentiments
reviews = dataset.get_X()
sentiments = dataset.get_y()

print("train x length", len(reviews))

# List of good and bad words
good_words = ["good", "great", "excellent", "awesome", "outstanding", 
              "fantastic", "terrific", "amazing", "superb", "wonderful"]
bad_words = ["bad", "terrible", "awful", "worst", "atrocious", "horrible", 
             "dreadful", "abysmal", "appalling", "lousy"]

# Variables to track correct and incorrect sentiments
correct_sentiments = 0
incorrect_sentiments = 0
unasisgned_sentiments = 0

# Iterate over the reviews
for index, review in reviews.iterrows():
    # print("review", review.values)
    
    good_words_count = 0
    bad_words_count = 0

    # Convert review to lowercase and remove punctuation
    review_cleaned = str(review.values).lower().translate(str.maketrans('', '', string.punctuation))

    # Count good and bad words in the cleaned review
    for word in review_cleaned.split():
        if word in good_words:
            good_words_count += 1
        if word in bad_words:
            bad_words_count += 1
    
    # Determine if the sentiment prediction is correct
    if good_words_count > bad_words_count:
        if sentiments[index] == "positive":
            correct_sentiments += 1
        else:
            incorrect_sentiments += 1
    elif good_words_count < bad_words_count:
        if sentiments[index] == "negative":
            correct_sentiments += 1
        else:
            incorrect_sentiments += 1
    else:
        unasisgned_sentiments += 1

# Output the results
print("correct sentiments:", correct_sentiments)
print("incorrect sentiments:", incorrect_sentiments)
print("unassigned sentiments:", unasisgned_sentiments)
print("accuracy:", correct_sentiments / len(reviews))
print("accuracy:", correct_sentiments / (correct_sentiments + incorrect_sentiments))
