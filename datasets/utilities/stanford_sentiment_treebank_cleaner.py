import csv

def process_sentence_sentiment(sentence_file, sentiment_file, output_file):
    # Read the sentiment values from the original sentiment file
    sentiment_dict = {}
    with open(sentiment_file, 'r') as infile:
        # Skip header
        next(infile)
        
        # Read phrase id and sentiment value into a dictionary
        for line in infile:
            phrase_id, sentiment_value = line.strip().split('|')
            sentiment_value = float(sentiment_value)

            # Determine sentiment word based on the value
            if sentiment_value < 0.5:
                sentiment_dict[phrase_id] = "negative"
            else:
                sentiment_dict[phrase_id] = "positive"

    # Now, process the sentence file and combine it with sentiment
    with open(sentence_file, 'r') as sentence_infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(sentence_infile, delimiter='\t')
        writer = csv.writer(outfile)

        # Write header to output file
        writer.writerow(["review", "sentiment"])

        # Skip header in the sentence file
        next(reader)

        # For each sentence, find the corresponding sentiment
        for row in reader:
            sentence_index, sentence = row
            # Find the sentiment for this sentence using the sentence index
            sentiment = sentiment_dict.get(sentence_index, "unknown")
            
            # Write sentence and sentiment to the output CSV
            writer.writerow([sentence, sentiment])

    print(f"Conversion complete. Check {output_file}.")

# Usage
sentence_file = 'datasetSentences.txt'
sentiment_file = 'sentiment_labels.txt'
output_file = 'stanford_rotten_tomatoes_dataset.csv'

process_sentence_sentiment(sentence_file, sentiment_file, output_file)
