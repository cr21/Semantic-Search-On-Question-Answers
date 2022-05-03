import argparse
import csv

ap = argparse.ArgumentParser()
ap.add_argument("-k", "--numPoints", help="Number of questions to sample")
args = vars(ap.parse_args())

k = args["numPoints"]
print("k", k)
if k:
    NUM_QUESTION_PROCESSED = int(k)
else:
    NUM_QUESTION_PROCESSED = 200000

print("NUM_QUESTION_PROCESSED", NUM_QUESTION_PROCESSED)
f = open('topkQuestions', 'w', encoding='latin1')
count = 0
with open('./data/Questions.csv', encoding='latin1') as ques_csv:
    reader = csv.reader(ques_csv, delimiter=',')
    next(reader, None)  # skip the headers
    for row in reader:
        # Id Field
        doc_id = row[0]
        # Title field
        title = row[5]
        # write to file
        f.write(doc_id + "," + title + "\n")
        count += 1
        if count % 100 == 0:
            print("[INFO] {} number of questions processed".format(count))
        if count == NUM_QUESTION_PROCESSED:
            break

    print("[INFO] Sample completed")

f.close()
