import csv
import google.generativeai as genai 

genai.configure(api_key="AIzaSyCF7WqrdntJOb39QXcBJtTEQfsi_3DLR3U")


# Set up the model
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 5000,
}


safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name="gemini-1.0-pro", generation_config=generation_config, safety_settings=safety_settings
)

# Read the CSV file
with open("Corona_NLP_test.csv", "r") as csvfile:
    reader = csv.DictReader(csvfile)
    
    # Add new columns for annotations, promptgemini, and reason
    fieldnames = reader.fieldnames + ["annotations", "promptgemini", "reason"]

    # Iterate through rows and annotate tweets
    with open("corona_NLP_test_annotated_LLM.csv", "w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            tweet = row["OriginalTweet"]
            # print(tweet)

            # Add the tweet to a new column named "tweetsgemini"
            prompt = f"annotate this tweet: {tweet}, from following labels: Positive, Extremely positive, Negative, Extremely Negative and give reason"
            row["promptgemini"] = prompt

            # Send prompt to Gemini model
            try:
                convo = model.start_chat(history=[])
                convo.send_message(prompt)

                print(convo.last.text)
                # Extract sentiment label and reason
                if convo and convo.last:
                    response = convo.last.text
                    # Split the response to extract label and reason
                    label_start_index = response.find("**")  # Find the start of the label
                    label_end_index = response.find("**", label_start_index + 2)  # Find the end of the label
                    label = response[label_start_index + 2 : label_end_index].strip()
                    reason_start_index = response.find("**Reason:**")  # Find the start of the reason
                    reason = response[reason_start_index + 11:].strip()
                    allowed_labels = {"Positive", "Extremely positive", "Negative", "Extremely Negative"}
                    if label not in allowed_labels:
                        label = "Unknown sentiment"
                else:
                    label = "Error: Unable to analyze label tweet"
                    reason = "Error: Unable to analyze reason tweet"
            except Exception as e:
                print("Error occurred during conversation:", e)
                label = "Error: Unable to analyze label exception tweet"
                reason = "Error: Unable to analyze reason exception tweet"

            row["annotations"] = label
            row["reason"] = reason  # Assigning reason to the "reason" column
            writer.writerow(row)
