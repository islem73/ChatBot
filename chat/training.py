import random
import json
import pickle
import numpy as np
import tensorflow as tf


import nltk
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents
intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']

# Process patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize and lemmatize
        word_list = nltk.word_tokenize(pattern)
        word_list = [lemmatizer.lemmatize(word.lower()) for word in word_list if word not in ignore_letters]
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Remove duplicates and sort
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    
    # Create bag of words
    bag = [1 if word in word_patterns else 0 for word in words]
    
    # Create output row
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])

# Shuffle and separate features and labels
random.shuffle(training)
train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

# Build model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train model
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5' )

print("Done")