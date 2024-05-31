import tensorflow as tf
import numpy as np
import pandas as pd
import collections
import pretty_midi

# Load MIDI file
sample_file = 'ample.mid'
pm = pretty_midi.PrettyMIDI(sample_file)

# Extract notes from MIDI file
def midi_to_notes(midi_file: str) -> pd.DataFrame:
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

raw_notes = midi_to_notes(sample_file)
raw_notes.head()

# Convert pitch to note names
get_note_names = np.vectorize(pretty_midi.note_number_to_name)
sample_note_names = get_note_names(raw_notes['pitch'])
sample_note_names[:10]

# Create training dataset
num_files = 5
all_notes = []
for f in filenames[:num_files]:
    notes = midi_to_notes(f)
    all_notes.append(notes)

all_notes = pd.concat(all_notes)

n_notes = len(all_notes)
print('Number of notes parsed:', n_notes)

# Create tf.data.Dataset
key_order = ['pitch', 'tep', 'duration']
train_notes = np.stack([all_notes[key] for key in key_order], axis=1)

notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
notes_ds.element_spec

# Define custom loss function
def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
    mse = (y_true - y_pred) ** 2
    positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
    return tf.reduce_mean(mse + positive_pressure)

# Create and train the model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(None, 3)),
    tf.keras.layers.Dense(3)
])

model.compile(optimizer='adam', loss=mse_with_positive_pressure)

batch_size = 64
buffer_size = n_notes - seq_length  # the number of items in the dataset
train_ds = (notes_ds
           .shuffle(buffer_size)
           .batch(batch_size, drop_remainder=True)
           .cache()
           .prefetch(tf.data.experimental.AUTOTUNE))

model.fit(train_ds, epochs=10)
