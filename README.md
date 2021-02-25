# Speech-Recognition-Using-Tensorflow

Implementation of a seq2seq model for speech recognition. Architecture similar to "Listen, Attend and Spell".
https://arxiv.org/pdf/1508.01211.pdf

```
Created: ['S', 'E', 'V', 'E', 'N', 'T', 'E', 'E', 'N', '<SPACE>', 'T', 'W', 'E', 'N', 'T', 'Y', '<SPACE>', 'F', 'O', 'U', 'R']
Actual: ['S', 'E', 'V', 'E', 'N', 'T', 'E', 'E', 'N', '<SPACE>', 'T', 'W', 'E', 'N', 'T', 'Y', '<SPACE>', 'F', 'O', 'U', 'R']
```

**Requirements:**

- Tensorflow
- numpy
- pandas 
- librosa
- python_speech_features

**Dataset:**

The dataset I used is the LibriSpeech dataset. It contains about 1000 hours of 16kHz read English speech.
Source: http://www.openslr.org/12/

**Architecture Used:**

**Seq2Seq model**

We're using pyramidal bidirectional LSTMs in the encoder. This reduces the time resolution and enhances the performance on longer sequences.

- Encoder-Decoder
- Pyramidal Bidirectional LSTM
- Bahdanau Attention
- Adam Optimizer
- exponential or cyclic learning rate
- Beam Search or Greedy Decoding
