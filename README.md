# tf-serving-chatbot-attention
### Description
The primary goal of this repository is to demonstrate tensorflow serving a chat bot model based on encoder-decoder design with attention. Model was built on tensorflow 2.4.1 following ideas in [tensorflow tutorial](https://www.tensorflow.org/tutorials/text/nmt_with_attention) and trained using [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html).
### Dependencies
- For simple chat-bot client: built-in python libraries: sys, json, request, re; external pythons library: numpy
- Model creation: tensorflow to create saved_model export based on trained weights.
- Serving: tensorflow_model_server installed on machine.
### Instructions
1. Download weights from [link](https://drive.google.com/file/d/1uzhRE9q3arb13BGyJvAGS-j1FUuZFzub/view?usp=sharing) and unzip in project folder
2. Go through steps in test.ipynb to create tensorflow models (encoder and decoder), load pre-trained weights for every layer, save model in saved_model (tf) format apropriate for tensorflow_model_server.
3. Run tensorflow_model_server as in serving_command.txt
4. Once server is up and running, start chat-bot.py without arguments for chatting in terminal or with any argument to get bot's reply to a fixed utterance.

### Info about models.

#### Encoder
As long as we use attention for decoder, the encoder's input sequence length must be 15. Input sequence of intergerized tokens is embedded to 256 dimentions. Then there's a simple RNN on 1 layer of bidirectional LSTM with 2048 units.
    
Input shape: 

    - (None, 15) - sequence of integerized tokens

Outputs shape: 

    - (None, 15, 2048) - main output sequence
    - (None, 2048) - hidden state of LSTM (sum of both directions)
    - (None, 2048) - cell (memory) of LSTM (sum of both directions)

#### Decoder
Encoder's hidden and cell states acts as an initial state for decoder's LSTM. Token <START> after embeding to 256 dims (LSTM input) is the first input for LSTM layer with 2048 units. Attention layer returns weighted average of main encoder output, there weights are calculated based on hidden state and encoder main output. Attention output is concatenated to decoder's LSTM input. LSTM output is passed through fully connected layer with units = decoder's vocabulary size.
Overall, decoder takes current token and predict next token until <END> is predicted.
    
Inputs shape:

    - (None, 1) - decoder input = current token
    - (None, 2048) - hidden state of LSTM on previous step
    - (None, 2048) - cell state of LSTM on previous step
    - (None, 15, 2048) - encoder main output

Outputs shape:

    - (None, decoder_vocab_size) - predictions for every token in decoder's vocabulary
    - (None, 2048) - hidden state of LSTM after iteration
    - (None, 2048) - cell state of LSTM after interation

Overall traffic usage (rough estimation): 140kB (download) for encoder, 140kB (UL) + 85kB(DL) for every token from decoder. Thus TF serving acts as fast local inference server while chatting is recommended to be deployed as a separate server (like flask, jango, etc.).
