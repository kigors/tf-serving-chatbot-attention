# tf-serving-chatbot-attention
### Description
The primary goal of this repository is to demonstrate tensorflow serving a chat bot model based on encoder-decoder design with attention. Model was built on tensorflow 2.4.1 following ideas in [tensorflow tutorial](https://www.tensorflow.org/tutorials/text/nmt_with_attention) and trained using [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html).
### Dependencies
- For simple chat-bot client: built-in python libraries: sys, json, request, re; external pythons library: numpy
- Model creation: tensorflow to create saved_model export based on trained weights.
- Serving: tensorflow_model_server installed on machine.
### Instructions
1. Download weights from <TBD-link> and unzip in project folder
2. Run test.ipynb to create tensorflow models (encoder and decoder), load pre-trained weights for every layer, save model in saved_model (tf) format apropriate for tensorflow_model_server.
3. Run tensorflow_model_server as in serving_command.txt
4. Once server is up and running, srat