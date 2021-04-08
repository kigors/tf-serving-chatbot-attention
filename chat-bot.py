# chatbot script is not for production but for simple testing of tf/serving concept


import sys
import json
import requests
import numpy as np
import re


class Chatbot:

    def __init__(self, temperature=1):
        # load vocab
        base_dir = './exports/'
        vocab_dir = base_dir + 'vocabulary/'

        # load dictionaries
        self.input_token2idx = np.load(vocab_dir + 'input_token2idx.npy', allow_pickle=True).item()
        self.target_token2idx = np.load(vocab_dir + 'target_token2idx.npy', allow_pickle=True).item()
        # load idx2token which is just a numpy array
        self.target_idx2token = np.load(vocab_dir + 'target_idx2token.npy', allow_pickle=True)
        self.TARGET_VOCAB_SIZE = len(self.target_token2idx)
        
        # other inits
        self.server_url = 'http://localhost:8501/v1/models'
        self.MAX_LEN = 15 # fixed length of sequence for encoder
        self.temperature = temperature # temperature for categorical predictions
        

    def prepare_string(self, s):
        """
        Prepair string before tokenization.
        Same prepocessing as for training data, done for consistency.
        """
        # separate punctuation
        s = s.lower()
        s = re.sub(r'[\n]', '', s) # remove \n if any
        s = re.sub(r"([?.!,-])", r" \1 ", s)
        # remove special characters
        s = re.sub(r"(&gt;)|(&lt;)|(&quot;)|([\*~|{}_\[\]`])", '', s)
        # substitute '&' to 'and'
        s = re.sub(r'\s&\s', ' and ', s)
        s = re.sub(r"(?<=\s)'(?=[a-z])|^'(?=[a-z])", "", s) # remove ' at the beginning of the word
        s = re.sub(r"(<[A-Za-z]+>)|(</[A-Za-z]+>)", "", s) # remove html-style tags
        s = re.sub(r"(^\$[0-9]+)|([0-9]+\$)", ' <PRICE> ', s) # replace prices with <PRICE> like $5 or 10$
        s = re.sub(r'([0-9]{1,2}h)|([0-9]{1,2}:[0-9]{2})', ' <TIME> ', s) # time formats like 5h, 10:35 to <TIME>
        s = re.sub(r"(?<=[0-9])(['%])", r" \g<1>", s) # separate % and ' from numbers, like "007's" or "10%"
        s = re.sub(r"[0-9]+-?((th)|(nd)|(st)|(rd))", " <NUM> ", s) # 1st, 2nd, 3rd, 10th or 10-th replace with <NUM>
        s = re.sub(r"(?<=[0-9])([A-Za-z]+)", r' \g<1>', s) # separate units like 33mm 300W
        s = re.sub(r"(#[0-9]+)|([0-9]+)", ' <NUM> ', s) # replace numbers with <NUM>
        # replace double spaces and \t with single space
        s = re.sub(r'[" "\t]+', " ", s).strip()

        return s


    def encoder_inference(self, enc_input):
        """Inference encoder model on TF/serving.
        enc_input is list representation of array of shape (1, 15).
            1 - batch, 15 - max number of tokens in sequence.
        """
        # prepare request
        request_data = json.dumps({
            "signature_name": "serving_default",
            "inputs": {'encoder_input': enc_input}
        })
        headers = {"content-type": "application/json"}

        # make a HTTP request to tensorflwo_model_server
        json_response = requests.post(
            self.server_url + '/encoder:predict',
            data=request_data, headers=headers)

        # process JSON reply
        predictions = json.loads(json_response.text)
        
        if 'outputs' in predictions:
            encoder_outputs = predictions['outputs']
        else:
            raise SystemError(f'Encoder error json: {predictions}')
        
        return encoder_outputs

    def decoder_inference(self, dec_input, h_out, c_out, enc_out):
        """Inference decoder model on TF/serving.
        dec_input is list representation of array of shape (1, 1)
            which is (batch, token).
        The rest of arguments are passed from encoder output."""
        # prepare request
        request_data = json.dumps({
            "signature_name": "serving_default",
            "inputs": {
                'decoder_input': dec_input,
                'h_input': h_out,
                'c_input': c_out,
                'encoder_output': enc_out,
                }
        })
        headers = {"content-type": "application/json"}

        # make a HTTP request to tensorflwo_model_server
        json_response = requests.post(
            self.server_url + '/decoder:predict',
            data=request_data, headers=headers)

        # process JSON reply
        predictions = json.loads(json_response.text)
        if 'outputs' in predictions:
            decoder_outputs = predictions['outputs']
        else:
            raise SystemError(f'Decoder error json: {predictions}')

        return decoder_outputs


    def get_reply(self, sentence):
        input_seq = self.prepare_string(sentence).split(' ')
        
        # trimming or padding input_seq so its length = self.MAX_LEN
        if len(input_seq) > self.MAX_LEN:
            input_seq = input_seq[-self.MAX_LEN:]
        elif len(input_seq) < self.MAX_LEN:
            input_seq += ['<PAD>' for _ in range(self.MAX_LEN-len(input_seq))]
        
        # convert to integers, shape=(1,15) in list representation
        input_seq = [[self.input_token2idx.get(token, self.input_token2idx['<UNK>']) for token in input_seq]]
        
        result = ''
        # encoder inference
        encoder_outputs = self.encoder_inference(input_seq)
        enc_out = encoder_outputs['enc_out']
        h_out = encoder_outputs['h_output']
        c_out = encoder_outputs['c_output']

        dec_input = [[self.target_token2idx['<START>']]] # token <START>

        # limit max output length to avoid infinitely-cycled outputs if any.
        # Model was trained on max decoder length = 16 (15 tokens + <END>).
        for _ in range(16): 

            # decoder inference
            decoder_outputs = self.decoder_inference(dec_input, h_out, c_out, enc_out)
            predictions = np.array(decoder_outputs['decoder_output'])
            h_out = decoder_outputs['h_output']
            c_out = decoder_outputs['c_output']

            # get a sample from categorical distribution of model's prediction
            # predictions are logits, less logits -> less probability difference between different tokens
            predictions = predictions[0] / self.temperature 
            pred_softmax = np.exp(predictions)
            pred_softmax /= np.sum(pred_softmax)
            predicted_id = np.random.choice(range(self.TARGET_VOCAB_SIZE), p=pred_softmax)

            if self.target_idx2token[predicted_id] == '<END>':
                break

            result += self.target_idx2token[predicted_id] + ' '

            # the predicted ID is fed back into the model
            dec_input = [[int(predicted_id)]]

        result = re.sub(r"\s([?.!,-])", r"\1", result) # remove spaces before puctuation

        return result


def test():
    bot = Chatbot()
    print(bot.get_reply('Hi, Elena. How are you?'))


def chatting():
    bot = Chatbot(temperature=1.0)
    
    dialog = []
    print('Program: Type Q to exit.')
    message = input('User: ')
    dialog.append(message)

    while message != 'Q':
        # send last 4 utterances of the dialog (cut to the last 15 tokens in get_reply function)
        answer = bot.get_reply(" ".join(dialog[-4:]))
        print(f'Bot: {answer}')
        dialog.append(answer)
        
        message = input('User: ')
        dialog.append(message)
    print('Program: Bye-bye :)')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # any argument triggers simple test
        test()
    else:
        # interactive chat
        chatting()

