import os
import io
import re
import json
import logging
import boto3
import email
import urllib.parse
import datetime
import re
from botocore.exceptions import ClientError
# from sms_spam_classifier_utilities import one_hot_encode
# from sms_spam_classifier_utilities import vectorize_sequences
# import one_hot_encode
# import vectorize_sequences

import string
import sys
import numpy as np

from hashlib import md5

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')

if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans
   
def vectorize_sequences(sequences, vocabulary_length):
    results = np.zeros((len(sequences), vocabulary_length))
    for i, sequence in enumerate(sequences):
       results[i, sequence] = 1.
    return results

def one_hot_encode(messages, vocabulary_length):
    data = []
    for msg in messages:
        temp = one_hot(msg, vocabulary_length)
        data.append(temp)
    return data

def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    """Converts a text to a sequence of words (or tokens).
    # Arguments
        text: Input text (string).
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to convert the input to lowercase.
        split: str. Separator for word splitting.
    # Returns
        A list of words (or tokens).
    """
    if lower:
        text = text.lower()

    if sys.version_info < (3,):
        if isinstance(text, unicode):
            translate_map = dict((ord(c), unicode(split)) for c in filters)
            text = text.translate(translate_map)
        elif len(split) == 1:
            translate_map = maketrans(filters, split * len(filters))
            text = text.translate(translate_map)
        else:
            for c in filters:
                text = text.replace(c, split)
    else:
        translate_dict = dict((c, split) for c in filters)
        translate_map = maketrans(translate_dict)
        text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]

def one_hot(text, n,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' '):
    """One-hot encodes a text into a list of word indexes of size n.
    This is a wrapper to the `hashing_trick` function using `hash` as the
    hashing function; unicity of word to index mapping non-guaranteed.
    # Arguments
        text: Input text (string).
        n: int. Size of vocabulary.
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to set the text to lowercase.
        split: str. Separator for word splitting.
    # Returns
        List of integers in [1, n]. Each integer encodes a word
        (unicity non-guaranteed).
    """
    return hashing_trick(text, n,
                         hash_function='md5',
                         filters=filters,
                         lower=lower,
                         split=split)


def hashing_trick(text, n,
                  hash_function=None,
                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                  lower=True,
                  split=' '):
    """Converts a text to a sequence of indexes in a fixed-size hashing space.
    # Arguments
        text: Input text (string).
        n: Dimension of the hashing space.
        hash_function: defaults to python `hash` function, can be 'md5' or
            any function that takes in input a string and returns a int.
            Note that 'hash' is not a stable hashing function, so
            it is not consistent across different runs, while 'md5'
            is a stable hashing function.
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to set the text to lowercase.
        split: str. Separator for word splitting.
    # Returns
        A list of integer word indices (unicity non-guaranteed).
    `0` is a reserved index that won't be assigned to any word.
    Two or more words may be assigned to the same index, due to possible
    collisions by the hashing function.
    The [probability](
        https://en.wikipedia.org/wiki/Birthday_problem#Probability_table)
    of a collision is in relation to the dimension of the hashing space and
    the number of distinct objects.
    """
    if hash_function is None:
        hash_function = hash
    elif hash_function == 'md5':
        hash_function = lambda w: int(md5(w.encode()).hexdigest(), 16)

    seq = text_to_word_sequence(text,
                                filters=filters,
                                lower=lower,
                                split=split)
    return [int(hash_function(w) % (n - 1) + 1) for w in seq]

def lambda_handler(event, context):
   
    logger.info('printing event')
    logger.info("event: {}".format(event))
   
   
    bucket = event['Records'][0]['s3']['bucket']['name']
    print("bucket  ", bucket)
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    print("key  ", key)
    #data = s3.get_object(Bucket='emails1asgn3', Key='j2aiunf80sq7k96bhv6cr3v7h1kiqsd1a96dto01')
    data = s3.get_object(Bucket=bucket,Key= key)
    contents = data['Body'].read()
    msg = email.message_from_bytes(contents)
   

    # ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
    ENDPOINT_NAME = "sms-spam-classifier-mxnet-2022-11-23-00-45-18-135"
    runtime= boto3.client('runtime.sagemaker')  
   
    payload = ""
   
    if msg.is_multipart():
        print("multi part")
        for part in msg.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get('Content-Disposition'))

        # skip any text/plain (txt) attachments
            if ctype == 'text/plain' and 'attachment' not in cdispo:
                payload1 = part.get_payload(decode=False)
                payload = part.get_payload(decode=True)  # decode
                print("multi part", payload)
                print("payload1: ", payload1)
                break
    else:
        #print("msg payload is = ", msg.get_payload())
        payload = msg.get_payload()
       
    print("payload str: {}".format(payload))
    print("payload is ", payload.decode("utf-8"))
    payload = payload.decode("utf-8")
    #re.sub('\s', " " , payload)
    payload = payload.replace('\r\n',' ').strip()
    #payload = "Write a short story in a weekend: An online workshop with Stuart Evers Though short stories have fewer words than long-form fiction, they are by no means easier to write. In this practical two-day workshop with award-winning author Stuart Evers, you will discover how to utilise economy of language without sacrificing detail and depth, to gain the confidence and tools to craft your"
    payloadtext = payload
   
    vocabulary_length = 9013
    test_messages = [payload]
    #test_messages = ["FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! ubscribe6GBP/ mnth inc 3hrs 16 stop?txtStop"]
    one_hot_test_messages = one_hot_encode(test_messages, vocabulary_length)
    encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)
    payload = json.dumps(encoded_test_messages.tolist())
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,ContentType='application/json',Body=payload)
   
    response_body = response['Body'].read().decode('utf-8')
    result = json.loads(response_body)
    print(result)
    pred = int(result['predicted_label'][0][0])
    if pred == 1:
        CLASSIFICATION = "SPAM"
    elif pred == 0:
        CLASSIFICATION = "NOT SPAM"
    CLASSIFICATION_CONFIDENCE_SCORE = str(float(result['predicted_probability'][0][0]) * 100)
   
   
    #########################################################################################################
    SENDER = "vn2065@nyu.edu"
    RECIPIENT = msg['From']
    EMAIL_RECEIVE_DATE = msg["Date"]
    EMAIL_SUBJECT = msg["Subject"]
    payloadtext = payloadtext[:240]
    EMAIL_BODY = payloadtext
    AWS_REGION = "us-east-1"

    # The email to send.
    SUBJECT = "Homework Assignment 3"
    BODY_TEXT = "We received your email sent at " + EMAIL_RECEIVE_DATE + " with the subject " + EMAIL_SUBJECT + ".\r\nHere is a 240 character sample of the email body:\r\n" + EMAIL_BODY + "\r\nThe email was categorized as " + CLASSIFICATION + " with a " + CLASSIFICATION_CONFIDENCE_SCORE + "% confidence."
    CHARSET = "UTF-8"
    client = boto3.client('ses',region_name=AWS_REGION)
   
    # Try to send the email.
    try:
        #Provide the contents of the email.
        response = client.send_email(
            Destination={
                'ToAddresses': [
                    RECIPIENT,
                ],
            },
            Message={
                'Body': {

                    'Text': {
                        'Charset': CHARSET,
                        'Data': BODY_TEXT,
                    },
                },
                'Subject': {
                    'Charset': CHARSET,
                    'Data': SUBJECT,
                },
            },
            Source=SENDER,
           
        )
    # Display an error if something goes wrong.
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        print("Email sent! Message ID:"),
        print(response['MessageId'])    
           
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }

