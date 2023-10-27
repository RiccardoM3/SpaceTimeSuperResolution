# WebRTC Server Setup

Note: Everything in this file is all relative to the `webrtc` directory.

The server allows users to connect and share audio/video at a chosen resolution. The frames are sent to the server at 0.25 the resolution and 0.5x the FPS. The server runs a machine learning model to reconstruct the frames before sending them back to the client.

## Installation
First run all installation instructions in the `docs/run_model.md` file.

Next, continue with the below instructions


```console
# Change to the webrtc directory
cd webrtc

# Install webrtc server packages
pip install -r requirements.txt
```

## Running

You may run the server on `http://127.0.0.1:9000` using with the below command


`python server.py`

To enable verbose logging, you can add the `-v` option, as seen below


`python server.py -v`

To change the host or post, you can use the following options


`python server.py -v --host=192.168.1.10 --port=9001`

If you wish to host the server via SSL, follow the below procedure:

```console
# First create a certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365

# Now run the server with the certificate
python ./server.py -v --host=192.168.1.10 --port=9001 --cert-file cert.pem --key-file key.pem
```