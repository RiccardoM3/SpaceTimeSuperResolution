import argparse
import asyncio
import json
import logging
import os
import sys
import ssl
import uuid
import numpy as np
import torch

# Add the project directory to path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
sys.path.append(module_dir)
from model.SRSRTModel import SRSRTModel

from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from av import VideoFrame
from collections import deque

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

host = "0.0.0.0"
port = 9000
model_name = "paper_model_final_pos_enc_tril"

class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform
        self.model = SRSRTModel().to('cuda')
        self.model.load_model(model_name)
        self.in_frame_buffer = deque(maxlen=4)
        self.out_frame_buffer = deque(maxlen=32)
        self.start_processing_frames()

    async def background_process_frames(self):
        while True:
            await self.process_frames()

    def start_processing_frames(self):
        asyncio.create_task(self.background_process_frames())

    async def process_frames(self):
        frame = await self.track.recv()

        if self.transform == "superresolution":
            self.in_frame_buffer.append(frame)

            if len(self.in_frame_buffer) >= 4:
                in_frames = list(self.in_frame_buffer)
                self.in_frame_buffer.clear()
                
                context = np.stack([frame.to_ndarray(format="bgr24") for frame in in_frames], axis=0)
                context = torch.tensor(context).to('cuda')
                context = context.unsqueeze(0)
                context = context.permute(0, 1, 4, 2, 3)
                context = context / 255

                for i in range(len(in_frames)-1):
                    j = i+1

                    input_images = context[:, i:j+1, :, :, :]
                    output_images = self.model(context, input_images, (i, j), skip_encoder=(i!=0))

                    output_images = output_images.permute(0, 1, 3, 4, 2)
                    output_frame_1 = VideoFrame.from_ndarray((output_images[0, 0] * 255).cpu().detach().numpy().astype(np.uint8), format="bgr24")
                    output_frame_2 = VideoFrame.from_ndarray((output_images[0, 1] * 255).cpu().detach().numpy().astype(np.uint8), format="bgr24")

                    # preserve timing information
                    output_frame_1.pts = in_frames[i].pts
                    output_frame_1.time_base = in_frames[i].time_base
                    output_frame_2.pts = (in_frames[i].pts + in_frames[j].pts)/2
                    output_frame_2.time_base = in_frames[j].time_base

                    self.out_frame_buffer.append(output_frame_1)
                    self.out_frame_buffer.append(output_frame_2)

                    print(len(self.out_frame_buffer))
        else:
            self.out_frame_buffer.append(frame)

    async def recv(self):
        while len(self.out_frame_buffer) < 1:
            await asyncio.sleep(0.01)

        return self.out_frame_buffer.popleft()



async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            pc.addTrack(track)
        elif track.kind == "video":
            pc.addTrack(
                VideoTransformTrack(
                    relay.subscribe(track), transform=params["video_transform"]
                )
            )

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)

    # handle offer
    await pc.setRemoteDescription(offer)

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default=host, help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=port, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--record-to", help="Write received media to a file."),
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
