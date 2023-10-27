# Changing Server Models

All the server code can be found in `webrtc/server.py`

In order to change the livestream processing to your own, please change the `process_frames` method. You may change the way that the input frames are processed in any way. The only requirement are that the final processed frames are added to `self.out_frame_buffer`.

When changing the `process_frames` method, keep in mind that:
- The method is called asynchronously whenever the input buffer fills up with four frames. This number of frames can be changed in the `background_process_frames` method
- You have access to the full input frame buffer `self.in_frame_buffer`
- You may write to the output frame buffer `self.out_frame_buffer`. These frames will be dispatched to the client on a separate thread whenever possible.
