const socket = io();

const localVideo = document.getElementById('localVideo');
const remoteVideo = document.getElementById('remoteVideo');

let peerConnection;

const configuration = { iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] };

socket.on('connect', () => {
  navigator.mediaDevices
    .getUserMedia({ video: true })
    .then((stream) => {
      localVideo.srcObject = stream;
      console.log(stream);
      peerConnection = new RTCPeerConnection(configuration);
      stream.getTracks().forEach((track) => peerConnection.addTrack(track, stream));

      peerConnection.createOffer()
        .then((offer) => peerConnection.setLocalDescription(offer))
        .then(() => {
          socket.emit('offer', peerConnection.localDescription, 'room-123');
        })
        .catch((error) => console.error('Error creating offer:', error));
    })
    .catch((error) => {
      console.error('Error accessing media devices:', error);
    });
});

navigator.mediaDevices.getUserMedia({ video: true })
  .then((stream) => {
    localVideo.srcObject = stream;
    console.log(stream);
    socket.emit('offer', stream, 'room-123');
  })
  .catch((error) => {
    console.error('Error accessing media devices:', error);
  });

socket.on('offer', (offer) => {
  peerConnection = new RTCPeerConnection();

  console.log(offer)

  peerConnection.setRemoteDescription(new RTCSessionDescription(offer))
    .then(() => peerConnection.createAnswer())
    .then((answer) => peerConnection.setLocalDescription(answer))
    .then(() => socket.emit('answer', peerConnection.localDescription, 'room-123'))
    .catch((error) => console.error('Error creating answer:', error));

  peerConnection.onicecandidate = (event) => {
    if (event.candidate) {
      socket.emit('ice-candidate', event.candidate, 'room-123');
    }
  };

  peerConnection.ontrack = (event) => {
    remoteVideo.srcObject = event.streams[0];
    // setTimeout(() => {
    //   console.log('reducing')
      
    //   const remoteVideoTrack = remoteVideo.srcObject.getVideoTracks()[0];

    //   const newConstraints = {
    //     width: { ideal: 640/2 },   // Set your desired width
    //     height: { ideal: 480/2 }   // Set your desired height
    //   };

    //   remoteVideoTrack.applyConstraints(newConstraints)
    //     .then(() => {
    //       console.log('Remote video resolution changed successfully');
    //     })
    //     .catch((error) => {
    //       console.error('Error changing remote video resolution:', error);
    //     });
    // }, 10000)
  };
});

socket.on('answer', (answer) => {
  peerConnection
    .setRemoteDescription(new RTCSessionDescription(answer))
    .catch((error) => console.error('Error setting remote description:', error));
});

socket.on('ice-candidate', (iceCandidate) => {
  peerConnection
    .addIceCandidate(new RTCIceCandidate(iceCandidate))
    .catch((error) => console.error('Error adding ICE candidate:', error));
});

socket.on('user-disconnected', (userId) => {
  if (peerConnection) {
    peerConnection.close();
  }
  remoteVideo.srcObject = null;
});

window.onbeforeunload = () => {
  socket.close();
};