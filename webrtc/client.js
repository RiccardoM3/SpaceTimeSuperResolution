var options = {
    video: {
        transform: "superresolution", // ["none", "superresolution"]
        codec: "H264/90000", // ["default", "VP8/90000", "H264/90000"]
        fps: 10
    },
    audio: {
        codec: "default" // ["default", "opus/48000/2", "PCMU/8000", "PCMA/8000"]
    },
    datachannel: {
        enabled: true,
        settings: {"ordered": true} // ["", {"ordered": true}, {"ordered": false, "maxRetransmits": 0}, {"ordered": false, "maxPacketLifetime": 500}]
    },
    stun: "" // ["", "stun:stun.l.google.com:19302"]
};

// get DOM elements
var dataChannelLog = document.getElementById('data-channel'),
    iceConnectionLog = document.getElementById('ice-connection-state'),
    iceGatheringLog = document.getElementById('ice-gathering-state'),
    signalingLog = document.getElementById('signaling-state');

// peer connection
var pc = null;

// data channel
var dc = null, dcInterval = null;

// video and audio streams
var videoStream = null;

function createPeerConnection() {
    var config = {
        sdpSemantics: 'unified-plan'
    };

    if (options.stun) {
        config.iceServers = [{urls: [options.stun]}];
    }

    pc = new RTCPeerConnection(config);

    // register some listeners to help debugging
    pc.addEventListener('icegatheringstatechange', function() {
        iceGatheringLog.textContent += ' -> ' + pc.iceGatheringState;
    }, false);
    iceGatheringLog.textContent = pc.iceGatheringState;

    pc.addEventListener('iceconnectionstatechange', function() {
        iceConnectionLog.textContent += ' -> ' + pc.iceConnectionState;
    }, false);
    iceConnectionLog.textContent = pc.iceConnectionState;

    pc.addEventListener('signalingstatechange', function() {
        signalingLog.textContent += ' -> ' + pc.signalingState;
    }, false);
    signalingLog.textContent = pc.signalingState;

    // connect audio / video
    pc.addEventListener('track', function(evt) {        
        if (evt.track.kind == 'video') {
            let id=`remote-video-${evt.track.id}`;
            document.getElementById('media-container').innerHTML += 
                `<div style="width: 384; height: 256; margin: 16px;" >` + 
                    `<video id="${id}" class="w-100 h-100" autoplay="true" playsinline="true"></video>` +
                `</div>`;
            document.getElementById(id).srcObject = evt.streams[0];
        } else {
            let id=`remote-audio-${evt.track.id}`;
            document.getElementById('media-container').innerHTML += 
                `<audio id="${id}" class="w-100 h-100" autoplay="true"></audio>`;
            document.getElementById(id).srcObject = evt.streams[0];
        }
    });

    return pc;
}

function negotiate() {
    return pc.createOffer({offerToReceiveAudio:true, offerToReceiveVideo:true}).then(function(offer) {
        return pc.setLocalDescription(offer);
    }).then(function() {
        // wait for ICE gathering to complete
        return new Promise(function(resolve) {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                function checkState() {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                }
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(function() {
        var offer = pc.localDescription;
        var codec;

        codec = options.audio.codec;
        if (codec !== 'default') {
            offer.sdp = sdpFilterCodec('audio', codec, offer.sdp);
        }

        codec = options.video.codec;
        if (codec !== 'default') {
            offer.sdp = sdpFilterCodec('video', codec, offer.sdp);
        }

        document.getElementById('offer-sdp').textContent = offer.sdp;
        return fetch('/offer', {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
                video_transform: options.video.transform
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then(function(response) {
        return response.json();
    }).then(function(answer) {
        document.getElementById('answer-sdp').textContent = answer.sdp;
        return pc.setRemoteDescription(answer);
    }).catch(function(e) {
        alert(e);
    });
}

function start() {
    document.getElementById('start').style.display = 'none';

    pc = createPeerConnection();

    var time_start = null;

    function current_stamp() {
        if (time_start === null) {
            time_start = new Date().getTime();
            return 0;
        } else {
            return new Date().getTime() - time_start;
        }
    }

    if (options.datachannel.enabled) {
        var parameters = options.datachannel.settings;

        dc = pc.createDataChannel('chat', parameters);
        dc.onclose = function() {
            clearInterval(dcInterval);
            dataChannelLog.textContent += '- close\n';
        };
        dc.onopen = function() {
            dataChannelLog.textContent += '- open\n';
            dcInterval = setInterval(function() {
                var message = 'ping ' + current_stamp();
                dataChannelLog.textContent += '> ' + message + '\n';
                dc.send(message);
            }, 1000);

            dc.send("subscribe_to_videos")
        };
        dc.onmessage = function(evt) {
            dataChannelLog.textContent += '< ' + evt.data + '\n';

            if (evt.data.substring(0, 4) === 'pong') {
                var elapsed_ms = current_stamp() - parseInt(evt.data.substring(5), 10);
                dataChannelLog.textContent += ' RTT ' + elapsed_ms + ' ms\n';
            } else if (evt.data.substring(0, 6) === 'offer:') {
                const offer = JSON.parse(evt.data.substring(6))
                console.log(offer)
                pc.setRemoteDescription(offer)

                answer = pc.createAnswer()
                pc.setLocalDescription(answer)
                
                dc.send('answer:' + JSON.stringify({'sdp': pc.localDescription.sdp, 'type': pc.localDescription.type}))
            }
        };
    }

    var constraints = {
        audio: true,
        video: true
    };

    const resolutionSelector = document.getElementById("resolution")
    let resolution = resolutionSelector.value;
    if (resolution != "") {
        resolution = resolution.split('x');
        constraints.video = {
            width: parseInt(resolution[0], 0),
            height: parseInt(resolution[1], 0),
            frameRate: {max: options.video.fps}
        };
    } else {
        constraints.audio = false
        constraints.video = false
    }

    resolutionSelector.addEventListener('change', function() {
        let resolution = resolutionSelector.value;
        resolution = resolution.split('x');
        changeResolution(resolution[0], resolution[1]);
    })

    if (constraints.audio || constraints.video) {
        navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
            videoStream = stream;

            stream.getTracks().forEach(function(track) {
                pc.addTrack(track, stream);
            });
            document.getElementById('local-video').srcObject = stream;
            return negotiate();
        }, function(err) {
            alert('Could not acquire media: ' + err);
        });
    } else {
        // // create empty tracks 
        // const stream1 = new MediaStream();
        // const audioTrack1 = new MediaStreamTrack();
        // audioTrack1.kind = 'audio'
        // const videoTrack1 = new MediaStreamTrack();
        // videoTrack1.kind = 'video'
        // stream1.addTrack(audioTrack1);
        // stream1.addTrack(videoTrack1);
        // peerConnection.addTrack(audioTrack1, stream1);
        // peerConnection.addTrack(videoTrack1, stream1);

        // const stream2 = new MediaStream();
        // const audioTrack2 = new MediaStreamTrack();
        // audioTrack2.kind = 'audio'
        // const videoTrack2 = new MediaStreamTrack();
        // videoTrack2.kind = 'audio'
        // stream2.addTrack(audioTrack2);
        // stream2.addTrack(videoTrack2);
        // peerConnection.addTrack(audioTrack2, stream2);
        // peerConnection.addTrack(videoTrack2, stream2);

        negotiate();
    }

    document.getElementById('stop').style.display = 'inline-block';
}

function stop() {
    videoStream = null;
    
    document.getElementById('start').style.display = 'inline-block';
    document.getElementById('stop').style.display = 'none';

    // close data channel
    if (dc) {
        dc.close();
    }

    // close transceivers
    if (pc.getTransceivers) {
        pc.getTransceivers().forEach(function(transceiver) {
            if (transceiver.stop) {
                transceiver.stop();
            }
        });
    }

    // close local audio / video
    pc.getSenders().forEach(function(sender) {
        sender.track.stop();
    });

    // close peer connection
    setTimeout(function() {
        pc.close();
    }, 500);
}

function sdpFilterCodec(kind, codec, realSdp) {
    var allowed = []
    var rtxRegex = new RegExp('a=fmtp:(\\d+) apt=(\\d+)\r$');
    var codecRegex = new RegExp('a=rtpmap:([0-9]+) ' + escapeRegExp(codec))
    var videoRegex = new RegExp('(m=' + kind + ' .*?)( ([0-9]+))*\\s*$')
    
    var lines = realSdp.split('\n');

    var isKind = false;
    for (var i = 0; i < lines.length; i++) {
        if (lines[i].startsWith('m=' + kind + ' ')) {
            isKind = true;
        } else if (lines[i].startsWith('m=')) {
            isKind = false;
        }

        if (isKind) {
            var match = lines[i].match(codecRegex);
            if (match) {
                allowed.push(parseInt(match[1]));
            }

            match = lines[i].match(rtxRegex);
            if (match && allowed.includes(parseInt(match[2]))) {
                allowed.push(parseInt(match[1]));
            }
        }
    }

    var skipRegex = 'a=(fmtp|rtcp-fb|rtpmap):([0-9]+)';
    var sdp = '';

    isKind = false;
    for (var i = 0; i < lines.length; i++) {
        if (lines[i].startsWith('m=' + kind + ' ')) {
            isKind = true;
        } else if (lines[i].startsWith('m=')) {
            isKind = false;
        }

        if (isKind) {
            var skipMatch = lines[i].match(skipRegex);
            if (skipMatch && !allowed.includes(parseInt(skipMatch[2]))) {
                continue;
            } else if (lines[i].match(videoRegex)) {
                sdp += lines[i].replace(videoRegex, '$1 ' + allowed.join(' ')) + '\n';
            } else {
                sdp += lines[i] + '\n';
            }
        } else {
            sdp += lines[i] + '\n';
        }
    }

    return sdp;
}

function changeResolution(width, height) {
    if (videoStream == null) {
        return;
    }
    
    const newConstraints = {
        width: { ideal: width }, // Set your desired width
        height: { ideal: height }   // Set your desired height
    };

    videoTrack = videoStream.getVideoTracks()[0];
    videoTrack.applyConstraints(newConstraints)
    .then(() => {
      // The resolution has been updated
      console.log('Resolution updated successfully');
    })
    .catch((error) => {
      console.error('Error updating resolution:', error);
    });

}

function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // $& means the whole matched string
}
