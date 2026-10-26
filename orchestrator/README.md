# hyperstudy-perception

Turn a video into a time-indexed **perception track**: shots and keyframes, a word-timed transcript, face tracks
clustered into people, and shot captions. The result is a versioned track bundle that the
[HyperStudy Annotator](https://github.com/hyperstudyio/hyperstudy-annotator) imports for review and correction.

```sh
uv tool install hyperstudy-perception
perception doctor                       # what can run on this machine
perception graphs                       # the graphs it can run, by name
perception run full.yaml --input video=clip.mp4 --only asr,faces --allow-downloads
```

Each step runs in its own locked environment, made by [uv](https://docs.astral.sh/uv/) the first time it is needed
(`perception setup` makes them all ahead of time), under `~/.hyperstudy-perception`. Nothing else is needed on the
machine: video is read with PyAV, whose wheels carry FFmpeg.

Model weights are downloaded at run time under their own licences; several (py-feat, ArcFace) are non-commercial, and
each bundle records the most restrictive licence among its layers.
