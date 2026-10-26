Synthetic clips the node tests read (made with ffmpeg, which the tests themselves do not need):

- `rotated-90.mp4`: 1 s of 64x32 H.264 at 10 fps, the left half red and the right half blue, stored with a 90° display
  rotation, as phone recordings are. Players and ffmpeg show it 32x64, blue on top and red below.

  ```sh
  ffmpeg -f lavfi -i color=red:size=32x32:rate=10:duration=1 -f lavfi -i color=blue:size=32x32:rate=10:duration=1 \
    -filter_complex hstack -c:v libx264 -pix_fmt yuv420p -map_metadata -1 -bitexact flat.mp4
  ffmpeg -display_rotation 90 -i flat.mp4 -c copy -map_metadata -1 -bitexact rotated-90.mp4
  ```
