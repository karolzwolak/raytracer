Simple raytracer that renders yaml scenes. Supports basic shapes and materials and .obj models

Command line usage:

```
Simple raytracer that renders yaml scenes. Supports basic shapes and materials and .obj models. Can render single images and animations

Usage: raytracer [OPTIONS] <SCENE_FILE> <COMMAND>

Commands:
  image    Render a single image
  animate  Render an animation. Use `animate` field on an object to add animation to it
  help     Print this message or the help of the given subcommand(s)

Arguments:
  <SCENE_FILE>  The scene file to render

Options:
  -o, --output-path <OUTPUT_PATH>
          The output path of the rendered image. By default it's `./<scene_filename>.<image_format>`
  -w, --width <WIDTH>
          Width (in pixels) of the output image.
          Overrides the one in the scene file. If not specified anywhere, defaults to 800
  -h, --height <HEIGHT>
          Height (in pixels) of the output image.
          Overrides the one in the scene file. If not specified anywhere, defaults to 800
      --fov <FOV>
          Field of view of the camera in radians. Overrides the one in the scene file. If not specified anywhere, defaults to π/3
  -d, --depth <DEPTH>
          Maximum number of times a ray can bounce (change direction). Direction change occurs when a ray hits a reflective or refractive surface. Overrides the one in the scene file
  -s, --supersampling-level <SUPERSAMPLING_LEVEL>
          Controls how many rays are shot per pixel. In other words, the quality of the anti-aliasing (supersampling). Overrides the one in the scene file
  -h, --help
          Print help
```

Rendering animations:

```
Render an animation. Use `animate` field on an object to add animation to it

Usage: raytracer <SCENE_FILE> animate [OPTIONS] --duration-sec <DURATION_SEC>

Options:
  -f, --format <FORMAT>              The format of the output video [default: mp4] [possible values: gif, mp4]
  -d, --duration-sec <DURATION_SEC>  The duration of the output video in seconds
      --fps <FPS>                    Frames per second of the output video. Note that not all formats support all framerates. Use lower framerates when rendering to gif (about 30) [default: 60]
  -h, --help                         Print help
```

Rendering single images:

```
Render a single image

Usage: raytracer <SCENE_FILE> image [OPTIONS]

Options:
  -f, --format <FORMAT>  The format of the output image [default: png] [possible values: ppm, png]
  -h, --help             Print help
```

## Sample images

- Rotating dragon ![Rotating dragon](samples/renders/rotating_dragon.webp)
- Cover image render ![Cover image render](samples/renders/cover.png)
- Refraction Chapter![Refraction chapter](samples/renders/refractions.png)
- Cubes Chapter![Cubes chapter](samples/renders/cubes.png)
- Dragons ![Dragons](samples/renders/dragons.png)
