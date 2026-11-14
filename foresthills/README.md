# Forest Hills flyover and yardage book

A hole-by-hole flyover video and a pocket yardage book for the Championship Course at
Forest Hills Country Club, Clarkson Valley, Missouri, from the blue tees (6,356 yards,
par 72, 71.5 / 132). Both are built from the same public data.

- `out/flyover.mp4`: 1080p flyover, title shot then holes 1 to 18
- `out/yardage_book.pdf`: 4.25 x 7 in pages, a hole map and a green page per hole

## Rebuild

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/fetch_data.py      # OSM, USGS elevation, NAIP imagery
.venv/bin/python scripts/fetch_lidar.py     # USGS lidar point cloud -> surface model
.venv/bin/python scripts/build_course.py    # data/derived/course.json
.venv/bin/python scripts/build_surface.py   # trees and buildings
.venv/bin/python scripts/build_texture.py   # ground texture for the video
.venv/bin/python scripts/yardage_book.py    # out/yardage_book.pdf
.venv/bin/python scripts/render_flyover.py  # out/flyover.mp4
```

`render_flyover.py --stills --holes 1,7` writes test frames; `--holes 7` renders one hole.
`yardage_book.py --holes 7 --png` builds sample pages. The renderer needs a display (it
uses VTK through pyvista) and takes a few minutes on a laptop.

## Data

| What | Source |
| --- | --- |
| Lines of play, greens, tees, bunkers, fairways, water, cart paths | OpenStreetMap (club relation 13333847), via Overpass |
| Scorecard: par, handicap, blue yardages, rating and slope | `course_config.json`, from the GolfPass listing of the club card |
| Ground elevation, 1 m | USGS 3DEP |
| Aerial imagery, 0.5 m, with near-infrared | USDA NAIP via the USGS image service |
| Tree and building heights | USGS lidar, project MO_StLouis_2012 (the only one covering the course) |

Everything is in EPSG:26915 (UTM 15N, metres). `data/derived/course.json` is the course
model both outputs read: per hole the blue tee point, line of play, green with front,
centre and back, hazards with reach and carry, and elevations.

## Known limits

- The blue tee on each hole is the mapped tee box that best matches the card yardage.
  Measured lengths agree with the card to within 2 yards on every hole.
- The lidar is from 2012. Trees are checked against current imagery, so removed trees
  mostly drop out, but trees planted since are missing and green contours predate any
  later rebuild.
- "Plays like" is the card yardage plus 1 yard per 3 ft of climb, nothing more.
