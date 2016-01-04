#!/bin/bash

cd cap-calib
./cap-calib.xc 1 ../tmp/cap.png

cd ../chess-calib
./chess-calib.xc ../tmp/cap.png ../tmp/chess-data.yml ../tmp/calib-data.yml

cd ../qr-track
./qr-track.xc ../tmp/calib-data.yml ../tmp/scn-data.yml 1