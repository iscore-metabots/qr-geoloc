#!/bin/bash

# Camera index chosen
cam=1

# Calibrate the camera based on the image previously captured
cd ../chess-calib
./chess-calib.xc $cam ../tmp/chess-data.yml ../tmp/calib-data.yml

# Access the camera's video stream and process it based on the calibration and scene data
cd ../qr-track
./qr-track.xc ../tmp/calib-data.yml ../tmp/scn-data.yml $cam