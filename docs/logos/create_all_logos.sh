#!/bin/bash

target_dir="../_static/logo/"

# For-loop over values in the sources array
sources=("challenges" "bench" "combined")
for source in "${sources[@]}"
do
  inkscape --export-type="png,svg" --export-id="logo-with-text" --export-id-only --export-dpi=250 "${source}.svg" --export-filename="${source}_logo-with-text"
  inkscape --export-type="png,svg" --export-id="logo" --export-id-only "${source}.svg" --export-filename="${source}_logo"
  mkdir -p "${target_dir}"
  mv "${source}_logo.png" "${target_dir}${source}_logo.png"
  mv "${source}_logo.svg" "${target_dir}${source}_logo.svg"
  mv "${source}_logo-with-text.png" "${target_dir}${source}_logo_with_text.png"
  mv "${source}_logo-with-text.svg" "${target_dir}${source}_logo_with_text.svg"
  convert "${target_dir}${source}_logo.png" -define icon:auto-resize=64,48,32,16 "${target_dir}${source}.ico"
done
