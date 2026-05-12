#!/bin/bash

echo "========================================"
echo " Resetting CFDEM simulation"
echo "========================================"

# -----------------------------
# OpenFOAM cleanup
# -----------------------------

echo "Removing OpenFOAM time directories..."

find CFD -maxdepth 1 -type d \
    \( -regex '.*/[0-9.]+' \) \
    ! -name "0" \
    -exec rm -rf {} +

# Remove processor directories
echo "Removing processor directories..."
rm -rf CFD/processor*
rm -rf CFD/dynamicCode

# Remove OpenFOAM logs
echo "Cleaning OpenFOAM logs..."
rm -f CFD/log*
rm -f CFD/*.log

# -----------------------------
# DEM cleanup
# -----------------------------

echo "Cleaning DEM post-processing files..."

rm -f DEM/post/*.vtk
rm -f DEM/post/*.stl
rm -f DEM/post/*.dump
rm -f DEM/post/*.txt

# Remove restart files
echo "Removing DEM restart files..."
rm -rf DEM/post/restart/*

# Remove LIGGGHTS logs
rm -f DEM/*.liggghts
rm -f DEM/log*
rm -f DEM/*.txt

# -----------------------------
# Coupling cleanup
# -----------------------------

echo "Removing coupling files..."

rm -rf couplingFiles/*
rm -rf CFD/constant/couplingFiles/*

# Remove mesh
echo "Removing polyMesh..."
rm -rf CFD/constant/polyMesh

# -----------------------------
# Finished
# -----------------------------

echo "========================================"
echo " Reset complete"
echo "========================================"
