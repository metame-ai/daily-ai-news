#!/bin/bash
set -e
set -x
date=$1
cd ./$2
git add .
git commit -m"update $date"
git push origin main
echo "------ done ------"
