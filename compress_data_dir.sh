#!/bin/bash

dir=data/$1

tar -cf - $dir | pigz -9 > $1.tar.gz