# DimensionReduction
EFANNA + LargeVis

## HOW TO RUN
### embedding:
```
cmake ./
make
./Vis -knn_k 100 -L 200 -checkK 200 -S 20 -knn_type largevis
./Vis -knn_k 10 -L 30 -checkK 25 -S 10 -knn_type efanna
```

### plot
```
cd plot
python plot.py -input mnist_vec2D.txt -label mnist_label.txt -output mnist_vec2D_plot
```

