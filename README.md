# Line Artist
Ever stumbled upon these awesome drawings made with only [a single continuous line](https://www.google.com/search?q=continuous+line)?
This is an attempt to generate drawings in this style from arbitrary images.
Some resulting images:

TODO: alternative link to medium post: https://medium.com/@michellegemmeke/the-art-of-one-line-drawings-8cd8fd5a5af7
TODO: add images

## How is it done?
The approach consists of the following steps:
1. Get edge points by applying the [Canny edge detector](https://en.wikipedia.org/wiki/Canny_edge_detector)
2. Find an ordering which approximates the [TSP](https://en.wikipedia.org/wiki/Travelling_salesman_problem) (currently, [NN](https://en.wikipedia.org/wiki/Nearest_neighbour_algorithm) is used)
3. Fit a [B-spline](https://en.wikipedia.org/wiki/B-spline) through the ordered points
4. Sample the spline and colour pixels on the output image

## Possible improvements
- The TSP approximation with NN is still naively implemented (brute force) and thus incredibly slow. Alternative approaches are being tested (**help welcome!**). Possible alternatives:
  - Some kind of space partitioning should speed things up significantly.
  - Identifying clusters and solving TSP between them, while using NN within the clusters
  - [Another approximation algorithm?](https://en.wikipedia.org/wiki/Travelling_salesman_problem#Computing_a_solution)
    - [Ant colony optimization](https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms) sounds cool
- NN greedily selects points which are suitable in the beginning, but the few remaining scattered points that remain at the end lead to some unnatural lines through the drawing when attempting to connect them
- The visualization of the B-spline in the output image is very limited (resulting images cannot easily modified). An alternative idea is to generate SVG paths. However, these are Bézier curves, so a conversion from B-splines is needed ([Boehm's algorithm](https://www.sciencedirect.com/science/article/pii/0010448580901542)).