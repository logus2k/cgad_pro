## Critical Analysis

### Bottleneck Evolution

As optimizations progress, the computational bottleneck shifts:

#### Backward-Facing Step (XS) - 287 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (68%) | Post-Proc (20%) |
| CPU Threaded | Assembly (51%) | Post-Proc (25%) |
| CPU Multiprocess | Assembly (50%) | Post-Proc (49%) |
| Numba CPU | Solve (55%) | BC (16%) |
| Numba CUDA | Solve (88%) | Assembly (6%) |
| CuPy GPU | Solve (72%) | BC (23%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-0" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":68.1},{"name":"Solve","value":7.8},{"name":"Apply BC","value":0.8},{"name":"Post-Process","value":20.0}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-1" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":1.8},{"name":"Solve","value":72.1},{"name":"Apply BC","value":23.2},{"name":"Post-Process","value":0.5}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-2" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[68.1,50.9,50.3,10.5,6.0,1.8]},{"name":"Solve","data":[7.8,13.5,0.4,55.0,88.2,72.1]},{"name":"Apply BC","data":[0.8,4.1,0.1,15.6,1.4,23.2]},{"name":"Post-Process","data":[20.0,24.6,48.9,0.8,1.8,0.5]}],"yAxisName":"Percentage (%)"}'></div>

#### Backward-Facing Step (M) - 195,362 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (50%) | Solve (36%) |
| CPU Threaded | Solve (56%) | Assembly (28%) |
| CPU Multiprocess | Solve (78%) | Assembly (12%) |
| Numba CPU | Solve (94%) | BC (4%) |
| Numba CUDA | Solve (50%) | Assembly (27%) |
| CuPy GPU | Solve (88%) | BC (12%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-3" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":49.7},{"name":"Solve","value":36.4},{"name":"Apply BC","value":0.3},{"name":"Post-Process","value":13.6}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-4" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":0.2},{"name":"Solve","value":88.1},{"name":"Apply BC","value":11.6},{"name":"Post-Process","value":0.0}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-5" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[49.7,28.0,12.1,1.4,26.8,0.2]},{"name":"Solve","data":[36.4,56.2,77.7,94.4,49.8,88.1]},{"name":"Apply BC","data":[0.3,2.8,3.7,4.2,22.2,11.6]},{"name":"Post-Process","data":[13.6,13.1,6.5,0.0,0.1,0.0]}],"yAxisName":"Percentage (%)"}'></div>

#### Elbow 90° (XS) - 411 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (71%) | Post-Proc (20%) |
| CPU Threaded | Assembly (59%) | Post-Proc (24%) |
| CPU Multiprocess | Assembly (50%) | Post-Proc (49%) |
| Numba CPU | Solve (53%) | BC (22%) |
| Numba CUDA | Solve (87%) | Assembly (6%) |
| CuPy GPU | Solve (74%) | BC (22%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-6" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":70.5},{"name":"Solve","value":6.2},{"name":"Apply BC","value":1.1},{"name":"Post-Process","value":19.6}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-7" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":1.5},{"name":"Solve","value":74.3},{"name":"Apply BC","value":21.9},{"name":"Post-Process","value":0.5}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-8" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[70.5,59.3,50.0,10.4,6.4,1.5]},{"name":"Solve","data":[6.2,9.1,0.5,52.6,87.4,74.3]},{"name":"Apply BC","data":[1.1,4.1,0.2,21.7,1.9,21.9]},{"name":"Post-Process","data":[19.6,24.1,49.2,0.8,1.8,0.5]}],"yAxisName":"Percentage (%)"}'></div>

#### Elbow 90° (M) - 161,984 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (47%) | Solve (40%) |
| CPU Threaded | Solve (51%) | Assembly (31%) |
| CPU Multiprocess | Solve (73%) | Assembly (15%) |
| Numba CPU | Solve (94%) | BC (4%) |
| Numba CUDA | Solve (48%) | Assembly (28%) |
| CuPy GPU | Solve (81%) | BC (19%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-9" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":47.4},{"name":"Solve","value":39.7},{"name":"Apply BC","value":0.3},{"name":"Post-Process","value":12.6}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-10" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":0.2},{"name":"Solve","value":81.1},{"name":"Apply BC","value":18.5},{"name":"Post-Process","value":0.0}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-11" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[47.4,30.8,14.8,1.4,28.0,0.2]},{"name":"Solve","data":[39.7,51.0,72.7,94.4,47.8,81.1]},{"name":"Apply BC","data":[0.3,3.1,4.3,4.1,23.1,18.5]},{"name":"Post-Process","data":[12.6,15.1,8.1,0.0,0.1,0.0]}],"yAxisName":"Percentage (%)"}'></div>

#### S-Bend (XS) - 387 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (69%) | Post-Proc (19%) |
| CPU Threaded | Assembly (50%) | Post-Proc (20%) |
| CPU Multiprocess | Assembly (50%) | Post-Proc (49%) |
| Numba CPU | Solve (52%) | BC (21%) |
| Numba CUDA | Solve (89%) | Assembly (5%) |
| CuPy GPU | Solve (72%) | BC (23%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-12" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":68.8},{"name":"Solve","value":8.0},{"name":"Apply BC","value":1.4},{"name":"Post-Process","value":18.9}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-13" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":1.8},{"name":"Solve","value":72.4},{"name":"Apply BC","value":23.4},{"name":"Post-Process","value":0.5}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-14" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[68.8,50.4,50.2,9.6,5.1,1.8]},{"name":"Solve","data":[8.0,16.8,0.5,52.4,88.5,72.4]},{"name":"Apply BC","data":[1.4,6.8,0.2,20.7,2.0,23.4]},{"name":"Post-Process","data":[18.9,20.1,48.9,1.1,2.1,0.5]}],"yAxisName":"Percentage (%)"}'></div>

#### S-Bend (M) - 196,078 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (46%) | Solve (41%) |
| CPU Threaded | Solve (58%) | Assembly (26%) |
| CPU Multiprocess | Solve (80%) | Assembly (11%) |
| Numba CPU | Solve (95%) | BC (4%) |
| Numba CUDA | Solve (52%) | Assembly (26%) |
| CuPy GPU | Solve (87%) | BC (12%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-15" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":46.1},{"name":"Solve","value":41.2},{"name":"Apply BC","value":0.3},{"name":"Post-Process","value":12.4}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-16" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":0.1},{"name":"Solve","value":87.3},{"name":"Apply BC","value":12.4},{"name":"Post-Process","value":0.0}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-17" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[46.1,26.4,10.6,1.3,25.8,0.1]},{"name":"Solve","data":[41.2,58.4,80.4,94.7,52.3,87.3]},{"name":"Apply BC","data":[0.3,2.2,3.6,4.0,20.9,12.4]},{"name":"Post-Process","data":[12.4,12.9,5.4,0.0,0.1,0.0]}],"yAxisName":"Percentage (%)"}'></div>

#### S-Bend (L) - 765,441 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Solve (73%) | Assembly (21%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-18" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":21.2},{"name":"Solve","value":73.0},{"name":"Apply BC","value":0.1},{"name":"Post-Process","value":5.7}]}'></div>

#### S-Bend (XL) - 1,286,039 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Solve (97%) | Assembly (2%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-19" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":2.0},{"name":"Solve","value":97.4},{"name":"Apply BC","value":0.0},{"name":"Post-Process","value":0.5}]}'></div>

#### T-Junction (XS) - 393 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (71%) | Post-Proc (19%) |
| CPU Threaded | Assembly (53%) | Post-Proc (24%) |
| CPU Multiprocess | Assembly (51%) | Post-Proc (48%) |
| Numba CPU | Solve (59%) | BC (17%) |
| Numba CUDA | Solve (88%) | Assembly (6%) |
| CuPy GPU | Solve (74%) | BC (22%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-20" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":70.9},{"name":"Solve","value":6.9},{"name":"Apply BC","value":1.1},{"name":"Post-Process","value":19.0}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-21" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":1.5},{"name":"Solve","value":74.2},{"name":"Apply BC","value":22.1},{"name":"Post-Process","value":0.4}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-22" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[70.9,53.0,51.1,7.3,5.7,1.5]},{"name":"Solve","data":[6.9,13.4,0.4,59.1,88.4,74.2]},{"name":"Apply BC","data":[1.1,4.4,0.1,16.5,1.8,22.1]},{"name":"Post-Process","data":[19.0,23.9,48.2,1.2,1.7,0.4]}],"yAxisName":"Percentage (%)"}'></div>

#### T-Junction (M) - 196,420 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (49%) | Solve (37%) |
| CPU Threaded | Solve (57%) | Assembly (28%) |
| CPU Multiprocess | Solve (79%) | Assembly (11%) |
| Numba CPU | Solve (95%) | BC (4%) |
| Numba CUDA | Solve (51%) | Assembly (27%) |
| CuPy GPU | Solve (87%) | BC (13%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-23" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":49.4},{"name":"Solve","value":37.2},{"name":"Apply BC","value":0.3},{"name":"Post-Process","value":13.0}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-24" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":0.1},{"name":"Solve","value":87.0},{"name":"Apply BC","value":12.7},{"name":"Post-Process","value":0.0}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-25" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[49.4,27.7,11.3,1.3,26.8,0.1]},{"name":"Solve","data":[37.2,56.9,79.2,94.9,51.5,87.0]},{"name":"Apply BC","data":[0.3,2.2,3.3,3.8,20.6,12.7]},{"name":"Post-Process","data":[13.0,13.1,6.1,0.0,0.1,0.0]}],"yAxisName":"Percentage (%)"}'></div>

#### T-Junction (L) - 768,898 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Solve (90%) | Assembly (8%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-26" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":8.2},{"name":"Solve","value":89.6},{"name":"Apply BC","value":0.1},{"name":"Post-Process","value":2.2}]}'></div>

#### Venturi (XS) - 341 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (69%) | Post-Proc (19%) |
| CPU Threaded | Assembly (51%) | Post-Proc (24%) |
| CPU Multiprocess | Assembly (50%) | Post-Proc (49%) |
| Numba CPU | Solve (49%) | BC (21%) |
| Numba CUDA | Solve (87%) | Assembly (6%) |
| CuPy GPU | Solve (67%) | BC (29%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-27" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":69.0},{"name":"Solve","value":7.1},{"name":"Apply BC","value":1.4},{"name":"Post-Process","value":19.5}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-28" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":1.6},{"name":"Solve","value":66.7},{"name":"Apply BC","value":29.3},{"name":"Post-Process","value":0.5}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-29" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[69.0,50.6,50.0,10.0,5.9,1.6]},{"name":"Solve","data":[7.1,13.3,0.4,49.3,87.1,66.7]},{"name":"Apply BC","data":[1.4,5.7,0.2,20.5,2.3,29.3]},{"name":"Post-Process","data":[19.5,24.3,49.3,0.9,1.9,0.5]}],"yAxisName":"Percentage (%)"}'></div>

#### Venturi (M) - 194,325 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (48%) | Solve (39%) |
| CPU Threaded | Solve (55%) | Assembly (29%) |
| CPU Multiprocess | Solve (78%) | Assembly (12%) |
| Numba CPU | Solve (94%) | BC (5%) |
| Numba CUDA | Solve (49%) | Assembly (28%) |
| CuPy GPU | Solve (79%) | BC (21%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-30" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":47.9},{"name":"Solve","value":39.0},{"name":"Apply BC","value":0.3},{"name":"Post-Process","value":12.8}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-31" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":0.1},{"name":"Solve","value":78.6},{"name":"Apply BC","value":21.1},{"name":"Post-Process","value":0.0}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-32" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[47.9,28.7,11.9,1.5,27.9,0.1]},{"name":"Solve","data":[39.0,55.2,78.0,93.7,49.4,78.6]},{"name":"Apply BC","data":[0.3,2.5,3.8,4.8,21.5,21.1]},{"name":"Post-Process","data":[12.8,13.7,6.3,0.0,0.1,0.0]}],"yAxisName":"Percentage (%)"}'></div>

#### Venturi (L) - 763,707 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Solve (71%) | Assembly (23%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-33" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":23.1},{"name":"Solve","value":70.6},{"name":"Apply BC","value":0.2},{"name":"Post-Process","value":6.2}]}'></div>

#### Venturi (XL) - 1,284,412 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Solve (81%) | Assembly (15%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-34" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":14.9},{"name":"Solve","value":81.0},{"name":"Apply BC","value":0.1},{"name":"Post-Process","value":4.0}]}'></div>

#### Y-Shaped (XS) - 201 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (63%) | Post-Proc (20%) |
| CPU Threaded | Assembly (44%) | Post-Proc (26%) |
| CPU Multiprocess | Assembly (51%) | Post-Proc (49%) |
| Numba CPU | Solve (52%) | BC (15%) |
| Numba CUDA | Solve (86%) | Assembly (6%) |
| CuPy GPU | Solve (61%) | BC (34%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-35" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":63.4},{"name":"Solve","value":9.8},{"name":"Apply BC","value":1.2},{"name":"Post-Process","value":20.3}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-36" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":1.9},{"name":"Solve","value":60.5},{"name":"Apply BC","value":34.4},{"name":"Post-Process","value":0.5}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-37" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[63.4,43.9,50.6,9.7,6.2,1.9]},{"name":"Solve","data":[9.8,15.9,0.4,51.7,86.5,60.5]},{"name":"Apply BC","data":[1.2,5.5,0.1,14.5,1.6,34.4]},{"name":"Post-Process","data":[20.3,25.6,48.7,1.1,2.2,0.5]}],"yAxisName":"Percentage (%)"}'></div>

#### Y-Shaped (M) - 195,853 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Assembly (46%) | Solve (41%) |
| CPU Threaded | Solve (59%) | Assembly (26%) |
| CPU Multiprocess | Solve (79%) | Assembly (10%) |
| Numba CPU | Solve (93%) | BC (6%) |
| Numba CUDA | Solve (48%) | BC (26%) |
| CuPy GPU | Solve (79%) | BC (20%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-38" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":46.4},{"name":"Solve","value":40.5},{"name":"Apply BC","value":0.6},{"name":"Post-Process","value":12.4}]}'></div>

<div class="echart-container pie-chart" id="analysis-pie-39" style="height:280px" data-chart='{"type":"pie","title":"CuPy GPU - Time Distribution","data":[{"name":"Assembly","value":0.1},{"name":"Solve","value":79.4},{"name":"Apply BC","value":19.5},{"name":"Post-Process","value":0.0}]}'></div>

<div class="echart-container stacked-bar-chart" id="analysis-stacked-40" style="height:320px" data-chart='{"type":"stacked-bar","title":"Time Distribution by Implementation (%)","categories":["CPU Baseline","CPU Threaded","CPU Multiprocess","Numba CPU","Numba CUDA","CuPy GPU"],"series":[{"name":"Assembly","data":[46.4,25.5,10.3,1.5,24.3,0.1]},{"name":"Solve","data":[40.5,58.9,79.3,92.8,47.7,79.4]},{"name":"Apply BC","data":[0.6,2.9,4.2,5.6,26.4,19.5]},{"name":"Post-Process","data":[12.4,12.6,6.1,0.0,0.1,0.0]}],"yAxisName":"Percentage (%)"}'></div>

#### Y-Shaped (L) - 772,069 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Solve (57%) | Assembly (34%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-41" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":33.7},{"name":"Solve","value":56.9},{"name":"Apply BC","value":0.4},{"name":"Post-Process","value":8.9}]}'></div>

#### Y-Shaped (XL) - 1,357,953 nodes

| Implementation | Primary Bottleneck | Secondary Bottleneck |
|----------------|--------------------|---------------------|
| CPU Baseline | Solve (67%) | Assembly (25%) |

**Time Distribution:**

<div class="echart-container pie-chart" id="analysis-pie-42" style="height:280px" data-chart='{"type":"pie","title":"CPU Baseline - Time Distribution","data":[{"name":"Assembly","value":25.4},{"name":"Solve","value":67.3},{"name":"Apply BC","value":0.3},{"name":"Post-Process","value":6.9}]}'></div>

### Why Each Optimization Helps

| Transition | Reason |
|------------|--------|
| Baseline → Threaded | Limited by Python GIL; threads only help for I/O |
| Threaded → Multiprocess | Bypasses GIL via separate processes; IPC overhead limits gains |
| Multiprocess → Numba CPU | JIT compilation eliminates interpreter overhead; true parallel loops |
| Numba CPU → Numba CUDA | GPU parallelism: thousands of threads vs dozens of CPU cores |
| Numba CUDA → CuPy GPU | CUDA C kernels more optimized than Numba-generated PTX |