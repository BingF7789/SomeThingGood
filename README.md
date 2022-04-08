
# Something Good.
Detailed Coming Soon...

## Requirements

* install clustering toolkit: metis and its Python interface.

  download and install metis: http://glaros.dtc.umn.edu/gkhome/metis/metis/download

  METIS - Serial Graph Partitioning and Fill-reducing Matrix Ordering ([official website](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview))

```
1) Download metis-5.1.0.tar.gz from http://glaros.dtc.umn.edu/gkhome/metis/metis/download and unpack it
2) cd metis-5.1.0
3) make config shared=1 prefix=~/.local/
4) make install
5) export METIS_DLL=~/.local/lib/libmetis.so
```

* install required Python packages

```
 pip install -r requirements.txt
```

quick test to see whether you install metis correctly:

```
>>> import networkx as nx
>>> import metis
>>> G = metis.example_networkx()
>>> (edgecuts, parts) = metis.part_graph(G, 3)
```

## Demo Data Download
```
http://snap.stanford.edu/graphsage/ppi.zip
```

Unzip into /data folder

## Run Experiments.
```
sh ./demo.sh
```
