# Esecuzione CON Self-Loop e metrica MODULARITÀ
echo Self-Loop/Modularity/Global
python3 main.py -d grid-creation/data/motoscooter_movimento_new.csv -e map-embedding/results/Map_embedding_128.pickle -m modularity -c 10
echo Self-Loop/Modularity/Weekday
python3 main.py -d grid-creation/data/motoscooter_movimento_new.csv -e map-embedding/results/Map_embedding_128.pickle -m modularity -c 10 -t weekday
echo Self-Loop/Modularity/Weekend
python3 main.py -d grid-creation/data/motoscooter_movimento_new.csv -e map-embedding/results/Map_embedding_128.pickle -m modularity -c 10 -t weekend
echo

# Esecuzione CON Self-Loop e metrica DISTANZA
echo Self-Loop/Distance/Global
python3 main.py -d grid-creation/data/motoscooter_movimento_new.csv -e map-embedding/results/Map_embedding_128.pickle -m distance -c 10
echo Self-Loop/Distance/Weekday
python3 main.py -d grid-creation/data/motoscooter_movimento_new.csv -e map-embedding/results/Map_embedding_128.pickle -m distance -c 10 -t weekday
echo Self-Loop/Distance/Weekend
python3 main.py -d grid-creation/data/motoscooter_movimento_new.csv -e map-embedding/results/Map_embedding_128.pickle -m distance -c 10 -t weekend
echo

# Esecuzione CON Self-Loop e metrica MAP-EMBEDDING
echo Self-Loop/Map-Embedding/Global
python3 main.py -d grid-creation/data/motoscooter_movimento_new.csv -e map-embedding/results/Map_embedding_128.pickle -m map_embedding -c 10
echo Self-Loop/Map-Embedding/Weekday
python3 main.py -d grid-creation/data/motoscooter_movimento_new.csv -e map-embedding/results/Map_embedding_128.pickle -m map_embedding -c 10 -t weekday
echo Self-Loop/Map-Embedding/Weekend
python3 main.py -d grid-creation/data/motoscooter_movimento_new.csv -e map-embedding/results/Map_embedding_128.pickle -m map_embedding -c 10 -t weekend
echo

# Esecuzione SENZA Self-Loop e metrica MODULARITÀ
echo NO Self-Loop/Modularity/Global
python3 main.py -d grid-creation/data/motoscooter_movimento_new.csv -e map-embedding/results/Map_embedding_128.pickle -m modularity -c 10 --no_selfloop
echo NO Self-Loop/Modularity/Weekday
python3 main.py -d grid-creation/data/motoscooter_movimento_new.csv -e map-embedding/results/Map_embedding_128.pickle -m modularity -c 10 --no_selfloop -t weekday
echo NO Self-Loop/Modularity/Weekend
python3 main.py -d grid-creation/data/motoscooter_movimento_new.csv -e map-embedding/results/Map_embedding_128.pickle -m modularity -c 10 --no_selfloop -t weekend
echo

# Esecuzione SENZA Self-Loop e metrica DISTANZA
echo NO Self-Loop/Distance/Global
python3 main.py -d grid-creation/data/motoscooter_movimento_new.csv -e map-embedding/results/Map_embedding_128.pickle -m distance -c 10 --no_selfloop
echo NO Self-Loop/Distance/Weekday
python3 main.py -d grid-creation/data/motoscooter_movimento_new.csv -e map-embedding/results/Map_embedding_128.pickle -m distance -c 10 --no_selfloop -t weekday
echo NO Self-Loop/Distance/Weekend
python3 main.py -d grid-creation/data/motoscooter_movimento_new.csv -e map-embedding/results/Map_embedding_128.pickle -m distance -c 10 --no_selfloop -t weekend
echo

# Esecuzione SENZA Self-Loop e metrica MAP-EMBEDDING
echo NO Self-Loop/Map-Embedding/Global
python3 main.py -d grid-creation/data/motoscooter_movimento_new.csv -e map-embedding/results/Map_embedding_128.pickle -m map_embedding -c 10 --no_selfloop
echo NO Self-Loop/Map-Embedding/Weekday
python3 main.py -d grid-creation/data/motoscooter_movimento_new.csv -e map-embedding/results/Map_embedding_128.pickle -m map_embedding -c 10 --no_selfloop -t weekday
echo NO Self-Loop/Map-Embedding/Weekend
python3 main.py -d grid-creation/data/motoscooter_movimento_new.csv -e map-embedding/results/Map_embedding_128.pickle -m map_embedding -c 10 --no_selfloop -t weekend
echo
